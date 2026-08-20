#!/usr/bin/env python
"""
Kalygina-style reliability screen (NSR OR rho), reconciled against the frozen
evaluation set and the dose axis.

WHY THIS SCRIPT EXISTS
Kalygina et al. (ICML 2026 workshop) retain 8,035 of 35,916 X-Atlas/Orion
condition units (22.4%) under a two-axis reliability rule. Our own effect-size
screen retained ~4%. Before adopting their inclusion criterion we need to know
whether the difference is a threshold choice or a different phenomenon -- and,
much more importantly, how many perturbations survive that ALSO sit in the
frozen evaluation set AND have enough cells in both dose bins.

That last count is the effective sample size of the paper. Global retention is
irrelevant if model inference only covers the frozen set.

THE TWO AXES (their Appendix H)

  Detectability.  NSR = ||y|| / se, where se^2 = V_p/n_p + V_ctrl/n_ctrl and V
  is the total per-cell variance (trace of the covariance). A relabelled control
  sits at NSR ~ 1; a strong effect is well above it.

  Reproducibility.  rho = across-gene correlation between responses estimated on
  two independent halves of the condition's cells. This measures whether the
  DIRECTION pattern reproduces, independent of magnitude, and rescues weak but
  consistently directed effects that detectability alone discards.

  Keep if EITHER clears the 95th percentile of a control-split null. Only the
  joint low-detectability, low-reproducibility corner is dropped.

Both thresholds are calibrated per context against pseudo-perturbations drawn
from control cells -- size-matched relabelled controls carrying pure technical
noise and zero effect, so their 95th percentile is a 5% false-positive cutoff.

SELECTION-HALF ONLY. The screen runs on selection cells so that adopting a
published inclusion criterion does not become a selection-on-evaluation leak.
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import h5py
import anndata as ad

from dose_binning import compute_dose_bins


class BackedCSRReader:
    def __init__(self, h5ad_path):
        self.handle = h5py.File(h5ad_path, "r")
        group = self.handle["X"]
        self.indptr = group["indptr"][:]
        self.indices = group["indices"]
        self.data = group["data"]

    def row(self, index):
        lo, hi = int(self.indptr[index]), int(self.indptr[index + 1])
        if hi <= lo:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        return self.indices[lo:hi], self.data[lo:hi]

    def close(self):
        self.handle.close()


def read_normalised(reader, rows, n_genes, target_sum=1e4):
    """Cells as log1p counts-per-target_sum, matching their preprocessing."""
    block = np.zeros((len(rows), n_genes), dtype=np.float32)
    kept = 0
    for position, row in enumerate(rows):
        columns, values = reader.row(int(row))
        depth = values.sum()
        if depth <= 0:
            continue
        block[kept, columns] = values * (target_sum / depth)
        kept += 1
    return np.log1p(block[:kept])


def response_and_variance(perturbed, control_mean, control_variance, n_control):
    """Pseudobulk response, its standard error, and the resulting NSR.

    Under the isotropic squared-Euclidean geometry only variance traces enter --
    never a full covariance matrix -- so the standard error collapses to two
    scalars: se^2 = V_p/n_p + V_ctrl/n_ctrl.
    """
    n_perturbed = perturbed.shape[0]
    response = perturbed.mean(axis=0) - control_mean
    perturbed_variance = float(np.sum(perturbed.var(axis=0, ddof=1))) \
        if n_perturbed > 1 else 0.0
    if n_control <= 0:
        raise ValueError("n_control is zero -- no control cells reached the "
                         "variance estimate.")
    squared_error = perturbed_variance / n_perturbed + control_variance / n_control
    standard_error = float(np.sqrt(max(squared_error, 1e-12)))
    return response, standard_error, float(np.linalg.norm(response)) / standard_error


def split_half_rho(perturbed, control_mean, rng):
    """Across-gene correlation of responses from two independent cell halves.

    Direction agreement, not magnitude agreement. A relabelled control is a
    reproducibly NULL measurement, so a magnitude-disagreement rule would keep
    fakes; a direction rule drops them.
    """
    order = rng.permutation(perturbed.shape[0])
    midpoint = len(order) // 2
    if midpoint < 2:
        return np.nan
    first = perturbed[order[:midpoint]].mean(axis=0) - control_mean
    second = perturbed[order[midpoint:]].mean(axis=0) - control_mean
    if first.std() < 1e-12 or second.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(first, second)[0, 1])


def calibrate_null(reader, control_rows, n_genes, control_mean, control_variance,
                   cell_counts, n_draws, rng):
    """Relabelled-control pseudo-perturbations: pure noise, zero effect."""
    null_nsr, null_rho = [], []
    for _ in range(n_draws):
        size = int(rng.choice(cell_counts))
        size = min(size, len(control_rows) // 2)
        if size < 4:
            continue
        drawn = rng.choice(control_rows, size=size, replace=False)
        block = read_normalised(reader, np.sort(drawn), n_genes)
        if block.shape[0] < 4:
            continue
        _, _, nsr = response_and_variance(block, control_mean, control_variance,
                                          len(control_rows))
        null_nsr.append(nsr)
        null_rho.append(split_half_rho(block, control_mean, rng))
    null_rho = [v for v in null_rho if np.isfinite(v)]
    return (float(np.quantile(null_nsr, 0.95)) if null_nsr else 1.0,
            float(np.quantile(null_rho, 0.95)) if null_rho else 0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", required=True)
    parser.add_argument("--guide_calls", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--eval_set", required=True,
                        help="The frozen 200. Reconciliation is against this.")
    parser.add_argument("--old_screen", default=None,
                        help="screen_*_ranked.csv, to compare old vs new selection.")
    parser.add_argument("--dose_bins", type=int, default=2)
    parser.add_argument("--min_cells_per_bin", type=int, default=30,
                        help="Cells required in EVERY dose bin, evaluation half.")
    parser.add_argument("--min_cells_screen", type=int, default=12)
    parser.add_argument("--max_cells_screen", type=int, default=200)
    parser.add_argument("--max_perturbations", type=int, default=2000)
    parser.add_argument("--n_control_cells", type=int, default=3000)
    parser.add_argument("--n_null_draws", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="nsr_rho_screen")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    adata = ad.read_h5ad(args.h5ad, backed="r")
    n_genes = int(adata.n_vars)
    obs = adata.obs
    barcode_index = pd.Index(obs.index.to_numpy())
    row_of = pd.Series(np.arange(adata.n_obs), index=barcode_index)
    print(f"[load] {adata.n_obs:,} cells x {n_genes:,} genes", flush=True)

    frozen = set(pd.read_csv(args.eval_set)["gene_target"].astype(str))
    split = pd.read_csv(args.split)
    selection = set(split.loc[split["split"] == "selection", "cell_barcode"])
    evaluation = set(split.loc[split["split"] == "evaluation", "cell_barcode"])
    print(f"[load] frozen eval set: {len(frozen)} perturbations")
    print(f"[load] {len(selection):,} selection / {len(evaluation):,} evaluation cells")

    dose_bins = compute_dose_bins(obs, args.guide_calls, n_bins=args.dose_bins,
                                  min_cells_per_perturbation=2 * args.min_cells_per_bin,
                                  verbose=False)
    genes = obs["gene_target"].astype(str).to_numpy()
    guides = obs["guide_target"].astype(str).to_numpy()
    is_control = np.array([len(g.split("|")) >= 2
                           and all(p.startswith("non-targeting") for p in g.split("|"))
                           for g in guides])
    in_selection = barcode_index.isin(selection)
    in_evaluation = barcode_index.isin(evaluation)

    reader = BackedCSRReader(args.h5ad)
    try:
        control_rows = np.where(is_control & in_selection)[0]
        if len(control_rows) == 0:
            # screen_perturbations.py only split PERTURBED cells, so controls
            # carry no split label. Assign them deterministically here so the
            # reference mean and the null calibration stay on selection cells.
            all_controls = np.where(is_control)[0]
            if len(all_controls) == 0:
                raise RuntimeError("No control cells found under the dual "
                                   "non-targeting rule -- check guide_target.")
            shuffled = rng.permutation(all_controls)
            control_rows = np.sort(shuffled[: len(shuffled) // 2])
            print(f"[control] split file has no controls; assigning "
                  f"{len(control_rows):,} of {len(all_controls):,} to the "
                  f"selection half by seed")
        control_rows = np.sort(rng.choice(control_rows,
                                          size=min(args.n_control_cells, len(control_rows)),
                                          replace=False))
        print(f"[control] reading {len(control_rows):,} selection-half controls...",
              flush=True)
        started = time.time()
        control_block = read_normalised(reader, control_rows, n_genes)
        control_mean = control_block.mean(axis=0)
        control_variance = float(np.sum(control_block.var(axis=0, ddof=1)))
        del control_block
        print(f"[control] done in {time.time() - started:.0f}s "
              f"(total per-cell variance {control_variance:.1f})")

        candidates = pd.Series(genes[in_selection & ~is_control]).value_counts()
        candidates = candidates[candidates >= args.min_cells_screen]
        candidates = candidates.drop(labels=["Non-Targeting"], errors="ignore")
        if len(candidates) > args.max_perturbations:
            keep = list(frozen & set(candidates.index))
            others = [g for g in candidates.index if g not in frozen]
            n_extra = max(0, args.max_perturbations - len(keep))
            keep += list(rng.choice(others, size=min(n_extra, len(others)), replace=False))
            candidates = candidates.loc[keep]
        print(f"[screen] {len(candidates):,} perturbations with "
              f">= {args.min_cells_screen} selection cells "
              f"({len(set(candidates.index) & frozen)} of them frozen)")

        print(f"[null] calibrating against {args.n_null_draws} relabelled controls...",
              flush=True)
        started = time.time()
        nsr_cutoff, rho_cutoff = calibrate_null(
            reader, control_rows, n_genes, control_mean, control_variance,
            candidates.to_numpy(), args.n_null_draws, rng)
        print(f"[null] done in {time.time() - started:.0f}s -- "
              f"NSR p95 = {nsr_cutoff:.3f}, rho p95 = {rho_cutoff:.3f}")

        records = []
        started = time.time()
        for position, gene in enumerate(candidates.index, start=1):
            rows = np.where((genes == gene) & in_selection & ~is_control)[0]
            if len(rows) > args.max_cells_screen:
                rows = rng.choice(rows, size=args.max_cells_screen, replace=False)
            block = read_normalised(reader, np.sort(rows), n_genes)
            if block.shape[0] < args.min_cells_screen:
                continue
            _, standard_error, nsr = response_and_variance(
                block, control_mean, control_variance, len(control_rows))
            rho = split_half_rho(block, control_mean, rng)

            evaluation_mask = (genes == gene) & in_evaluation & ~is_control
            bins_here = dose_bins[evaluation_mask]
            per_bin = [int((bins_here == b).sum()) for b in range(args.dose_bins)]

            records.append({
                "gene_target": gene,
                "nsr": nsr,
                "rho": rho,
                "standard_error": standard_error,
                "n_selection_cells": int(block.shape[0]),
                "n_evaluation_cells": int(evaluation_mask.sum()),
                "min_cells_in_a_bin": min(per_bin) if per_bin else 0,
                "in_frozen": gene in frozen,
            })
            if position % 100 == 0:
                rate = position / max(time.time() - started, 1e-9)
                print(f"[screen] {position:,}/{len(candidates):,}  "
                      f"~{(len(candidates) - position) / max(rate, 1e-9) / 60:.0f} min left",
                      flush=True)
    finally:
        reader.close()

    screen = pd.DataFrame(records)
    screen["passes_nsr"] = screen["nsr"] > nsr_cutoff
    screen["passes_rho"] = screen["rho"] > rho_cutoff
    screen["retained"] = screen["passes_nsr"] | screen["passes_rho"]
    screen["rho_rescued"] = screen["passes_rho"] & ~screen["passes_nsr"]
    screen["dose_valid"] = screen["min_cells_in_a_bin"] >= args.min_cells_per_bin
    screen.to_csv(f"{args.out}.csv", index=False)

    total = len(screen)
    frozen_rows = screen[screen["in_frozen"]]
    paper = screen[screen["in_frozen"] & screen["retained"] & screen["dose_valid"]]

    print(f"\n{'=' * 62}")
    print("Kalygina-style screen (NSR OR rho)")
    print("=" * 62)
    print(f"  cutoffs: NSR > {nsr_cutoff:.3f}, rho > {rho_cutoff:.3f}\n")
    print(f"  screened:                      {total:,}")
    print(f"  passes NSR:                    {int(screen['passes_nsr'].sum()):,}")
    print(f"  passes rho:                    {int(screen['passes_rho'].sum()):,}")
    print(f"  passes NSR OR rho:             {int(screen['retained'].sum()):,} "
          f"({screen['retained'].mean():.1%})")
    print(f"  rho-rescued:                   {int(screen['rho_rescued'].sum()):,}")
    print("\n  Kalygina report 22.4% retention on Orion (8,035 of 35,916).")

    print(f"\n  frozen eval set screened:      {len(frozen_rows):,}")
    print(f"    passes NSR OR rho:           {int(frozen_rows['retained'].sum()):,} "
          f"({frozen_rows['retained'].mean():.1%})")
    print(f"    dose-valid (>= {args.min_cells_per_bin} cells in every bin): "
          f"{int(frozen_rows['dose_valid'].sum()):,}")

    print(f"\n  *** EFFECTIVE SAMPLE SIZE: {len(paper):,} ***")
    print("      frozen AND passes screen AND dose-valid.")
    print("      This is the n every model-arm comparison is powered by --")
    print("      not the global retention count.")

    if len(paper) >= 10:
        print(f"\n  At the PCA pilot's gap SD of 0.596, n={len(paper)} resolves an "
              f"effect of\n  {2.8 * 0.596 / np.sqrt(len(paper)):.3f} at 80% power "
              f"(ground-truth dose effect ~0.073).")

    if args.old_screen:
        old = pd.read_csv(args.old_screen)
        old_strong = set(old.loc[old["effect_size_unbiased"] >= 2.5, "gene_target"])
        new_kept = set(screen.loc[screen["retained"], "gene_target"])
        print(f"\n  old effect-size screen (>= 2.5): {len(old_strong):,}")
        print(f"    old AND new:                   {len(old_strong & new_kept):,}")
        print(f"    old only:                      {len(old_strong - new_kept):,}")
        print(f"    new only:                      {len(new_kept - old_strong):,}")

    with open(f"{args.out}_summary.json", "w") as handle:
        json.dump({
            "nsr_cutoff": nsr_cutoff, "rho_cutoff": rho_cutoff,
            "screened": total, "retained": int(screen["retained"].sum()),
            "retained_fraction": float(screen["retained"].mean()),
            "rho_rescued": int(screen["rho_rescued"].sum()),
            "frozen_screened": len(frozen_rows),
            "frozen_retained": int(frozen_rows["retained"].sum()),
            "effective_sample_size": len(paper),
            "config": vars(args),
        }, handle, indent=2)
    print(f"\n[save] {args.out}.csv, {args.out}_summary.json")


if __name__ == "__main__":
    main()