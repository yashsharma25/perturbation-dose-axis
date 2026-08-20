#!/usr/bin/env python
"""
Measure the floor and ceiling of achievable performance within each dose bin.

No models. The question this answers:

  Is there ROOM for a model to differentiate across dose bins, or does the
  measurable signal itself collapse at low knockdown?

Three curves per bin:

  ceiling  split-half correlation of true logFC. The score a perfect model
           would get given the noise actually present in that bin.
  floor    correlation of the global-perturbation-mean prediction with true
           logFC. What you get for knowing nothing about perturbation identity.
  effect   L2 norm of true logFC. How large the real displacement is.

Reading:

  ceiling collapses at low dose  -> no room; degradation is signal strength;
                                    every model will fall and the finding is
                                    modest.
  ceiling flat, gap stays wide   -> room exists; whether models exploit it is
                                    the empirical question worth GPU hours.

Everything is computed on per-cell count FRACTIONS, not raw counts, because
guide UMI correlates with sequencing depth and depth inflates raw counts.

Use --bin_on pct_counts_mt for the sham axis: a cell-quality variable with no
causal link to knockdown. If the envelope degrades the same way there, the
"dose effect" is a quality artifact.
"""

import argparse
import time

import numpy as np
import pandas as pd
import h5py
import anndata
from scipy import stats


class BackedCSRReader:
    """Direct h5py access to a backed CSR matrix, bypassing AnnData overhead."""

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


def accumulate_fractions(reader, rows, n_genes, also_count_detections=False):
    """Sum per-cell count fractions over a set of rows."""
    total = np.zeros(n_genes, dtype=np.float64)
    detections = np.zeros(n_genes, dtype=np.int64) if also_count_detections else None
    n_used = 0
    for row in rows:
        columns, values = reader.row(int(row))
        depth = values.sum()
        if depth <= 0:
            continue
        total[columns] += values / depth
        if also_count_detections:
            detections[columns] += 1
        n_used += 1
    if n_used == 0:
        raise ValueError("No usable cells in accumulation.")
    return total / n_used, detections, n_used


def load_dose(guide_calls_path, dose_column):
    calls = pd.read_csv(guide_calls_path)
    keep = calls["pass_guide_filter"].astype(str).str.lower().isin({"true", "1"})
    calls = calls[keep]
    umis = calls["num_umis"].astype(str).str.split("|").map(
        lambda parts: np.array([float(p) for p in parts])
    )
    aggregations = {
        "dose_sum": umis.map(np.sum),
        "dose_min": umis.map(np.min),
        "dose_geomean": umis.map(lambda u: float(np.exp(np.log(u).mean()))),
    }
    return pd.DataFrame(
        {"gene_target": calls["gene_target"].astype(str).to_numpy(),
         "dose": aggregations[dose_column].to_numpy()},
        index=calls["cell_barcode"].to_numpy(),
    )


def residualize(dose_values, depth_values):
    log_dose, log_depth = np.log1p(dose_values), np.log1p(depth_values)
    slope, intercept = np.polyfit(log_depth, log_dose, 1)
    return log_dose - (slope * log_depth + intercept)


def log_fold_change(mean_vector, control_vector, epsilon):
    return np.log((mean_vector + epsilon) / (control_vector + epsilon))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", required=True)
    parser.add_argument("--guide_calls", required=True)
    parser.add_argument("--dose_column", default="dose_min",
                        choices=["dose_sum", "dose_min", "dose_geomean"])
    parser.add_argument("--bin_on", default="dose", choices=["dose", "pct_counts_mt"],
                        help="'pct_counts_mt' runs the sham axis control.")
    parser.add_argument("--no_residualize", action="store_true")
    parser.add_argument("--n_bins", type=int, default=4)
    parser.add_argument("--min_cells_per_perturbation", type=int, default=80,
                        help="Needs n_bins * 2 halves * a few cells each.")
    parser.add_argument("--max_cells_per_perturbation", type=int, default=240)
    parser.add_argument("--max_perturbations", type=int, default=120)
    parser.add_argument("--n_control_cells", type=int, default=2000)
    parser.add_argument("--min_detection_rate", type=float, default=0.05,
                        help="Drop genes detected in fewer than this fraction of "
                             "control cells. Shared zeros inflate correlation.")
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="dose_envelope.csv")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    adata = anndata.read_h5ad(args.h5ad, backed="r")
    n_genes = adata.n_vars
    barcode_to_row = pd.Series(np.arange(adata.n_obs), index=adata.obs_names)
    print(f"[load] h5ad: {adata.n_obs:,} cells x {n_genes:,} genes")

    dose = load_dose(args.guide_calls, args.dose_column)
    shared = dose.index.intersection(adata.obs_names)
    dose = dose.loc[shared]
    print(f"[join] {len(shared):,} barcodes matched")

    if args.bin_on == "pct_counts_mt":
        dose["binning_variable"] = adata.obs.loc[shared, "pct_counts_mt"].to_numpy(float)
        print("[bins] SHAM AXIS: binning on mitochondrial fraction")
    else:
        dose["binning_variable"] = dose["dose"].to_numpy(float)

    reader = BackedCSRReader(args.h5ad)
    try:
        control_barcodes = dose.index[dose["gene_target"] == "Non-Targeting"]
        control_rows = barcode_to_row.reindex(control_barcodes).dropna().astype(int).to_numpy()
        control_rows = np.sort(rng.choice(control_rows,
                                          size=min(args.n_control_cells, len(control_rows)),
                                          replace=False))
        print(f"[control] reading {len(control_rows)} non-targeting cells...", flush=True)
        started = time.time()
        control_mean, detections, n_control = accumulate_fractions(
            reader, control_rows, n_genes, also_count_detections=True
        )
        print(f"[control] done in {time.time() - started:.0f}s")

        gene_mask = (detections / n_control) >= args.min_detection_rate
        print(f"[genes] {gene_mask.sum():,} of {n_genes:,} genes detected in "
              f">= {args.min_detection_rate:.0%} of controls")

        perturbed = dose[dose["gene_target"] != "Non-Targeting"]
        sample_rows = barcode_to_row.reindex(
            rng.choice(perturbed.index, size=min(args.n_control_cells, len(perturbed)),
                       replace=False)
        ).dropna().astype(int).to_numpy()
        print(f"[floor] reading {len(sample_rows)} perturbed cells for the "
              f"global mean...", flush=True)
        global_perturbed_mean, _, _ = accumulate_fractions(reader, np.sort(sample_rows), n_genes)

        control_masked = control_mean[gene_mask]
        floor_prediction = log_fold_change(global_perturbed_mean[gene_mask],
                                           control_masked, args.epsilon)

        counts = perturbed.groupby("gene_target").size()
        eligible = counts[counts >= args.min_cells_per_perturbation].index.tolist()
        if len(eligible) > args.max_perturbations:
            eligible = sorted(rng.choice(eligible, size=args.max_perturbations, replace=False))
        print(f"[scan] {len(eligible)} perturbations with "
              f">= {args.min_cells_per_perturbation} cells\n", flush=True)

        records = []
        started = time.time()
        for position, gene in enumerate(eligible, start=1):
            cells = perturbed[perturbed["gene_target"] == gene]
            rows = barcode_to_row.reindex(cells.index).dropna().astype(int)
            if len(rows) > args.max_cells_per_perturbation:
                rows = rows.sample(args.max_cells_per_perturbation,
                                   random_state=int(rng.integers(1 << 31)))
            cells = cells.loc[rows.index]

            values = cells["binning_variable"].to_numpy(float)
            if args.bin_on == "dose" and not args.no_residualize:
                depth = adata.obs["total_counts"].reindex(cells.index).to_numpy(float)
                values = residualize(values, depth)

            try:
                bins = pd.qcut(values, args.n_bins, labels=False, duplicates="drop")
            except ValueError:
                continue
            if np.nanmax(bins) < args.n_bins - 1:
                continue

            row_values = rows.to_numpy()
            for bin_id in range(args.n_bins):
                in_bin = np.flatnonzero(bins == bin_id)
                if len(in_bin) < 6:
                    continue
                shuffled = rng.permutation(in_bin)
                half_a, half_b = shuffled[: len(shuffled) // 2], shuffled[len(shuffled) // 2:]

                try:
                    mean_a, _, _ = accumulate_fractions(reader, np.sort(row_values[half_a]), n_genes)
                    mean_b, _, _ = accumulate_fractions(reader, np.sort(row_values[half_b]), n_genes)
                except ValueError:
                    continue

                logfc_a = log_fold_change(mean_a[gene_mask], control_masked, args.epsilon)
                logfc_b = log_fold_change(mean_b[gene_mask], control_masked, args.epsilon)
                logfc_full = 0.5 * (logfc_a + logfc_b)

                ceiling = stats.pearsonr(logfc_a, logfc_b).statistic
                floor = stats.pearsonr(floor_prediction, logfc_full).statistic

                records.append({
                    "gene_target": gene,
                    "bin": bin_id,
                    "n_cells": len(in_bin),
                    "ceiling_split_half": float(ceiling),
                    "ceiling_corrected": float(2 * ceiling / (1 + ceiling))
                    if ceiling > -1 else np.nan,
                    "floor_global_mean": float(floor),
                    "effect_size": float(np.linalg.norm(logfc_full)),
                })

            if position % 10 == 0:
                rate = position / (time.time() - started)
                print(f"[scan] {position}/{len(eligible)}  {rate:.2f} pert/s  "
                      f"~{(len(eligible) - position) / max(rate, 1e-9) / 60:.1f} min left",
                      flush=True)
    finally:
        reader.close()

    envelope = pd.DataFrame(records)
    envelope.to_csv(args.out, index=False)
    print(f"\n[save] {args.out}  ({len(envelope)} perturbation-bin rows)")

    per_bin = envelope.groupby("bin").agg(
        ceiling=("ceiling_corrected", "median"),
        floor=("floor_global_mean", "median"),
        effect_size=("effect_size", "median"),
        cells=("n_cells", "median"),
        perturbations=("gene_target", "nunique"),
    )
    per_bin["room"] = per_bin["ceiling"] - per_bin["floor"]

    label = "mitochondrial fraction (SHAM)" if args.bin_on == "pct_counts_mt" else "dose"
    print(f"\n[envelope] binned on {label}; bin 0 = lowest\n")
    print(per_bin.to_string(float_format=lambda v: f"{v:.4f}"))

    ceiling_drop = per_bin["ceiling"].iloc[0] - per_bin["ceiling"].iloc[-1]
    room_low, room_high = per_bin["room"].iloc[0], per_bin["room"].iloc[-1]
    print(f"\n[read] ceiling change across bins: {ceiling_drop:+.4f}")
    print(f"[read] room in lowest bin: {room_low:.4f}   highest bin: {room_high:.4f}")

    if room_low < 0.15:
        print("\n[verdict] Little room in the low bin. Any model will degrade there. "
              "The finding is signal-strength, not a model failure. Modest paper.")
    elif abs(ceiling_drop) < 0.10:
        print("\n[verdict] Ceiling roughly flat with room throughout. Models CAN "
              "differentiate across bins. Worth the GPU hours.")
    else:
        print("\n[verdict] Ceiling moves but room persists. Report models against "
              "the ceiling, not raw. Normalised score is the headline.")


if __name__ == "__main__":
    main()