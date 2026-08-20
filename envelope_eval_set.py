#!/usr/bin/env python
"""
Floor and ceiling within dose bins, on the frozen evaluation set.

The decision point for the project. No models involved.

  ceiling  split-half correlation of true logFC within a bin. The score a
           perfect model would get given the noise actually present.
  floor    correlation of a global-perturbation-mean prediction with true
           logFC. What you get for knowing nothing about which gene was hit.
  room     ceiling - floor. The span a model has available to demonstrate
           anything at all.

Room persisting in the low-dose bin means dose stratification can distinguish
models, and the C2S re-run is justified. Room collapsing means the degradation
is signal-strength rather than model failure, and the honest finding is smaller.

Three changes over the first version:

  SPLIT-AWARE. Reads only cells assigned to the evaluation half by the screen.
  Perturbation selection used the other half, so nothing here is circular.

  NOISE-MATCHED FLOOR. Previously the ceiling correlated two half-bins while
  the floor correlated a 2000-cell global mean against a full-bin estimate --
  so the floor was measured at far lower noise and could exceed the ceiling.
  Both are now computed against the same half-bin estimates.

  STRATUM-AWARE. Results break out by effect-size stratum, because the whole
  point is to show where the dose axis works and where it does not.
"""

import argparse
import time

import numpy as np
import pandas as pd
import h5py
import anndata
from scipy import stats


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


def mean_count_fraction(reader, rows, n_genes, count_detections=False):
    total = np.zeros(n_genes, dtype=np.float64)
    detections = np.zeros(n_genes, dtype=np.int64) if count_detections else None
    n_used = 0
    for row in rows:
        columns, values = reader.row(int(row))
        depth = values.sum()
        if depth <= 0:
            continue
        total[columns] += values / depth
        if count_detections:
            detections[columns] += 1
        n_used += 1
    if n_used == 0:
        return None, detections, 0
    return total / n_used, detections, n_used


def log_fold_change(mean_vector, control_vector, epsilon):
    return np.log((mean_vector + epsilon) / (control_vector + epsilon))


def residualize(dose_values, depth_values):
    """Strip the depth-driven component out of dose.

    Guide UMI correlates with sequencing depth (rho ~ 0.25). Binning on the raw
    number shuffles cells across bin boundaries by depth rather than by dose,
    which attenuates the gradient -- classic regression dilution.
    """
    log_dose, log_depth = np.log1p(dose_values), np.log1p(depth_values)
    slope, intercept = np.polyfit(log_depth, log_dose, 1)
    return log_dose - (slope * log_depth + intercept)


def load_dose(guide_calls_path, dose_column):
    calls = pd.read_csv(guide_calls_path)
    calls = calls[calls["pass_guide_filter"].astype(str).str.lower().isin({"true", "1"})]
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", required=True)
    parser.add_argument("--guide_calls", required=True)
    parser.add_argument("--eval_set", required=True, help="from build_eval_set.py")
    parser.add_argument("--split", required=True, help="{out}_split.csv from the screen")
    parser.add_argument("--dose_column", default="dose_min",
                        choices=["dose_sum", "dose_min", "dose_geomean"])
    parser.add_argument("--bin_on", default="dose", choices=["dose", "pct_counts_mt"],
                        help="'pct_counts_mt' runs the sham axis control.")
    parser.add_argument("--no_residualize", action="store_true")
    parser.add_argument("--n_bins", type=int, default=3)
    parser.add_argument("--n_control_cells", type=int, default=3000)
    parser.add_argument("--min_detection_rate", type=float, default=0.25)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="envelope_eval.csv")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    adata = anndata.read_h5ad(args.h5ad, backed="r")
    n_genes = adata.n_vars
    barcode_to_row = pd.Series(np.arange(adata.n_obs), index=adata.obs_names)
    print(f"[load] h5ad: {adata.n_obs:,} cells x {n_genes:,} genes")

    eval_set = pd.read_csv(args.eval_set)
    stratum_of = dict(zip(eval_set["gene_target"], eval_set["stratum"]))
    print(f"[load] eval set: {len(eval_set)} perturbations")

    split = pd.read_csv(args.split)
    evaluation_barcodes = set(
        split.loc[split["split"] == "evaluation", "cell_barcode"]
    )
    print(f"[load] {len(evaluation_barcodes):,} cells in the evaluation half")

    dose = load_dose(args.guide_calls, args.dose_column)
    dose = dose.loc[dose.index.intersection(adata.obs_names)]

    if args.bin_on == "pct_counts_mt":
        dose["binning_variable"] = adata.obs.loc[dose.index, "pct_counts_mt"].to_numpy(float)
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
        print(f"[control] reading {len(control_rows):,} non-targeting cells...", flush=True)
        started = time.time()
        control_mean, detections, n_control = mean_count_fraction(
            reader, control_rows, n_genes, count_detections=True
        )
        print(f"[control] done in {time.time() - started:.0f}s")

        gene_mask = (detections / n_control) >= args.min_detection_rate
        control_masked = control_mean[gene_mask]
        print(f"[genes] {gene_mask.sum():,} of {n_genes:,} genes retained")

        # Global perturbation mean, built only from evaluation-half cells of
        # eval-set perturbations, so the floor lives in the same universe as
        # the ceiling.
        in_eval_set = dose["gene_target"].isin(stratum_of)
        pool = dose.index[in_eval_set & dose.index.isin(evaluation_barcodes)]
        global_rows = barcode_to_row.reindex(
            rng.choice(pool, size=min(args.n_control_cells, len(pool)), replace=False)
        ).dropna().astype(int).to_numpy()
        print(f"[floor] reading {len(global_rows):,} perturbed cells...", flush=True)
        global_mean, _, _ = mean_count_fraction(reader, np.sort(global_rows), n_genes)
        floor_prediction = log_fold_change(global_mean[gene_mask], control_masked, args.epsilon)

        records = []
        started = time.time()
        for position, gene in enumerate(sorted(stratum_of), start=1):
            cells = dose[(dose["gene_target"] == gene)
                         & dose.index.isin(evaluation_barcodes)]
            if len(cells) < args.n_bins * 12:
                continue

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

            rows = barcode_to_row.reindex(cells.index).dropna().astype(int).to_numpy()
            for bin_id in range(args.n_bins):
                in_bin = np.flatnonzero(bins == bin_id)
                if len(in_bin) < 12:
                    continue
                shuffled = rng.permutation(in_bin)
                half_a, half_b = shuffled[: len(shuffled) // 2], shuffled[len(shuffled) // 2:]

                mean_a, _, n_a = mean_count_fraction(reader, np.sort(rows[half_a]), n_genes)
                mean_b, _, n_b = mean_count_fraction(reader, np.sort(rows[half_b]), n_genes)
                if mean_a is None or mean_b is None:
                    continue

                logfc_a = log_fold_change(mean_a[gene_mask], control_masked, args.epsilon)
                logfc_b = log_fold_change(mean_b[gene_mask], control_masked, args.epsilon)

                ceiling = float(stats.pearsonr(logfc_a, logfc_b).statistic)
                # Noise-matched: floor scored against each half separately, then
                # averaged, so both statistics see the same sample size.
                floor = 0.5 * (
                    float(stats.pearsonr(floor_prediction, logfc_a).statistic)
                    + float(stats.pearsonr(floor_prediction, logfc_b).statistic)
                )

                records.append({
                    "gene_target": gene,
                    "stratum": stratum_of[gene],
                    "bin": bin_id,
                    "n_cells": len(in_bin),
                    "ceiling_raw": ceiling,
                    "ceiling_corrected": 2 * ceiling / (1 + ceiling) if ceiling > -1 else np.nan,
                    "floor": floor,
                    "effect_size": float(np.sqrt(max(np.dot(logfc_a, logfc_b), 0.0))),
                })

            if position % 20 == 0:
                rate = position / (time.time() - started)
                print(f"[scan] {position}/{len(stratum_of)}  {rate:.2f} pert/s  "
                      f"~{(len(stratum_of) - position) / max(rate, 1e-9) / 60:.0f} min left",
                      flush=True)
    finally:
        reader.close()

    envelope = pd.DataFrame(records)
    envelope.to_csv(args.out, index=False)
    print(f"\n[save] {args.out}  ({len(envelope)} perturbation-bin rows)")

    label = "mitochondrial fraction (SHAM)" if args.bin_on == "pct_counts_mt" else "dose"
    print(f"\n[envelope] binned on {label}; bin 0 = lowest\n")

    for stratum in ["strong", "medium", "weak", "near_null"]:
        subset = envelope[envelope["stratum"] == stratum]
        if len(subset) == 0:
            continue
        table = subset.groupby("bin").agg(
            ceiling=("ceiling_corrected", "median"),
            floor=("floor", "median"),
            effect_size=("effect_size", "median"),
            cells=("n_cells", "median"),
            n=("gene_target", "nunique"),
        )
        table["room"] = table["ceiling"] - table["floor"]
        print(f"--- {stratum} ---")
        print(table.to_string(float_format=lambda v: f"{v:.4f}"))
        print()

    strong = envelope[envelope["stratum"] == "strong"]
    if len(strong):
        by_bin = strong.groupby("bin").agg(
            ceiling=("ceiling_corrected", "median"), floor=("floor", "median")
        )
        room = by_bin["ceiling"] - by_bin["floor"]
        print(f"[read] strong stratum room -- lowest bin {room.iloc[0]:.4f}, "
              f"highest bin {room.iloc[-1]:.4f}")
        print(f"[read] strong stratum ceiling change: "
              f"{by_bin['ceiling'].iloc[-1] - by_bin['ceiling'].iloc[0]:+.4f}")

        if room.iloc[0] > 0.15:
            print("\n[verdict] Room persists in the low-dose bin. Models can be "
                  "distinguished there. The C2S re-run is justified.")
        else:
            print("\n[verdict] Room collapses at low dose even for strong "
                  "perturbations. Degradation is signal-strength. Report that "
                  "honestly rather than as a model failure.")


if __name__ == "__main__":
    main()

