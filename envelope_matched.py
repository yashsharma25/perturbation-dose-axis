#!/usr/bin/env python
"""
Perturbation-specific reproducible signal within dose bins.

Four corrections over the previous version, each fixing a way the earlier
numbers were contaminated:

  1. MATCHED CONTROLS. Binning on any per-cell variable sorts cells by state.
     Comparing every bin against one global control mean let each bin's
     cell-state offset enter the logFC as if it were perturbation signal.
     Controls are now indexed on a cell-state grid and each bin gets a baseline
     built from controls occupying the same grid cells in the same proportions.

  2. CROSS-PAIRED FLOOR. Previously the floor prediction was built from control
     half A and then scored against both perturbed halves -- so it shared
     denominator cells with half A and correlated with it through denominator
     noise alone. The A-based floor is now scored against half B and vice versa.

  3. SPECIFIC SIGNAL instead of ceiling-minus-floor. Subtracting one correlation
     from another has no clean interpretation. Writing each half as
         logFC = alpha * generic_direction + specific + noise
     the generic direction is projected out of both halves and the residuals
     correlated. That is the reproducible signal unique to this perturbation --
     precisely what a model must capture to beat the pedestal.

  4. PERMUTATION NULL. The previous acceptance test (bin on pct_counts_mt) was
     circular: the matching grid contains pct_counts_mt, so matching removed the
     binning variable by construction. The real null is --shuffle_dose, which
     permutes dose within each perturbation. Same cells, same marginal
     distribution, assignment randomized. Any structure surviving that is
     artifact, and the shuffled ceiling is the reference value expected under no
     dose effect at matched cell count.

Run --shuffle_dose FIRST. Its numbers are the baseline every real dose number
must be read against.
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


def project_out(vector, direction):
    """Remove the component of `vector` along `direction`, both centered."""
    centered_vector = vector - vector.mean()
    centered_direction = direction - direction.mean()
    denominator = float(np.dot(centered_direction, centered_direction))
    if denominator <= 0:
        return centered_vector
    coefficient = float(np.dot(centered_vector, centered_direction)) / denominator
    return centered_vector - coefficient * centered_direction


def residualize_dose_against_depth(dose_values, depth_values):
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


class ControlAtlas:
    """Control means indexed by cell state.

    Means of count fractions are linear, so a baseline matched to any set of
    perturbed cells is a weighted average of grid-cell means -- computed once,
    reused everywhere, no re-reading.

    Two independent halves so that perturbed half A and half B never share
    control cells in their denominators.
    """

    def __init__(self, depth_edges, mito_edges):
        self.depth_edges = depth_edges
        self.mito_edges = mito_edges
        self.means = {"a": {}, "b": {}}
        self.global_mean = {"a": None, "b": None}

    def assign(self, depths, mito):
        depth_bin = np.clip(
            np.searchsorted(self.depth_edges, np.log10(np.maximum(depths, 1)),
                            side="right") - 1,
            0, len(self.depth_edges) - 2)
        mito_bin = np.clip(
            np.searchsorted(self.mito_edges, mito, side="right") - 1,
            0, len(self.mito_edges) - 2)
        return list(zip(depth_bin.tolist(), mito_bin.tolist()))

    def matched_baseline(self, coordinates, half):
        means, fallback = self.means[half], self.global_mean[half]
        baseline = np.zeros_like(fallback)
        unmatched = 0.0
        for coordinate, count in pd.Series(coordinates).value_counts().items():
            weight = count / len(coordinates)
            if coordinate in means:
                baseline += weight * means[coordinate]
            else:
                unmatched += weight
        if unmatched > 0:
            baseline += unmatched * fallback
        return baseline, unmatched


def build_atlas(reader, control_barcodes, obs, barcode_to_row, n_genes,
                n_depth_bins, n_mito_bins, cells_per_grid_cell, rng):
    depths = obs.loc[control_barcodes, "total_counts"].to_numpy(float)
    mito = obs.loc[control_barcodes, "pct_counts_mt"].to_numpy(float)

    depth_edges = np.quantile(np.log10(np.maximum(depths, 1)),
                              np.linspace(0, 1, n_depth_bins + 1))
    mito_edges = np.quantile(mito, np.linspace(0, 1, n_mito_bins + 1))
    depth_edges[0], depth_edges[-1] = -np.inf, np.inf
    mito_edges[0], mito_edges[-1] = -np.inf, np.inf

    atlas = ControlAtlas(depth_edges, mito_edges)
    frame = pd.DataFrame({"barcode": control_barcodes,
                          "coordinate": atlas.assign(depths, mito)})

    detections_total = np.zeros(n_genes, dtype=np.int64)
    cells_read = 0
    pooled = {"a": np.zeros(n_genes), "b": np.zeros(n_genes)}
    pooled_n = {"a": 0, "b": 0}

    grid_cells = frame["coordinate"].unique()
    print(f"[atlas] {len(grid_cells)} occupied grid cells "
          f"({n_depth_bins} depth x {n_mito_bins} mito)", flush=True)

    started = time.time()
    for position, coordinate in enumerate(grid_cells, start=1):
        pool = frame.loc[frame["coordinate"] == coordinate, "barcode"].to_numpy()
        if len(pool) < 40:
            continue
        if len(pool) > cells_per_grid_cell:
            pool = rng.choice(pool, size=cells_per_grid_cell, replace=False)
        rows = barcode_to_row.reindex(pool).dropna().astype(int).to_numpy()
        shuffled = rng.permutation(rows)
        halves = {"a": shuffled[: len(shuffled) // 2], "b": shuffled[len(shuffled) // 2:]}

        computed = {}
        for half, subset in halves.items():
            mean_vector, detections, n_used = mean_count_fraction(
                reader, np.sort(subset), n_genes, count_detections=True)
            if mean_vector is None:
                break
            computed[half] = (mean_vector, n_used)
            detections_total += detections
            cells_read += n_used
        if len(computed) != 2:
            continue

        for half, (mean_vector, n_used) in computed.items():
            atlas.means[half][coordinate] = mean_vector
            pooled[half] += mean_vector * n_used
            pooled_n[half] += n_used

        if position % 10 == 0:
            print(f"[atlas] {position}/{len(grid_cells)}  "
                  f"{time.time() - started:.0f}s", flush=True)

    for half in ("a", "b"):
        atlas.global_mean[half] = pooled[half] / max(pooled_n[half], 1)
    print(f"[atlas] built from {cells_read:,} control cells in "
          f"{time.time() - started:.0f}s")
    return atlas, detections_total, cells_read


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", required=True)
    parser.add_argument("--guide_calls", required=True)
    parser.add_argument("--eval_set", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--dose_column", default="dose_min",
                        choices=["dose_sum", "dose_min", "dose_geomean"])
    parser.add_argument("--shuffle_dose", action="store_true",
                        help="PERMUTATION NULL. Shuffles dose within each "
                             "perturbation. Run this first; its numbers are the "
                             "reference every real dose number is read against.")
    parser.add_argument("--no_residualize", action="store_true")
    parser.add_argument("--no_match", action="store_true",
                        help="Disable cell-state matching, for comparison only.")
    parser.add_argument("--n_bins", type=int, default=2)
    parser.add_argument("--n_depth_bins", type=int, default=6)
    parser.add_argument("--n_mito_bins", type=int, default=5)
    parser.add_argument("--cells_per_grid_cell", type=int, default=200)
    parser.add_argument("--min_cells_per_bin", type=int, default=24)
    parser.add_argument("--min_detection_rate", type=float, default=0.25)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="envelope.csv")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    adata = anndata.read_h5ad(args.h5ad, backed="r")
    n_genes = adata.n_vars
    barcode_to_row = pd.Series(np.arange(adata.n_obs), index=adata.obs_names)
    obs = adata.obs[["total_counts", "pct_counts_mt"]]
    print(f"[load] h5ad: {adata.n_obs:,} cells x {n_genes:,} genes")

    eval_set = pd.read_csv(args.eval_set)
    stratum_of = dict(zip(eval_set["gene_target"], eval_set["stratum"]))
    split = pd.read_csv(args.split)
    evaluation_barcodes = set(split.loc[split["split"] == "evaluation", "cell_barcode"])
    print(f"[load] {len(eval_set)} perturbations, "
          f"{len(evaluation_barcodes):,} evaluation-half cells")

    if args.shuffle_dose:
        print("[null] PERMUTATION NULL -- dose shuffled within perturbation")

    dose = load_dose(args.guide_calls, args.dose_column)
    dose = dose.loc[dose.index.intersection(adata.obs_names)]

    reader = BackedCSRReader(args.h5ad)
    try:
        control_barcodes = dose.index[dose["gene_target"] == "Non-Targeting"].to_numpy()
        atlas, detections, control_cells_read = build_atlas(
            reader, control_barcodes, obs, barcode_to_row, n_genes,
            args.n_depth_bins, args.n_mito_bins, args.cells_per_grid_cell, rng)

        gene_mask = (detections / control_cells_read) >= args.min_detection_rate
        control_levels = atlas.global_mean["a"][gene_mask]
        below_epsilon = float((control_levels < args.epsilon).mean())
        print(f"[genes] {gene_mask.sum():,} of {n_genes:,} retained; "
              f"{below_epsilon:.1%} sit below epsilon={args.epsilon:g} "
              f"and are shrunk toward zero logFC")
        print(f"[genes] control mean fraction quantiles: "
              f"{np.quantile(control_levels, [.05, .5, .95])}")

        # Generic perturbation direction: what every knockdown shares. Built
        # from evaluation-half cells so it lives in the same universe as the
        # per-bin estimates. Kept in both atlas halves so it can be cross-paired.
        in_eval = dose["gene_target"].isin(stratum_of)
        pool = dose.index[in_eval & dose.index.isin(evaluation_barcodes)].to_numpy()
        generic_barcodes = rng.choice(pool, size=min(3000, len(pool)), replace=False)
        generic_rows = barcode_to_row.reindex(generic_barcodes).dropna().astype(int).to_numpy()
        print(f"[generic] reading {len(generic_rows):,} perturbed cells...", flush=True)
        generic_mean, _, _ = mean_count_fraction(reader, np.sort(generic_rows), n_genes)
        generic_coordinates = atlas.assign(
            obs.loc[generic_barcodes, "total_counts"].to_numpy(float),
            obs.loc[generic_barcodes, "pct_counts_mt"].to_numpy(float))
        generic_direction = {}
        for half in ("a", "b"):
            baseline, _ = atlas.matched_baseline(generic_coordinates, half)
            generic_direction[half] = log_fold_change(
                generic_mean[gene_mask], baseline[gene_mask], args.epsilon)

        records = []
        started = time.time()
        for position, gene in enumerate(sorted(stratum_of), start=1):
            cells = dose[(dose["gene_target"] == gene)
                         & dose.index.isin(evaluation_barcodes)]
            if len(cells) < args.n_bins * args.min_cells_per_bin:
                continue

            values = cells["dose"].to_numpy(float)
            if not args.no_residualize:
                depth = obs.loc[cells.index, "total_counts"].to_numpy(float)
                values = residualize_dose_against_depth(values, depth)
            if args.shuffle_dose:
                values = rng.permutation(values)

            try:
                bins = pd.qcut(values, args.n_bins, labels=False, duplicates="drop")
            except ValueError:
                continue
            if np.nanmax(bins) < args.n_bins - 1:
                continue

            barcodes = cells.index.to_numpy()
            rows = barcode_to_row.reindex(barcodes).dropna().astype(int).to_numpy()
            coordinates = atlas.assign(
                obs.loc[barcodes, "total_counts"].to_numpy(float),
                obs.loc[barcodes, "pct_counts_mt"].to_numpy(float))

            for bin_id in range(args.n_bins):
                in_bin = np.flatnonzero(bins == bin_id)
                if len(in_bin) < args.min_cells_per_bin:
                    continue
                shuffled = rng.permutation(in_bin)
                indices = {"a": shuffled[: len(shuffled) // 2],
                           "b": shuffled[len(shuffled) // 2:]}

                logfc, unmatched_weights = {}, []
                usable = True
                for half, index in indices.items():
                    mean_vector, _, _ = mean_count_fraction(
                        reader, np.sort(rows[index]), n_genes)
                    if mean_vector is None:
                        usable = False
                        break
                    if args.no_match:
                        baseline, unmatched = atlas.global_mean[half], 0.0
                    else:
                        baseline, unmatched = atlas.matched_baseline(
                            [coordinates[i] for i in index], half)
                    logfc[half] = log_fold_change(
                        mean_vector[gene_mask], baseline[gene_mask], args.epsilon)
                    unmatched_weights.append(unmatched)
                if not usable:
                    continue

                total_reproducibility = float(
                    stats.pearsonr(logfc["a"], logfc["b"]).statistic)

                # Cross-paired: the generic direction built on control half A is
                # scored against perturbed half B, so no shared control cells.
                generic_alignment = 0.5 * (
                    float(stats.pearsonr(generic_direction["a"], logfc["b"]).statistic)
                    + float(stats.pearsonr(generic_direction["b"], logfc["a"]).statistic))

                residual_a = project_out(logfc["a"], generic_direction["b"])
                residual_b = project_out(logfc["b"], generic_direction["a"])
                specific_signal = float(stats.pearsonr(residual_a, residual_b).statistic)

                records.append({
                    "gene_target": gene,
                    "stratum": stratum_of[gene],
                    "bin": bin_id,
                    "n_cells": len(in_bin),
                    "specific_signal": specific_signal,
                    "total_reproducibility": total_reproducibility,
                    "generic_alignment": generic_alignment,
                    "effect_size": float(np.sqrt(max(np.dot(logfc["a"], logfc["b"]), 0.0))),
                    "median_depth": float(obs.loc[barcodes[in_bin], "total_counts"].median()),
                    "median_mito": float(obs.loc[barcodes[in_bin], "pct_counts_mt"].median()),
                    "unmatched_weight": float(np.mean(unmatched_weights)),
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

    mode = "PERMUTATION NULL" if args.shuffle_dose else "real dose"
    matching = "UNMATCHED" if args.no_match else "cell-state matched"
    print(f"\n[envelope] {mode}, {matching}; bin 0 = lowest\n")

    for stratum in ["strong", "medium", "weak", "near_null"]:
        subset = envelope[envelope["stratum"] == stratum]
        if len(subset) == 0:
            continue
        table = subset.groupby("bin").agg(
            specific=("specific_signal", "median"),
            total=("total_reproducibility", "median"),
            generic=("generic_alignment", "median"),
            effect=("effect_size", "median"),
            depth=("median_depth", "median"),
            mito=("median_mito", "median"),
            n=("gene_target", "nunique"),
        )
        print(f"--- {stratum} ---")
        print(table.to_string(float_format=lambda v: f"{v:.4f}"))
        print()

    print(f"[diag] median unmatched weight: {envelope['unmatched_weight'].median():.3f} "
          f"(fraction of cells with no control in their grid cell; "
          f"above ~0.10 means coarsen the grid)")

    print("\n[read] `specific` is the headline: reproducible signal unique to this "
          "perturbation,\n       after the generic response shared by all knockdowns is "
          "projected out.")
    if args.shuffle_dose:
        print("\n[read] These are null values. Real dose bins must be compared against\n"
              "       them, not against zero. near_null `specific` should sit near 0 and\n"
              "       be flat across bins; if it is not, the design still leaks.")
    else:
        print("\n[read] Compare `specific` per bin against the shuffled run at the same\n"
              "       cell count. A dose effect means the gap between bins EXCEEDS what\n"
              "       the permutation produces.")


if __name__ == "__main__":
    main()