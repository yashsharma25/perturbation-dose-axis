#!/usr/bin/env python
"""
Validate the sgRNA-UMI dose proxy before spending GPU hours on it.

Question: within a perturbation, do cells with more guide UMIs show more
suppression of the gene that guide targets?

Two corrections over the naive version:

  1. Raw h5py row access instead of adata[rows].to_memory(). The latter issues
     two latency-bound reads per row THROUGH a fresh AnnData construction; at
     100k rows on a network filesystem that is hours of pure overhead for a
     few GB of actual data.

  2. Target expression is measured as a COUNT FRACTION, not a raw count.
     Guide UMI correlates with sequencing depth (rho ~ 0.25 in HCT116), and
     depth inflates every raw count including the gene you are trying to show
     is suppressed. Comparing raw counts across dose bins biases toward null.

Ground truth only. No models involved.
"""

import argparse
import time

import numpy as np
import pandas as pd
import h5py
import anndata
from scipy import stats


# ---------------------------------------------------------------- guide calls

def load_dose_per_cell(guide_calls_path):
    """Explode pipe-delimited per-guide UMI counts into one dose value per cell.

    Three candidate aggregations of the two guides, since the right choice is a
    modelling decision rather than a technical one:

      sum      guides act as additive dosing agents
      min      silencing is rate-limited by the weaker guide
      geomean  robust to one guide's capture failing
    """
    calls = pd.read_csv(guide_calls_path)
    keep = calls["pass_guide_filter"].astype(str).str.lower().isin({"true", "1"})
    calls = calls[keep]

    umis = calls["num_umis"].astype(str).str.split("|").map(
        lambda parts: np.array([float(p) for p in parts])
    )

    dose = pd.DataFrame(
        {
            "gene_target": calls["gene_target"].astype(str).to_numpy(),
            "dose_sum": umis.map(np.sum).to_numpy(),
            "dose_min": umis.map(np.min).to_numpy(),
            "dose_geomean": umis.map(lambda u: float(np.exp(np.log(u).mean()))).to_numpy(),
        },
        index=calls["cell_barcode"].to_numpy(),
    )
    dose.index.name = "cell_barcode"
    return dose


def residualize_against_depth(dose_values, depth_values):
    """Strip the depth-driven component out of the dose signal.

    Regress log dose on log depth and keep the residual. What remains is the
    part of guide abundance NOT explained by how deeply the cell was sequenced,
    which is the part that could plausibly reflect silencing strength.
    """
    log_dose = np.log1p(dose_values)
    log_depth = np.log1p(depth_values)
    slope, intercept = np.polyfit(log_depth, log_dose, 1)
    return log_dose - (slope * log_depth + intercept)


# ------------------------------------------------------- raw sparse row reads

class BackedCSRReader:
    """Direct h5py access to a backed CSR matrix, bypassing AnnData overhead."""

    def __init__(self, h5ad_path):
        self.handle = h5py.File(h5ad_path, "r")
        group = self.handle["X"]
        if isinstance(group, h5py.Dataset):
            raise ValueError("X is dense; this reader expects CSR.")
        self.indptr = group["indptr"][:]
        self.indices = group["indices"]
        self.data = group["data"]

    def gene_fraction(self, row, gene_column):
        """Return (count of gene_column, total counts) for one cell."""
        lo, hi = int(self.indptr[row]), int(self.indptr[row + 1])
        if hi <= lo:
            return 0.0, 0.0
        cols = self.indices[lo:hi]
        vals = self.data[lo:hi]
        total = float(vals.sum())
        hit = np.flatnonzero(cols == gene_column)
        count = float(vals[hit[0]]) if hit.size else 0.0
        return count, total

    def gene_counts_multi(self, row, gene_columns):
        """Return {gene_column: count} plus total counts, for several genes."""
        lo, hi = int(self.indptr[row]), int(self.indptr[row + 1])
        if hi <= lo:
            return {}, 0.0
        cols = self.indices[lo:hi]
        vals = self.data[lo:hi]
        total = float(vals.sum())
        wanted = np.isin(cols, gene_columns)
        return dict(zip(cols[wanted].tolist(), vals[wanted].tolist())), total

    def close(self):
        self.handle.close()


# --------------------------------------------------------------- the analysis

def control_baseline(reader, control_rows, gene_columns):
    """Mean count-fraction of each target gene across non-targeting cells."""
    totals = {column: 0.0 for column in gene_columns}
    gene_array = np.asarray(sorted(gene_columns))
    n_used = 0

    for row in control_rows:
        counts, depth = reader.gene_counts_multi(int(row), gene_array)
        if depth <= 0:
            continue
        for column in gene_columns:
            totals[column] += counts.get(column, 0.0) / depth
        n_used += 1

    if n_used == 0:
        raise ValueError("No usable control cells.")
    return {column: total / n_used for column, total in totals.items()}, n_used


def knockdown_by_dose_bin(reader, adata, dose, dose_column, n_bins,
                          min_cells, max_cells, max_perturbations,
                          residualize, rng):
    gene_to_column = {name: i for i, name in enumerate(adata.var_names)}
    barcode_to_row = pd.Series(np.arange(adata.n_obs), index=adata.obs_names)

    per_perturbation = dose.groupby("gene_target").size()
    eligible = [
        gene for gene, count in per_perturbation.items()
        if count >= min_cells and gene != "Non-Targeting" and gene in gene_to_column
    ]
    if len(eligible) > max_perturbations:
        eligible = sorted(rng.choice(eligible, size=max_perturbations, replace=False))

    target_columns = {gene_to_column[gene] for gene in eligible}
    print(f"[scan] {len(eligible)} perturbations pass the >= {min_cells} cell threshold")

    control_barcodes = dose.index[dose["gene_target"] == "Non-Targeting"]
    control_rows = barcode_to_row.reindex(control_barcodes).dropna().astype(int).to_numpy()
    control_rows = np.sort(rng.choice(control_rows, size=min(2000, len(control_rows)),
                                      replace=False))
    print(f"[baseline] reading {len(control_rows)} non-targeting cells...", flush=True)
    started = time.time()
    baseline, n_control_used = control_baseline(reader, control_rows, target_columns)
    print(f"[baseline] done in {time.time() - started:.1f}s ({n_control_used} usable)")

    records = []
    started = time.time()
    for position, gene in enumerate(eligible, start=1):
        cells = dose[dose["gene_target"] == gene]
        rows = barcode_to_row.reindex(cells.index).dropna().astype(int)
        if len(rows) < min_cells:
            continue
        if len(rows) > max_cells:
            rows = rows.sample(max_cells, random_state=int(rng.integers(1 << 31)))
        cells = cells.loc[rows.index]

        dose_values = cells[dose_column].to_numpy(dtype=float)
        if residualize:
            depth = adata.obs["total_counts"].reindex(cells.index).to_numpy(dtype=float)
            dose_values = residualize_against_depth(dose_values, depth)

        try:
            bins = pd.qcut(dose_values, n_bins, labels=False, duplicates="drop")
        except ValueError:
            continue
        if np.nanmax(bins) < n_bins - 1:
            continue

        target_column = gene_to_column[gene]
        control_level = baseline[target_column]
        if control_level <= 0:
            continue

        row_values = rows.to_numpy()
        read_order = np.argsort(row_values)
        fractions = np.full(len(rows), np.nan)
        depths = np.full(len(rows), np.nan)
        for slot in read_order:
            count, total = reader.gene_fraction(int(row_values[slot]), target_column)
            if total > 0:
                fractions[slot] = count / total
                depths[slot] = total

        for bin_id in range(n_bins):
            in_bin = (bins == bin_id) & np.isfinite(fractions)
            if in_bin.sum() == 0:
                continue
            records.append({
                "gene_target": gene,
                "dose_bin": bin_id,
                "n_cells": int(in_bin.sum()),
                "mean_dose": float(np.mean(dose_values[in_bin])),
                "median_depth": float(np.median(depths[in_bin])),
                "relative_target_expression": float(
                    np.mean(fractions[in_bin]) / control_level
                ),
            })

        if position % 10 == 0:
            rate = position / (time.time() - started)
            remaining = (len(eligible) - position) / max(rate, 1e-9)
            print(f"[scan] {position}/{len(eligible)}  {rate:.2f} pert/s  "
                  f"~{remaining / 60:.1f} min left", flush=True)

    return pd.DataFrame(records)


def summarise(knockdown, n_bins):
    per_bin = knockdown.groupby("dose_bin").agg(
        mean_relative_expression=("relative_target_expression", "mean"),
        median_relative_expression=("relative_target_expression", "median"),
        median_depth=("median_depth", "median"),
        n_perturbations=("gene_target", "nunique"),
    )
    print("\n[validation] Target-gene count fraction relative to non-targeting control")
    print(f"             bin 0 = lowest dose, bin {n_bins - 1} = highest")
    print(per_bin.to_string())
    print("\n[validation] If median_depth trends across bins, depth is leaking into "
          "the binning; rerun with --residualize_depth.")

    trends = (
        knockdown.groupby("gene_target")
        .apply(
            lambda g: stats.spearmanr(g["dose_bin"],
                                      g["relative_target_expression"]).statistic
            if len(g) >= 3 else np.nan,
            include_groups=False,
        )
        .dropna()
    )
    decreasing = float((trends < 0).mean())
    print(f"\n[validation] Perturbations with decreasing target expression: "
          f"{decreasing:.1%} of {len(trends)}")
    print(f"[validation] Median within-perturbation Spearman: {trends.median():.4f}")

    if decreasing > 0.70:
        print("\n[verdict] Proxy validated. Guide UMI tracks knockdown strength.")
    elif decreasing > 0.55:
        print("\n[verdict] Weak but present. Report the effect size; do not "
              "over-claim curve precision.")
    else:
        print("\n[verdict] No monotone relationship. The axis is reading guide "
              "capture, not silencing.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", required=True)
    parser.add_argument("--guide_calls", required=True)
    parser.add_argument("--dose_column", default="dose_min",
                        choices=["dose_sum", "dose_min", "dose_geomean"])
    parser.add_argument("--n_bins", type=int, default=4)
    parser.add_argument("--min_cells_per_perturbation", type=int, default=40)
    parser.add_argument("--max_cells_per_perturbation", type=int, default=160,
                        help="Subsample cap. 160 cells over 4 bins is 40 per bin, "
                             "ample for a monotonicity test and far faster than "
                             "reading every cell.")
    parser.add_argument("--max_perturbations", type=int, default=60)
    parser.add_argument("--residualize_depth", action="store_true",
                        help="Bin on log-dose residualized against log-total-counts.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="dose_validation.csv")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    dose = load_dose_per_cell(args.guide_calls)
    print(f"[load] {len(dose):,} cells with guide calls")

    adata = anndata.read_h5ad(args.h5ad, backed="r")
    print(f"[load] h5ad: {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    shared = dose.index.intersection(adata.obs_names)
    print(f"[join] {len(shared):,} barcodes matched ({len(shared) / adata.n_obs:.1%})")
    dose = dose.loc[shared]

    depth = adata.obs.loc[shared, "total_counts"].to_numpy(dtype=float)
    coupling = stats.spearmanr(dose[args.dose_column].to_numpy(), depth).statistic
    print(f"[depth] Spearman({args.dose_column}, total_counts) = {coupling:.4f}")

    reader = BackedCSRReader(args.h5ad)
    try:
        knockdown = knockdown_by_dose_bin(
            reader, adata, dose, args.dose_column, args.n_bins,
            args.min_cells_per_perturbation, args.max_cells_per_perturbation,
            args.max_perturbations, args.residualize_depth, rng,
        )
    finally:
        reader.close()

    knockdown.to_csv(args.out, index=False)
    print(f"\n[save] {args.out}  ({len(knockdown)} perturbation-bin rows)")
    summarise(knockdown, args.n_bins)


if __name__ == "__main__":
    main()