#!/usr/bin/env python
"""
Screen perturbations by true transcriptional effect size, to build the frozen
evaluation set every downstream arm will use.

Why this exists: a random sample of perturbations is mostly perturbations that
do nothing. Median split-half reliability on 120 random HCT116 perturbations was
0.086 -- the null. The top of the distribution (NCBP1, EIF3D, SNRPA: cap-binding,
translation initiation, spliceosome) reached 0.45-0.74. The instrument works; it
was pointed at genes with nothing to see.

Two properties that make the output publishable:

  NON-CIRCULAR. Every perturbation's cells are split in two. Screening happens
  on the SELECTION half only. The evaluation half is never touched here, so the
  later envelope and model scoring are measured on cells that played no part in
  choosing which perturbations to keep. Without this you select partly on noise
  and the reported ceiling is inflated by construction.

  UNBIASED EFFECT SIZE. ||logFC|| is inflated by noise, so ranking on it ranks
  noisy perturbations. Instead the selection half is itself split, and the effect
  is estimated as <logFC_a, logFC_b>. Independent noise cancels in expectation,
  leaving the true squared effect.

Writes two files:
  {out}_ranked.csv   one row per perturbation: effect size, reliability, counts
  {out}_split.csv    cell_barcode -> selection|evaluation, for downstream reuse
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


def mean_count_fraction(reader, rows, n_genes, count_detections=False):
    """Mean per-cell count fraction over a set of rows.

    Fractions rather than raw counts: guide UMI correlates with sequencing depth,
    and depth inflates every raw count including the ones being compared.
    """
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


def assign_halves(barcodes, rng):
    """Deterministic per-perturbation split into selection and evaluation."""
    shuffled = rng.permutation(len(barcodes))
    midpoint = len(barcodes) // 2
    assignment = np.empty(len(barcodes), dtype=object)
    assignment[shuffled[:midpoint]] = "selection"
    assignment[shuffled[midpoint:]] = "evaluation"
    return assignment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", required=True)
    parser.add_argument("--guide_calls", required=True)
    parser.add_argument("--min_total_cells", type=int, default=200,
                        help="Both halves must support downstream dose binning. "
                             "200 total gives ~100 per half, ~33 per cell in a "
                             "3-bin design.")
    parser.add_argument("--screen_cells", type=int, default=100,
                        help="Fixed cell cap from the selection half. Fixed so "
                             "estimator variance is equal across perturbations "
                             "and ranking does not track cell count.")
    parser.add_argument("--max_perturbations", type=int, default=3000)
    parser.add_argument("--n_control_cells", type=int, default=3000)
    parser.add_argument("--min_detection_rate", type=float, default=0.25)
    parser.add_argument("--epsilon", type=float, default=1e-4,
                        help="Must be near the scale of count fractions. 1e-6 "
                             "sits a hundredfold below and manufactures huge "
                             "log ratios on near-zero genes.")
    parser.add_argument("--flush_every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="screen_hct116")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    adata = anndata.read_h5ad(args.h5ad, backed="r")
    n_genes = adata.n_vars
    barcode_to_row = pd.Series(np.arange(adata.n_obs), index=adata.obs_names)
    print(f"[load] h5ad: {adata.n_obs:,} cells x {n_genes:,} genes")

    calls = pd.read_csv(args.guide_calls)
    calls = calls[calls["pass_guide_filter"].astype(str).str.lower().isin({"true", "1"})]
    calls = calls.set_index("cell_barcode")
    calls = calls.loc[calls.index.intersection(adata.obs_names)]
    print(f"[join] {len(calls):,} cells with passing guide calls")

    reader = BackedCSRReader(args.h5ad)
    try:
        control_barcodes = calls.index[calls["gene_target"].astype(str) == "Non-Targeting"]
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
        print(f"[genes] {gene_mask.sum():,} of {n_genes:,} genes detected in "
              f">= {args.min_detection_rate:.0%} of controls")

        perturbed = calls[calls["gene_target"].astype(str) != "Non-Targeting"]
        cells_per_gene = perturbed.groupby("gene_target").size()
        eligible = cells_per_gene[cells_per_gene >= args.min_total_cells].index.tolist()
        print(f"[eligible] {len(eligible):,} perturbations with "
              f">= {args.min_total_cells} cells")
        if len(eligible) > args.max_perturbations:
            eligible = sorted(rng.choice(eligible, size=args.max_perturbations, replace=False))
            print(f"[eligible] screening a random {len(eligible):,} of them")

        ranked_rows = []
        split_rows = []
        started = time.time()

        for position, gene in enumerate(eligible, start=1):
            barcodes = perturbed.index[perturbed["gene_target"].astype(str) == gene].to_numpy()
            assignment = assign_halves(barcodes, rng)
            split_rows.append(pd.DataFrame({
                "cell_barcode": barcodes,
                "gene_target": gene,
                "split": assignment,
            }))

            selection_barcodes = barcodes[assignment == "selection"]
            selection_rows = barcode_to_row.reindex(selection_barcodes).dropna().astype(int).to_numpy()
            if len(selection_rows) > args.screen_cells:
                selection_rows = rng.choice(selection_rows, size=args.screen_cells, replace=False)
            if len(selection_rows) < 20:
                continue

            shuffled = rng.permutation(selection_rows)
            quarter_a, quarter_b = shuffled[: len(shuffled) // 2], shuffled[len(shuffled) // 2:]

            mean_a, _, n_a = mean_count_fraction(reader, np.sort(quarter_a), n_genes)
            mean_b, _, n_b = mean_count_fraction(reader, np.sort(quarter_b), n_genes)
            if mean_a is None or mean_b is None:
                continue

            logfc_a = log_fold_change(mean_a[gene_mask], control_masked, args.epsilon)
            logfc_b = log_fold_change(mean_b[gene_mask], control_masked, args.epsilon)

            # Independent noise cancels in expectation; what survives is the
            # true squared effect. Negative values mean no detectable effect.
            effect_squared = float(np.dot(logfc_a, logfc_b))
            effect_size = float(np.sqrt(effect_squared)) if effect_squared > 0 else 0.0
            reliability = float(stats.pearsonr(logfc_a, logfc_b).statistic)

            ranked_rows.append({
                "gene_target": gene,
                "effect_size_unbiased": effect_size,
                "effect_squared_raw": effect_squared,
                "reliability_selection": reliability,
                "naive_norm": float(np.linalg.norm(0.5 * (logfc_a + logfc_b))),
                "n_cells_total": int(len(barcodes)),
                "n_cells_screened": int(n_a + n_b),
            })

            if position % args.flush_every == 0:
                pd.DataFrame(ranked_rows).to_csv(f"{args.out}_ranked.csv", index=False)
                rate = position / (time.time() - started)
                print(f"[screen] {position}/{len(eligible)}  {rate:.2f} pert/s  "
                      f"~{(len(eligible) - position) / max(rate, 1e-9) / 60:.0f} min left",
                      flush=True)
    finally:
        reader.close()

    ranked = pd.DataFrame(ranked_rows).sort_values("effect_size_unbiased", ascending=False)
    ranked.to_csv(f"{args.out}_ranked.csv", index=False)
    pd.concat(split_rows, ignore_index=True).to_csv(f"{args.out}_split.csv", index=False)

    print(f"\n[save] {args.out}_ranked.csv  ({len(ranked)} perturbations)")
    print(f"[save] {args.out}_split.csv    (selection/evaluation assignment)")

    print("\n[distribution] unbiased effect size")
    print(ranked["effect_size_unbiased"].quantile([.5, .75, .9, .95, .99]).to_string())
    print("\n[distribution] selection-half reliability")
    print(ranked["reliability_selection"].quantile([.5, .75, .9, .95, .99]).to_string())

    detectable = (ranked["effect_squared_raw"] > 0) & (ranked["reliability_selection"] > 0.2)
    print(f"\n[detectable] {detectable.sum():,} of {len(ranked):,} perturbations "
          f"({detectable.mean():.1%}) have a positive effect estimate and "
          f"selection-half reliability above 0.2")

    print("\n[top 15]")
    print(ranked.head(15)[["gene_target", "effect_size_unbiased",
                           "reliability_selection", "n_cells_total"]].to_string(index=False))

    if detectable.sum() < 100:
        print("\n[verdict] Fewer than 100 detectable perturbations. Widen the screen "
              "or drop to a perturbation-level dose contrast.")
    else:
        print(f"\n[verdict] Enough signal to build a stratified eval set. Draw from "
              f"the {detectable.sum():,} detectable perturbations across the effect-size "
              f"range so the paper can show where the dose axis works and where it does not.")


if __name__ == "__main__":
    main()