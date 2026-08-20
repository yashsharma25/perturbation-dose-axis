#!/usr/bin/env python
"""
Write a subset h5ad containing only the cells STATE needs to predict.

Two reasons this exists rather than sharding drive_state.py over 3.4M cells:
speed, and control of the gene panel.

THE GENE PANEL IS THE RISK. xatlas_align.py selects a 2,000-gene panel, and if
it picks highly variable genes from whatever file it is handed, a subset would
produce a DIFFERENT panel -- silently making the new predictions incomparable
to everything else. So the subset is written already restricted to the existing
panel (read from a prior infer_celleval.h5ad). A top-2000-HVG selection applied
to ~1,957 genes takes all of them, which pins the panel without touching Arc's
alignment code.

Controls are included: STATE's pipeline and any downstream baseline need
non-targeting cells in the same file.
"""

import argparse
import time

import numpy as np
import pandas as pd
import h5py
import anndata as ad
from scipy import sparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", required=True)
    parser.add_argument("--eval_set", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--gene_panel_h5ad", required=True,
                        help="A prior infer_celleval.h5ad. Its var_names pin "
                             "the panel.")
    parser.add_argument("--max_cells_per_perturbation", type=int, default=80,
                        help="Evaluation-half cap. 80 gives 40 per dose bin.")
    parser.add_argument("--n_control_cells", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    panel = ad.read_h5ad(args.gene_panel_h5ad, backed="r")
    panel_genes = [str(g) for g in panel.var_names]
    print(f"[panel] {len(panel_genes):,} genes pinned from {args.gene_panel_h5ad}")

    adata = ad.read_h5ad(args.h5ad, backed="r")
    all_genes = np.array([str(g) for g in adata.var_names])
    gene_position = {g: i for i, g in enumerate(all_genes)}
    panel_columns = np.array([gene_position[g] for g in panel_genes
                              if g in gene_position], dtype=np.int64)
    print(f"[panel] {len(panel_columns):,} of them present in the readout space")

    eval_genes = set(pd.read_csv(args.eval_set)["gene_target"].astype(str))
    split = pd.read_csv(args.split)
    evaluation = set(split.loc[split["split"] == "evaluation", "cell_barcode"])
    control_barcodes = set(split.loc[split["is_control"], "cell_barcode"])
    print(f"[load] {len(eval_genes):,} perturbations, "
          f"{adata.n_obs:,} cells in source")

    barcodes = pd.Index(adata.obs_names.to_numpy())
    genes = adata.obs["gene_target"].astype(str).to_numpy()
    in_evaluation = barcodes.isin(evaluation)
    is_control = barcodes.isin(control_barcodes)

    chosen = []
    wanted = pd.Index(genes).isin(eval_genes) & in_evaluation & ~is_control
    frame = pd.DataFrame({"position": np.arange(adata.n_obs)[wanted],
                          "gene": genes[wanted]})
    for gene, group in frame.groupby("gene", sort=False):
        positions = group["position"].to_numpy()
        if len(positions) > args.max_cells_per_perturbation:
            positions = rng.choice(positions, size=args.max_cells_per_perturbation,
                                   replace=False)
        chosen.append(positions)
    perturbed_positions = np.concatenate(chosen) if chosen else np.array([], dtype=int)

    control_positions = np.where(is_control)[0]
    if len(control_positions) > args.n_control_cells:
        control_positions = rng.choice(control_positions, size=args.n_control_cells,
                                       replace=False)

    keep = np.sort(np.concatenate([perturbed_positions, control_positions]))
    print(f"[select] {len(perturbed_positions):,} perturbed + "
          f"{len(control_positions):,} control = {len(keep):,} cells")

    handle = h5py.File(args.h5ad, "r")
    group = handle["X"]
    indptr, indices, data = group["indptr"][:], group["indices"], group["data"]

    panel_lookup = np.full(len(all_genes), -1, dtype=np.int64)
    panel_lookup[panel_columns] = np.arange(len(panel_columns))

    rows, columns, values = [], [], []
    started = time.time()
    for output_row, source_row in enumerate(keep):
        lo, hi = int(indptr[source_row]), int(indptr[source_row + 1])
        if hi <= lo:
            continue
        source_columns = indices[lo:hi]
        mapped = panel_lookup[source_columns]
        wanted_positions = mapped >= 0
        if not wanted_positions.any():
            continue
        kept_columns = mapped[wanted_positions]
        kept_values = data[lo:hi][wanted_positions]
        rows.append(np.full(kept_columns.size, output_row, dtype=np.int32))
        columns.append(kept_columns.astype(np.int32))
        values.append(kept_values.astype(np.float32))
        if (output_row + 1) % 5000 == 0:
            rate = (output_row + 1) / max(time.time() - started, 1e-9)
            print(f"[read] {output_row + 1:,}/{len(keep):,}  "
                  f"~{(len(keep) - output_row - 1) / max(rate, 1e-9) / 60:.1f} min left",
                  flush=True)
    handle.close()

    matrix = sparse.csr_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=(len(keep), len(panel_columns)), dtype=np.float32)
    print(f"[read] done in {(time.time() - started) / 60:.1f} min; "
          f"{matrix.nnz:,} nonzeros")

    subset = ad.AnnData(
        X=matrix,
        obs=adata.obs.iloc[keep].copy(),
        var=adata.var.iloc[panel_columns].copy(),
    )
    subset.write_h5ad(args.out, compression="gzip")
    print(f"\n[save] {args.out}")
    print(f"[save] {subset.n_obs:,} cells x {subset.n_vars:,} genes")
    print(f"[save] {subset.obs['gene_target'].nunique():,} unique gene_target values")
    print("\n[note] Gene panel is pinned to the prior STATE run. If "
          "xatlas_align.py\n       still reselects, pass this panel explicitly "
          "or the arms stop being\n       comparable.")


if __name__ == "__main__":
    main()
