#!/usr/bin/env python
"""
Regenerate the selection/evaluation cell split across EVERY perturbation.

Why this replaces screen_hct116_split.csv:

  1. The old split was built only for perturbations passing a >=200-cell filter
     under the old effect-size screen. The NSR-or-rho screen retains a largely
     different population -- 869 perturbations it keeps were never in the old
     pool -- and those cells carry no split assignment, so they cannot enter a
     new evaluation set.

  2. The old split omitted control cells entirely (it filtered to
     gene_target != "Non-Targeting"), which meant downstream code asking for
     selection-half controls got an empty set.

What the split is for. Selection cells choose things: which perturbations are
reliable, which genes are differentially expressed, what the PC-space deltas
are. Evaluation cells measure things: every reported score. Nothing is chosen
and measured on the same cells, so adopting a published inclusion criterion
cannot become selection-on-evaluation leakage.

Assignment is per perturbation and seeded, so it is reproducible from the seed
alone and stable if the script is re-run. Cells are shuffled within each
gene_target and cut in half, which keeps both halves balanced per perturbation
rather than only in aggregate -- necessary because downstream dose binning needs
enough cells per bin in the evaluation half of EACH perturbation.
"""

import argparse

import numpy as np
import pandas as pd
import anndata as ad


CONTROL_LABEL = "Non-Targeting"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--h5ad", required=True)
    parser.add_argument("--guide_calls", default=None,
                        help="Restrict to cells with a passing guide call. "
                             "Recommended: matches what every arm actually uses.")
    parser.add_argument("--min_cells", type=int, default=24,
                        help="Perturbations below this are still assigned, but "
                             "flagged as too small for a usable evaluation half.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="split_hct116.csv")
    return parser.parse_args()


def is_dual_nontargeting(guide_string):
    guides = str(guide_string).split("|")
    return len(guides) >= 2 and all(g.startswith("non-targeting") for g in guides)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    adata = ad.read_h5ad(args.h5ad, backed="r")
    obs = adata.obs
    print(f"[load] {adata.n_obs:,} cells")

    barcodes = obs.index.to_numpy()
    genes = obs["gene_target"].astype(str).to_numpy()
    guides = obs["guide_target"].astype(str).to_numpy()

    usable = np.ones(len(barcodes), dtype=bool)
    if "pass_guide_filter" in obs.columns:
        usable &= obs["pass_guide_filter"].to_numpy().astype(bool)
        print(f"[filter] pass_guide_filter keeps {int(usable.sum()):,}")

    if args.guide_calls:
        calls = pd.read_csv(args.guide_calls, usecols=["cell_barcode",
                                                       "pass_guide_filter"])
        passing = set(calls.loc[
            calls["pass_guide_filter"].astype(str).str.lower().isin({"true", "1"}),
            "cell_barcode"])
        usable &= pd.Index(barcodes).isin(passing)
        print(f"[filter] guide calls keep {int(usable.sum()):,}")

    # Controls are labelled by guide, not by gene_target, and must be split too:
    # downstream code needs selection-half controls for reference means and null
    # calibration, and evaluation-half controls for scoring baselines.
    control_mask = np.array([is_dual_nontargeting(g) for g in guides])
    group_labels = np.where(control_mask, CONTROL_LABEL, genes)
    print(f"[groups] {int(control_mask.sum()):,} control cells, "
          f"{len(set(group_labels[usable])) - 1:,} perturbations")

    frame = pd.DataFrame({
        "cell_barcode": barcodes,
        "gene_target": group_labels,
        "is_control": control_mask,
    })[usable].reset_index(drop=True)

    assignment = np.empty(len(frame), dtype=object)
    small_groups = []

    for group, rows in frame.groupby("gene_target", sort=True).groups.items():
        positions = np.asarray(rows)
        shuffled = rng.permutation(len(positions))
        midpoint = len(positions) // 2
        assignment[positions[shuffled[:midpoint]]] = "selection"
        assignment[positions[shuffled[midpoint:]]] = "evaluation"
        if len(positions) < args.min_cells and group != CONTROL_LABEL:
            small_groups.append(group)

    frame["split"] = assignment
    frame.to_csv(args.out, index=False)

    counts = frame["split"].value_counts()
    print(f"\n[split] selection {counts.get('selection', 0):,} / "
          f"evaluation {counts.get('evaluation', 0):,}")

    per_group = (frame[~frame["is_control"]]
                 .groupby(["gene_target", "split"]).size().unstack(fill_value=0))
    if "evaluation" in per_group.columns:
        evaluation_cells = per_group["evaluation"]
        print(f"[split] evaluation cells per perturbation: "
              f"median {int(evaluation_cells.median())}, "
              f"min {int(evaluation_cells.min())}, "
              f"max {int(evaluation_cells.max())}")
        for threshold in (30, 60, 90, 120):
            n_above = int((evaluation_cells >= threshold).sum())
            print(f"[split]   >= {threshold:>3} evaluation cells: {n_above:,} "
                  f"perturbations")

    controls = frame[frame["is_control"]]["split"].value_counts()
    print(f"[split] controls: selection {controls.get('selection', 0):,} / "
          f"evaluation {controls.get('evaluation', 0):,}")

    if small_groups:
        print(f"\n[warn] {len(small_groups):,} perturbations have < {args.min_cells} "
              f"cells; still assigned, but their evaluation half will be too "
              f"small to dose-bin")

    print(f"\n[save] {args.out}")
    print("[note] Freeze this file. Every arm and every screen must read the "
          "same split,\n       or scores stop being comparable across arms.")

    try:
        adata.file.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()