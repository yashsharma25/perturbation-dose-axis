#!/usr/bin/env python
"""
Freeze the evaluation set from the NSR-or-rho screen.

Four strata, chosen so the paper can show where the dose axis works AND where
it does not:

  strong_nsr   high detectability. Large, well-measured effects.
  mid_nsr      detectable but modest.
  rho_only     passes reproducibility but NOT detectability -- weak effects
               that are nonetheless consistently directed. Arguably where dose
               modulation matters most: a weak effect may vanish entirely at
               low knockdown while a strong one saturates.
  null         fails both. The negative control. A dose gradient appearing
               here would mean the gradient is an artifact.

Selection used the SELECTION half only, so nothing here is circular.
"""

import argparse

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", required=True, help="nsr_rho_*.csv")
    parser.add_argument("--split", required=True, help="split_hct116_v2.csv")
    parser.add_argument("--gene_panel_h5ad", default=None,
                        help="A STATE infer_celleval.h5ad. If given, only "
                             "perturbations whose response can be scored in that "
                             "gene panel are eligible.")
    parser.add_argument("--n_strong", type=int, default=250)
    parser.add_argument("--n_mid", type=int, default=250)
    parser.add_argument("--n_rho_only", type=int, default=150)
    parser.add_argument("--n_null", type=int, default=150)
    parser.add_argument("--min_evaluation_cells", type=int, default=60,
                        help="Two dose bins of 30. The floor for a stable "
                             "pseudo-bulk mean per bin.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="eval_set_v2.csv")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    screen = pd.read_csv(args.screen)
    split = pd.read_csv(args.split)

    evaluation_counts = (split[(split["split"] == "evaluation")
                               & (~split["is_control"])]
                         .groupby("gene_target").size().rename("n_evaluation_cells"))
    # the screen already carries an n_evaluation_cells column computed against
    # the OLD split; drop it so the merge against the current split is what counts
    screen = screen.drop(columns=["n_evaluation_cells"], errors="ignore")
    screen = screen.merge(evaluation_counts, on="gene_target", how="left")
    screen["n_evaluation_cells"] = screen["n_evaluation_cells"].fillna(0)

    eligible = screen[screen["n_evaluation_cells"] >= args.min_evaluation_cells].copy()
    print(f"[filter] {len(eligible):,} of {len(screen):,} perturbations have "
          f">= {args.min_evaluation_cells} evaluation cells")

    detectable = eligible[eligible["retained"]]
    strong_cutoff = detectable["nsr"].quantile(0.66) if len(detectable) else np.inf

    pools = {
        "strong_nsr": eligible[eligible["passes_nsr"] & (eligible["nsr"] >= strong_cutoff)],
        "mid_nsr": eligible[eligible["passes_nsr"] & (eligible["nsr"] < strong_cutoff)],
        "rho_only": eligible[eligible["passes_rho"] & ~eligible["passes_nsr"]],
        "null": eligible[~eligible["retained"]],
    }
    targets = {"strong_nsr": args.n_strong, "mid_nsr": args.n_mid,
               "rho_only": args.n_rho_only, "null": args.n_null}

    selected = []
    for name, pool in pools.items():
        want = targets[name]
        take = min(want, len(pool))
        if take == 0:
            print(f"[stratum] {name:<11} EMPTY")
            continue
        drawn = pool.sample(take, random_state=int(rng.integers(1 << 31))).copy()
        drawn["stratum"] = name
        selected.append(drawn)
        flag = "  (POOL EXHAUSTED)" if take < want else ""
        print(f"[stratum] {name:<11} {take:>4} of {len(pool):,} available{flag}")

    eval_set = pd.concat(selected, ignore_index=True)
    columns = ["gene_target", "stratum", "nsr", "rho",
               "n_evaluation_cells", "n_selection_cells"]
    columns = [c for c in columns if c in eval_set.columns]
    eval_set[columns].to_csv(args.out, index=False)

    print(f"\n[save] {args.out}  ({len(eval_set)} perturbations)")
    print("\n[summary]")
    print(eval_set.groupby("stratum").agg(
        n=("gene_target", "size"),
        median_nsr=("nsr", "median"),
        median_rho=("rho", "median"),
        median_eval_cells=("n_evaluation_cells", "median"),
    ).to_string(float_format=lambda v: f"{v:.3f}"))

    total_cells = int(eval_set["n_evaluation_cells"].sum())
    print(f"\n[cells] {total_cells:,} evaluation cells across the set")
    print(f"[power] at the PCA pilot gap SD of 0.596, n={len(eval_set)} resolves "
          f"{2.8 * 0.596 / np.sqrt(len(eval_set)):.3f} at 80% power "
          f"(target ~0.073)")


if __name__ == "__main__":
    main()