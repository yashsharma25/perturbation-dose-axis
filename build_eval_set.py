#!/usr/bin/env python
"""
Freeze the evaluation set from the screen's ranked output.

Stratified across the effect-size range rather than taking the top N. The weak
and near-null strata are not padding: they are what let the paper show where
dose stratification works AND where it does not, which is the difference between
a method and a claim about a method. They also serve as a negative control -- a
dose gradient appearing in perturbations with no measurable effect would mean
the gradient is an artifact.

Perturbations are drawn using effect_size_unbiased, which was measured on the
SELECTION half only. The evaluation half stays untouched, so downstream scoring
is non-circular.

Writes {out}_eval_set.csv: gene_target, stratum, effect size, cell counts.
"""

import argparse

import numpy as np
import pandas as pd


STRATA = [
    ("strong", 2.5, np.inf, 60),
    ("medium", 1.5, 2.5, 60),
    ("weak", 0.8, 1.5, 40),
    ("near_null", -np.inf, 0.8, 40),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranked", required=True, help="{out}_ranked.csv from the screen")
    parser.add_argument("--split", required=True, help="{out}_split.csv from the screen")
    parser.add_argument("--min_evaluation_cells", type=int, default=90,
                        help="Cells in the evaluation half. 90 over 3 dose bins "
                             "is 30 per bin, which is the floor for a stable "
                             "pseudo-bulk mean.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="eval_set_hct116")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    ranked = pd.read_csv(args.ranked)
    split = pd.read_csv(args.split)

    evaluation_counts = (
        split[split["split"] == "evaluation"]
        .groupby("gene_target")
        .size()
        .rename("n_evaluation_cells")
    )
    ranked = ranked.merge(evaluation_counts, on="gene_target", how="left")

    usable = ranked[ranked["n_evaluation_cells"] >= args.min_evaluation_cells].copy()
    print(f"[filter] {len(usable):,} of {len(ranked):,} perturbations have "
          f">= {args.min_evaluation_cells} evaluation cells")

    selected = []
    for name, low, high, target_count in STRATA:
        pool = usable[
            (usable["effect_size_unbiased"] >= low)
            & (usable["effect_size_unbiased"] < high)
        ]
        if len(pool) == 0:
            print(f"[stratum] {name:<10} EMPTY -- no perturbations in "
                  f"[{low}, {high})")
            continue
        take = min(target_count, len(pool))
        drawn = pool.sample(take, random_state=int(rng.integers(1 << 31))).copy()
        drawn["stratum"] = name
        selected.append(drawn)
        print(f"[stratum] {name:<10} {take:>3} drawn from a pool of {len(pool):,}"
              + ("   (POOL EXHAUSTED)" if take < target_count else ""))

    eval_set = pd.concat(selected, ignore_index=True)
    eval_set = eval_set.sort_values("effect_size_unbiased", ascending=False)

    columns = ["gene_target", "stratum", "effect_size_unbiased",
               "reliability_selection", "n_cells_total", "n_evaluation_cells"]
    eval_set[columns].to_csv(f"{args.out}_eval_set.csv", index=False)
    print(f"\n[save] {args.out}_eval_set.csv  ({len(eval_set)} perturbations)")

    print("\n[summary] by stratum")
    print(
        eval_set.groupby("stratum")
        .agg(
            n=("gene_target", "size"),
            median_effect=("effect_size_unbiased", "median"),
            median_reliability=("reliability_selection", "median"),
            median_eval_cells=("n_evaluation_cells", "median"),
        )
        .to_string(float_format=lambda v: f"{v:.3f}")
    )

    print("\n[sample] three from each stratum")
    for name, *_ in STRATA:
        rows = eval_set[eval_set["stratum"] == name].head(3)
        if len(rows):
            genes = ", ".join(rows["gene_target"].tolist())
            print(f"  {name:<10} {genes}")


if __name__ == "__main__":
    main()