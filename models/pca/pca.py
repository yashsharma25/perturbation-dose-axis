#!/usr/bin/env python
"""
PCA baseline for perturbation prediction on X-Atlas/Orion.

Method: fit PCA on control cells, learn one mean shift per gene_target in PC
space, add that shift to a perturbation's control baseline, reconstruct.

The shift is learned per gene_target and NOT per dose bin. PCA has no notion of
knockdown strength, so it emits the same prediction for both bins while the
ground truth differs. Any per-bin asymmetry in its scores is the measurement.

WHY SCORING USES PER-PERTURBATION DE GENES
Scoring every tuple on a shared 3000-gene panel gave a per-perturbation bin gap
with SD 0.56 -- so large that a 60-perturbation stratum could only resolve
effects above ~0.17, while the ground-truth dose effect measured ~0.073. The
test could not have detected the effect either way.

A typical knockdown moves a few hundred genes. The rest contribute noise to both
the predicted and the observed logFC, and low-expression genes contribute wild
log-ratios. So each perturbation is now scored on its own top-|logFC| genes,
chosen among reasonably expressed genes only.

WHY THE SELECTION/EVALUATION SPLIT
Choosing genes and scoring on the same cells selects on noise and inflates every
score. Cells are split by --split (from screen_perturbations.py): the selection
half picks each perturbation's DE genes AND trains the PC-space deltas; the
evaluation half is scored. Nothing is chosen and measured on the same cells.

This also fixes the circularity in the earlier version, where deltas were
learned from the very tuples they were scored against.
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from sklearn.decomposition import PCA

from dose_binning import compute_dose_bins


OBS_SAMPLE = "sample"
OBS_GUIDE_TARGET = "guide_target"
OBS_GENE_TARGET = "gene_target"
TUPLE_SEP = "\x1f"
BIN_SEP = "@@"

METRIC_NAMES = ["MAE", "MSE", "RMSE", "Pearson", "Spearman",
                "CosineLogFC", "CosineLogFCSpecific", "StructuralSignAgreement"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--h5ad", required=True)
    parser.add_argument("--tag", default="A")
    parser.add_argument("--save_dir", default="save/xatlas_pca")

    parser.add_argument("--guide_calls", default=None,
                        help="Per-cell guide-call CSV. Omit for pooled evaluation.")
    parser.add_argument("--dose_bins", type=int, default=2)
    parser.add_argument("--dose_column", default="dose_min",
                        choices=["dose_min", "dose_sum", "dose_geomean"])
    parser.add_argument("--min_cells_per_perturbation", type=int, default=40)
    parser.add_argument("--shuffle_dose", action="store_true",
                        help="PERMUTATION CONTROL. Bins become random subsets.")

    parser.add_argument("--eval_set", required=True,
                        help="CSV with a gene_target column.")
    parser.add_argument("--split", required=True,
                        help="screen_*_split.csv: cell_barcode, split. The selection "
                             "half picks DE genes and trains deltas; the evaluation "
                             "half is scored.")
    parser.add_argument("--de_genes", type=int, default=200,
                        help="Genes scored per perturbation, chosen by |logFC| on "
                             "the selection half.")
    parser.add_argument("--min_expression_quantile", type=float, default=0.60,
                        help="DE candidates must sit above this quantile of control "
                             "expression. Low-expression genes produce wild log "
                             "ratios that dominate the variance.")
    parser.add_argument("--min_cells_per_tuple", type=int, default=20)

    parser.add_argument("--pca_components", type=int, default=128)
    parser.add_argument("--pca_fit_cells", type=int, default=50000)
    parser.add_argument("--min_controls_per_sample", type=int, default=5)
    parser.add_argument("--magnitude_epsilon", type=float, default=1e-5)
    parser.add_argument("--log_every_seconds", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def format_seconds(seconds):
    if not np.isfinite(seconds):
        return "unknown"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def sum_rows(matrix, row_indices):
    row_indices = np.asarray(row_indices, dtype=np.int64)
    if row_indices.size == 0:
        return None, 0
    block = matrix[row_indices]
    totals = (np.asarray(block.sum(axis=0)).ravel() if sparse.issparse(block)
              else np.asarray(block, dtype=np.float32).sum(axis=0))
    return totals.astype(np.float32, copy=False), int(row_indices.size)


def dense_rows(matrix, row_indices):
    block = matrix[np.asarray(row_indices, dtype=np.int64)]
    if sparse.issparse(block):
        return block.toarray().astype(np.float32, copy=False)
    return np.asarray(block, dtype=np.float32)


def safe_log1p(values):
    return np.log1p(np.clip(values, 0.0, None)).astype(np.float32, copy=False)


def is_dual_nontargeting(guide_string):
    guides = str(guide_string).split("|")
    return len(guides) >= 2 and all(g.startswith("non-targeting") for g in guides)


def control_mask_from_guides(guide_strings):
    return np.array([is_dual_nontargeting(g) for g in guide_strings], dtype=bool)


def control_means_per_sample(matrix, sample_labels, control_indices):
    means, counts = {}, {}
    control_samples = sample_labels[control_indices]
    order = np.argsort(control_samples, kind="mergesort")
    sorted_indices = control_indices[order]
    sorted_samples = control_samples[order]
    if sorted_samples.size == 0:
        return means, counts
    boundaries = np.where(sorted_samples[1:] != sorted_samples[:-1])[0] + 1
    for start, end in zip(np.concatenate([[0], boundaries]),
                          np.concatenate([boundaries, [sorted_samples.size]])):
        sample = str(sorted_samples[start])
        totals, n_cells = sum_rows(matrix, sorted_indices[start:end])
        if n_cells > 0:
            means[sample] = (totals / float(n_cells)).astype(np.float32, copy=False)
            counts[sample] = n_cells
    return means, counts


def build_sample_baseline_matrix(sample_order, global_mean, sample_means, sample_counts,
                                 min_controls_per_sample):
    matrix = np.empty((len(sample_order), global_mean.size), dtype=np.float32)
    n_fallback = 0
    for row, sample in enumerate(sample_order):
        if (sample_counts.get(sample, 0) >= min_controls_per_sample
                and sample in sample_means):
            matrix[row] = sample_means[sample]
        else:
            matrix[row] = global_mean
            n_fallback += 1
    log(f"[baseline] {len(sample_order) - n_fallback}/{len(sample_order)} samples use "
        f"their own controls")
    return matrix


def group_cells(matrix, keys, cell_mask, sample_columns, n_samples,
                min_cells, label):
    """Mean expression per unique key, plus each group's sample composition.

    Composition is what lets the baseline stay batch-corrected without putting
    sample into the grouping key -- with 109 samples, doing that shattered every
    perturbation into 1.9-cell groups.
    """
    indices = np.where(cell_mask)[0].astype(np.int64)
    if indices.size == 0:
        raise RuntimeError(f"[{label}] no cells survived filtering")

    group_keys = keys[indices]
    order = np.argsort(group_keys, kind="mergesort")
    sorted_indices = indices[order]
    sorted_keys = group_keys[order]

    boundaries = np.where(sorted_keys[1:] != sorted_keys[:-1])[0] + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [sorted_keys.size]])
    log(f"[{label}] {len(starts):,} groups from {indices.size:,} cells "
        f"({indices.size / max(len(starts), 1):.1f} cells each)")

    groups = {"keys": [], "means": [], "n_cells": [], "weights": [], "n_samples": []}
    n_dropped = 0
    started = time.time()

    for position, (start, end) in enumerate(zip(starts, ends), start=1):
        rows = sorted_indices[start:end]
        if rows.size < min_cells:
            n_dropped += 1
            continue
        totals, n_cells = sum_rows(matrix, rows)
        if n_cells == 0:
            n_dropped += 1
            continue
        counts = np.bincount(sample_columns[rows], minlength=n_samples)
        groups["keys"].append(str(sorted_keys[start]))
        groups["means"].append((totals / float(n_cells)).astype(np.float32, copy=False))
        groups["n_cells"].append(n_cells)
        groups["weights"].append(counts.astype(np.float32) / float(n_cells))
        groups["n_samples"].append(int((counts > 0).sum()))

        if position % 200 == 0 or position == len(starts):
            rate = position / max(time.time() - started, 1e-9)
            log(f"[{label}] {position:,}/{len(starts):,}  "
                f"ETA {format_seconds((len(starts) - position) / max(rate, 1e-9))}")

    groups["weights"] = (np.stack(groups["weights"], axis=0) if groups["weights"]
                         else np.zeros((0, n_samples), dtype=np.float32))
    log(f"[{label}] kept {len(groups['keys']):,}; dropped {n_dropped:,} with "
        f"< {min_cells} cells")
    return groups


def choose_de_genes(logfc, candidate_mask, n_genes_to_keep):
    """Top genes by |logFC| among reasonably expressed candidates."""
    scores = np.where(candidate_mask, np.abs(logfc), -np.inf)
    n_available = int(np.isfinite(scores).sum())
    keep = min(n_genes_to_keep, n_available)
    if keep <= 0:
        return None
    return np.argpartition(-scores, keep - 1)[:keep].astype(np.int64)


def project_out(vector, direction, epsilon=1e-12):
    squared_norm = float(np.dot(direction, direction))
    if squared_norm <= epsilon:
        return vector
    return vector - (float(np.dot(vector, direction)) / squared_norm) * direction


def cosine(left, right, epsilon=1e-8):
    left_norm, right_norm = np.linalg.norm(left), np.linalg.norm(right)
    if left_norm < epsilon or right_norm < epsilon:
        return float("nan")
    return float(np.dot(left, right) / (left_norm * right_norm))


def pearson(left, right, epsilon=1e-8):
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = np.sqrt(np.dot(left_centered, left_centered)
                          * np.dot(right_centered, right_centered))
    return float(np.dot(left_centered, right_centered) / denominator) if denominator > epsilon \
        else float("nan")


def spearman(left, right):
    return pearson(np.argsort(np.argsort(left)).astype(np.float64),
                   np.argsort(np.argsort(right)).astype(np.float64))


def summarise(per_tuple, weight_key=None):
    summary = {}
    for name in METRIC_NAMES:
        values, weights = [], []
        for row in per_tuple.values():
            value = row.get(name)
            if isinstance(value, (int, float)) and np.isfinite(value):
                values.append(value)
                weights.append(float(row.get(weight_key, 1.0)) if weight_key else 1.0)
        summary[name] = (float(np.average(values, weights=weights)) if values
                         else float("nan"))
    summary["n_tuples"] = len(per_tuple)
    return summary


def evaluate(args):
    rng = np.random.default_rng(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / f"eval_{args.tag}.json"
    if output_path.exists():
        raise FileExistsError(f"{output_path} exists -- pick a different --tag")

    log(f"[load] {args.h5ad}")
    adata = ad.read_h5ad(args.h5ad, backed="r")
    obs = adata.obs
    matrix = adata.X
    n_genes = int(adata.n_vars)
    log(f"[load] {adata.n_obs:,} cells x {n_genes:,} genes")

    gene_names = (adata.var["gene_name"].astype(str).tolist()
                  if "gene_name" in adata.var.columns
                  else [str(name) for name in adata.var_names])
    gene_to_column = {name: index for index, name in enumerate(gene_names)}

    eval_genes = set(pd.read_csv(args.eval_set)["gene_target"].astype(str))
    log(f"[eval_set] {len(eval_genes):,} perturbations")

    split = pd.read_csv(args.split)
    selection_barcodes = set(split.loc[split["split"] == "selection", "cell_barcode"])
    evaluation_barcodes = set(split.loc[split["split"] == "evaluation", "cell_barcode"])
    log(f"[split] {len(selection_barcodes):,} selection / "
        f"{len(evaluation_barcodes):,} evaluation cells")

    if args.guide_calls:
        obs = obs.copy()
        dose_bins = compute_dose_bins(
            obs, args.guide_calls, dose_column=args.dose_column, n_bins=args.dose_bins,
            min_cells_per_perturbation=args.min_cells_per_perturbation,
            shuffle=args.shuffle_dose, seed=args.seed)
        tagged = obs[OBS_GUIDE_TARGET].astype(str).to_numpy().copy()
        assigned = dose_bins >= 0
        tagged[assigned] = [f"{guide}{BIN_SEP}bin{bin_id}"
                            for guide, bin_id in zip(tagged[assigned], dose_bins[assigned])]
        obs[OBS_GUIDE_TARGET] = tagged
        log(f"[dose] tagged {int(assigned.sum()):,} cells")

    guides = obs[OBS_GUIDE_TARGET].astype(str).to_numpy()
    genes = obs[OBS_GENE_TARGET].astype(str).to_numpy()
    samples = obs[OBS_SAMPLE].astype(str).to_numpy()
    barcodes = obs.index.to_numpy()
    sample_order = sorted(set(samples))
    sample_position = {sample: index for index, sample in enumerate(sample_order)}
    sample_columns = np.array([sample_position[s] for s in samples], dtype=np.int64)

    control_indices = np.where(control_mask_from_guides(guides))[0].astype(np.int64)
    log(f"[baseline] global control mean over {control_indices.size:,} cells")
    totals, n_controls = sum_rows(matrix, control_indices)
    global_mean = (totals / float(n_controls)).astype(np.float32, copy=False)

    log("[baseline] per-sample control means")
    sample_means, sample_counts = control_means_per_sample(matrix, samples, control_indices)
    sample_baselines = build_sample_baseline_matrix(
        sample_order, global_mean, sample_means, sample_counts, args.min_controls_per_sample)

    in_eval_set = pd.Index(genes).isin(eval_genes)
    is_perturbed = (~control_mask_from_guides(guides)) & in_eval_set
    if "pass_guide_filter" in obs.columns:
        is_perturbed &= obs["pass_guide_filter"].to_numpy().astype(bool)

    barcode_index = pd.Index(barcodes)
    in_selection = barcode_index.isin(selection_barcodes)
    in_evaluation = barcode_index.isin(evaluation_barcodes)

    # Selection half: one group per gene_target, pooling guides and dose bins.
    selection = group_cells(matrix, genes, is_perturbed & in_selection, sample_columns,
                            len(sample_order), args.min_cells_per_tuple, "selection")

    # Evaluation half: one group per (guide[+bin], gene).
    evaluation_keys = np.array([TUPLE_SEP.join(pair) for pair in zip(guides, genes)],
                               dtype=object)
    evaluation = group_cells(matrix, evaluation_keys, is_perturbed & in_evaluation,
                             sample_columns, len(sample_order), args.min_cells_per_tuple,
                             "evaluation")

    fit_indices = control_indices
    if fit_indices.size > args.pca_fit_cells:
        fit_indices = np.sort(rng.choice(fit_indices, size=args.pca_fit_cells, replace=False))
    log(f"[pca] fitting {args.pca_components} components on {fit_indices.size:,} cells")
    started = time.time()
    pca = PCA(n_components=args.pca_components, svd_solver="randomized", random_state=args.seed)
    pca.fit(dense_rows(matrix, fit_indices))
    log(f"[pca] fit in {format_seconds(time.time() - started)}; explained variance "
        f"{pca.explained_variance_ratio_.sum():.3f}")
    components = pca.components_.astype(np.float32, copy=False)
    pca_mean = pca.mean_.astype(np.float32, copy=False)
    projected_samples = pca.transform(sample_baselines).astype(np.float32)

    # Deltas and DE gene sets, both from the selection half only.
    selection_baselines = selection["weights"] @ sample_baselines
    selection_projected = pca.transform(np.stack(selection["means"], axis=0)).astype(np.float32)
    selection_projected_baselines = selection["weights"] @ projected_samples

    expression_threshold = float(np.quantile(global_mean, args.min_expression_quantile))
    candidate_mask = global_mean >= expression_threshold
    log(f"[de] {int(candidate_mask.sum()):,} genes above the "
        f"{args.min_expression_quantile:.0%} expression quantile are DE candidates")

    deltas, de_gene_sets = {}, {}
    generic_logfc = np.zeros(n_genes, dtype=np.float64)
    total_cells = float(sum(selection["n_cells"]))
    for position, gene in enumerate(selection["keys"]):
        deltas[gene] = (selection_projected[position]
                        - selection_projected_baselines[position]).astype(np.float32)
        logfc = (safe_log1p(selection["means"][position])
                 - safe_log1p(selection_baselines[position]))
        chosen = choose_de_genes(logfc, candidate_mask, args.de_genes)
        if chosen is not None:
            de_gene_sets[gene] = chosen
        generic_logfc += logfc * (selection["n_cells"][position] / total_cells)
    log(f"[de] chose gene sets for {len(de_gene_sets):,} perturbations; "
        f"learned {len(deltas):,} deltas")
    generic_logfc = generic_logfc.astype(np.float32)

    evaluation_baselines = evaluation["weights"] @ sample_baselines
    evaluation_projected_baselines = evaluation["weights"] @ projected_samples

    per_tuple = {}
    n_missing = 0
    started = time.time()
    last_log = started

    for position, key in enumerate(evaluation["keys"]):
        guide_with_bin, gene = key.split(TUPLE_SEP)
        guide, _, dose_bin = guide_with_bin.partition(BIN_SEP)
        columns = de_gene_sets.get(gene)
        shift = deltas.get(gene)
        if columns is None or shift is None:
            n_missing += 1
            continue

        baseline = evaluation_baselines[position][columns]
        observed = evaluation["means"][position][columns]
        predicted = ((evaluation_projected_baselines[position] + shift)
                     @ components[:, columns]) + pca_mean[columns]

        logfc_observed = safe_log1p(observed) - safe_log1p(baseline)
        logfc_predicted = safe_log1p(predicted) - safe_log1p(baseline)
        direction = generic_logfc[columns]

        difference = predicted - observed
        above = ((np.abs(logfc_observed) > args.magnitude_epsilon)
                 | (np.abs(logfc_predicted) > args.magnitude_epsilon))
        n_above = int(above.sum())

        per_tuple[key] = {
            "guide_target": guide,
            "gene_target": gene,
            "dose_bin": dose_bin,
            "n_cells_in_tuple": int(evaluation["n_cells"][position]),
            "n_samples_in_tuple": int(evaluation["n_samples"][position]),
            "n_genes_scored": int(columns.size),
            "MAE": float(np.mean(np.abs(difference))),
            "MSE": float(np.mean(difference ** 2)),
            "RMSE": float(np.sqrt(np.mean(difference ** 2))),
            "Pearson": pearson(predicted, observed),
            "Spearman": spearman(predicted, observed),
            "CosineLogFC": cosine(logfc_predicted, logfc_observed),
            "CosineLogFCSpecific": cosine(project_out(logfc_predicted, direction),
                                          project_out(logfc_observed, direction)),
            "StructuralSignAgreement": (
                float(np.sum((np.sign(logfc_observed) == np.sign(logfc_predicted)) & above)
                      / n_above) if n_above else float("nan")),
        }

        now = time.time()
        if now - last_log > args.log_every_seconds:
            log(f"[score] {position + 1:,}/{len(evaluation['keys']):,}")
            last_log = now

    if n_missing:
        log(f"[warn] {n_missing:,} evaluation tuples had no selection-half counterpart")

    output = {
        "tag": args.tag,
        "file": args.h5ad,
        "n_tuples": len(per_tuple),
        "dose_binned": bool(args.guide_calls),
        "shuffled": bool(args.shuffle_dose),
        "summary": {"unweighted": summarise(per_tuple),
                    "weighted_by_cells": summarise(per_tuple, "n_cells_in_tuple")},
        "per_tuple": per_tuple,
        "config": vars(args),
        "definitions": {
            "grouping": "(guide_target[, dose_bin], gene_target); sample is NOT in the "
                        "key -- batch is handled by composition-weighted baselines",
            "scored_genes": f"top {args.de_genes} by |logFC| on the SELECTION half, among "
                            f"genes above the {args.min_expression_quantile:.0%} control "
                            f"expression quantile; frozen per gene_target so both dose "
                            f"bins are scored on identical genes",
            "delta": "PC-space shift per gene_target from the SELECTION half only",
            "evaluation": "EVALUATION half cells only -- disjoint from selection",
        },
    }
    with open(output_path, "w") as handle:
        json.dump(output, handle, indent=2)
    log(f"[done] wrote {output_path}")

    rows = pd.DataFrame(per_tuple.values())
    if args.guide_calls and len(rows):
        mode = "PERMUTATION CONTROL" if args.shuffle_dose else "real dose"
        log(f"\n[summary] by dose bin ({mode})")
        print(rows.groupby("dose_bin").agg(
            cosine_logfc=("CosineLogFC", "median"),
            cosine_specific=("CosineLogFCSpecific", "median"),
            genes=("n_genes_scored", "median"),
            cells=("n_cells_in_tuple", "median"),
            n=("gene_target", "nunique"),
        ).to_string(float_format=lambda v: f"{v:.4f}"))

        wide = rows.pivot_table(index="gene_target", columns="dose_bin",
                                values="CosineLogFCSpecific")
        if wide.shape[1] == 2:
            gap = (wide.iloc[:, 1] - wide.iloc[:, 0]).dropna()
            detectable = 2.8 * gap.std() / np.sqrt(len(gap))
            log(f"\n[power] per-perturbation gap: median {gap.median():+.4f}, "
                f"SD {gap.std():.4f}, n={len(gap)}")
            log(f"[power] smallest effect detectable at 80% power: {detectable:.4f}")
            log(f"[power] ground-truth dose effect from the envelope: ~0.073")

    try:
        adata.file.close()
    except Exception:
        pass


if __name__ == "__main__":
    evaluate(parse_args())