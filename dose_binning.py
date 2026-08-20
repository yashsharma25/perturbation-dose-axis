#!/usr/bin/env python
"""
Assign each cell a knockdown-dose bin from its sgRNA UMI count.

Shared by every model arm so that ground truth and predictions are grouped
identically. The only change any arm needs is to add the returned bin to its
aggregation key:

    (sample, guide_target, gene_target)  ->  (sample, guide_target, gene_target, bin)

Aggregation itself does not change. This is still per-group pseudo-bulk, which
is correct: single-cell logFC is mostly counting noise. What changes is which
cells go into which group.

Two properties carried over from the validated envelope analysis:

  RESIDUALIZED AGAINST DEPTH. Guide UMI correlates with sequencing depth
  (Spearman ~0.25 in HCT116). Binning on the raw count sorts cells partly by
  how deeply they were sequenced, which shuffles them across bin boundaries and
  attenuates the dose gradient. Regressing log dose on log depth and binning on
  the residual strengthened the measured knockdown gradient from -0.21 to -0.40.

  BINNED WITHIN PERTURBATION. Guide sequences differ in capture efficiency for
  reasons unrelated to silencing. Binning globally would fill the low bin with
  poorly-captured guides -- a guide-identity axis wearing a dose costume.
  Quantile bins within each perturbation hold guide identity constant.
"""

import numpy as np
import pandas as pd


DOSE_AGGREGATIONS = ("dose_min", "dose_sum", "dose_geomean")


def load_dose_per_cell(guide_calls_path, dose_column="dose_min"):
    """Per-cell dose from the pipe-delimited guide-call CSV.

    Orion gives two guides per cell, e.g. num_umis = "1794|1481". Collapsing
    them is a modelling choice:
      dose_min      silencing is rate-limited by the weaker guide
      dose_sum      guides act as additive dosing agents
      dose_geomean  robust to one guide's capture failing
    """
    if dose_column not in DOSE_AGGREGATIONS:
        raise ValueError(f"dose_column must be one of {DOSE_AGGREGATIONS}")

    calls = pd.read_csv(guide_calls_path)
    passing = calls["pass_guide_filter"].astype(str).str.lower().isin({"true", "1"})
    calls = calls[passing]

    per_guide_umis = calls["num_umis"].astype(str).str.split("|").map(
        lambda parts: np.array([float(p) for p in parts])
    )
    collapse = {
        "dose_min": np.min,
        "dose_sum": np.sum,
        "dose_geomean": lambda u: float(np.exp(np.log(np.maximum(u, 1e-9)).mean())),
    }[dose_column]

    return pd.Series(
        per_guide_umis.map(collapse).to_numpy(dtype=float),
        index=calls["cell_barcode"].to_numpy(),
        name=dose_column,
    )


def _residualize(dose_values, depth_values):
    """Return log dose with its depth-explained component removed."""
    log_dose = np.log1p(dose_values)
    log_depth = np.log1p(depth_values)
    if np.std(log_depth) <= 0:
        return log_dose
    slope, intercept = np.polyfit(log_depth, log_dose, 1)
    return log_dose - (slope * log_depth + intercept)


def compute_dose_bins(
    obs,
    guide_calls_path,
    *,
    dose_column="dose_min",
    n_bins=2,
    min_cells_per_perturbation=40,
    residualize=True,
    shuffle=False,
    seed=0,
    gene_column="gene_target",
    sample_column="sample",
    depth_column="total_counts",
    control_label="Non-Targeting",
    verbose=True,
):
    """Bin label per cell, aligned positionally to `obs`.

    Returns an int array of length len(obs). -1 means unassigned: the cell has
    no guide call, belongs to a control, or sits in a perturbation with too few
    cells to bin. Callers should exclude -1 cells from their tuples.

    `shuffle=True` is the PERMUTATION CONTROL. Dose values are permuted within
    each perturbation, so bins become random subsets of the same cells with the
    same marginal dose distribution and the same bin sizes. Any per-bin
    difference that survives is an artifact of slicing, not of knockdown
    strength. Every real dose number should be read against its shuffled
    counterpart rather than against zero.
    """
    dose = load_dose_per_cell(guide_calls_path, dose_column)
    aligned_dose = dose.reindex(obs.index).to_numpy(dtype=float)

    n_matched = int(np.isfinite(aligned_dose).sum())
    if verbose:
        print(f"[dose] {n_matched:,} of {len(obs):,} cells matched to a guide call "
              f"({n_matched / max(len(obs), 1):.1%})")
    if n_matched == 0:
        raise ValueError(
            "No barcodes matched. Check that obs index and cell_barcode share "
            "a format -- Orion uses the 10X barcode with -1 replaced by -<SAMPLE>."
        )

    frame = pd.DataFrame({
        "position": np.arange(len(obs)),
        "dose": aligned_dose,
        "depth": obs[depth_column].to_numpy(dtype=float),
        "gene": obs[gene_column].astype(str).to_numpy(),
    })

    bins = np.full(len(obs), -1, dtype=np.int64)
    n_binned_perturbations = 0
    n_skipped_too_few = 0
    rng = np.random.default_rng(seed)

    if shuffle and verbose:
        print("[dose] PERMUTATION CONTROL -- dose shuffled within each perturbation")

    for gene, group in frame.groupby("gene", sort=False):
        if gene == control_label:
            continue
        usable = np.isfinite(group["dose"].to_numpy())
        if usable.sum() < min_cells_per_perturbation:
            n_skipped_too_few += 1
            continue

        values = group["dose"].to_numpy()[usable]
        if residualize:
            values = _residualize(values, group["depth"].to_numpy()[usable])
        if shuffle:
            values = rng.permutation(values)

        try:
            labels = pd.qcut(values, n_bins, labels=False, duplicates="drop")
        except ValueError:
            n_skipped_too_few += 1
            continue
        if labels is None or np.nanmax(labels) < n_bins - 1:
            n_skipped_too_few += 1
            continue

        bins[group["position"].to_numpy()[usable]] = labels.astype(np.int64)
        n_binned_perturbations += 1

    if verbose:
        assigned = bins >= 0
        print(f"[dose] binned {n_binned_perturbations:,} perturbations into {n_bins} bins "
              f"({assigned.sum():,} cells); skipped {n_skipped_too_few:,} with "
              f"< {min_cells_per_perturbation} usable cells")
        for bin_id in range(n_bins):
            in_bin = bins == bin_id
            print(f"[dose]   bin {bin_id}: {in_bin.sum():,} cells, "
                  f"median depth {np.median(frame['depth'].to_numpy()[in_bin]):,.0f}")

        if sample_column in obs.columns:
            samples = obs[sample_column].astype(str).to_numpy()
            composition = pd.crosstab(samples[assigned], bins[assigned], normalize="columns")
            spread = float((composition.max(axis=1) - composition.min(axis=1)).max())
            print(f"[dose] max per-sample share difference across bins: {spread:.3f} "
                  f"(large values mean bins are confounded with batch)")

    return bins