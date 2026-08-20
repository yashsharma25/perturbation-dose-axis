#!/usr/bin/env python
"""
Kalygina-style reliability screen (NSR OR rho), reconciled against the frozen
evaluation set and the dose axis.

WHY THIS SCRIPT EXISTS
Kalygina et al. (ICML 2026 workshop) retain 8,035 of 35,916 X-Atlas/Orion
condition units (22.4%) under a two-axis reliability rule. Our own effect-size
screen retained ~4%. Before adopting their inclusion criterion we need to know
whether the difference is a threshold choice or a different phenomenon -- and,
much more importantly, how many perturbations survive that ALSO sit in the
frozen evaluation set AND have enough cells in both dose bins.

That last count is the effective sample size of the paper. Global retention is
irrelevant if model inference only covers the frozen set.

THE TWO AXES (their Appendix H)

  Detectability.  NSR = ||y|| / se, where se^2 = V_p/n_p + V_ctrl/n_ctrl and V
  is the total per-cell variance (trace of the covariance). A relabelled control
  sits at NSR ~ 1; a strong effect is well above it.

  Reproducibility.  rho = across-gene correlation between responses estimated on
  two independent halves of the condition's cells. This measures whether the
  DIRECTION pattern reproduces, independent of magnitude, and rescues weak but
  consistently directed effects that detectability alone discards.

  Keep if EITHER clears the 95th percentile of a control-split null. Only the
  joint low-detectability, low-reproducibility corner is dropped.

Both thresholds are calibrated per context against pseudo-perturbations drawn
from control cells -- size-matched relabelled controls carrying pure technical
noise and zero effect, so their 95th percentile is a 5% false-positive cutoff.

SELECTION-HALF ONLY. The screen runs on selection cells so that adopting a
published inclusion criterion does not become a selection-on-evaluation leak.


HOW THIS VERSION COMPUTES THE SAME NUMBERS FASTER

The statistics are unchanged. Four things about HOW they are obtained changed,
because the previous loop-over-perturbations structure spent most of its wall
clock on bookkeeping and random disk seeks rather than on arithmetic.

  1. ONE SEQUENTIAL PASS, NOT 17,879 RANDOM ONES. Every cell the screen needs
     is collected up front, sorted into file order, and read once in a single
     forward sweep. Nearby rows are pulled in contiguous slabs, so the number of
     HDF5 read calls drops by roughly three orders of magnitude and every
     compressed chunk is touched once instead of repeatedly.

  2. PRECOMPUTED ROW INDEX. The old loop evaluated (genes == gene) over all
     4.5M cells twice per perturbation -- once for selection rows, once for the
     evaluation dose-bin counts. Measured at ~0.13 s per perturbation, that is
     ~40 minutes of pure string comparison. Both are now single groupbys.

  3. SPARSE ACCUMULATION. The old code materialised a dense
     (n_cells x 38,606) float32 block per perturbation and took log1p of it.
     Since log1p(0) = 0 the zeros were pure waste. We accumulate over nonzeros
     into per-gene sum vectors instead, ~10x faster and equal to float32
     rounding (verified: max abs mean difference 4.5e-08).

     The variance trace comes out of an algebraic identity rather than a dense
     per-gene variance:

         sum_g var_g = ( sum_i ||x_i||^2  -  n * ||mean||^2 ) / (n - 1)

     which needs one scalar accumulator per perturbation instead of a full
     sum-of-squares vector. Same reason the standard error only ever needed
     variance traces: the geometry is isotropic.

  4. PER-PERTURBATION RNG. Each perturbation draws from
     default_rng([seed, crc32(gene)]) rather than from one shared stream, so the
     cell subsample and the split-half assignment depend only on the gene name.
     Results are then identical whether the screen runs in one process or four,
     and whether or not it was restarted. crc32 rather than hash() because
     Python salts hash() per process.

ONE INTENTIONAL NUMERICAL CHANGE. The standard error now divides the control
variance by the number of control cells that actually reached the mean, not by
the number of rows requested. The old code passed len(control_rows), which
overcounts by however many control cells have zero depth. On a fixture with one
empty control in 300 this moved NSR by 1.7e-04 relative; if your h5ad has no
zero-depth cells the two are identical. Everything else agrees with the old
implementation to ~1e-08, i.e. to float32 rounding.

REPRODUCIBILITY NOTE. Because the random stream is now per gene rather than one
shared sequence, the cell subsample and the split-half assignment differ from
any previous run. Perturbations sitting within noise of a cutoff will flip in
or out. That is the same sensitivity the old script had to its seed, not a new
one, but retention counts will not reproduce a previous run exactly.

MEMORY. The pass holds two float32 accumulator vectors per perturbation, so
2 * n_perturbations * n_genes * 4 bytes. For 17,879 x 38,606 that is 5.5 GB.
The script prints the requirement before allocating and splits into passes if
--memory_budget_gb is smaller.
"""

import argparse
import json
import time
import zlib

import numpy as np
import pandas as pd
import h5py
import anndata as ad

from dose_binning import compute_dose_bins


_process_started = time.time()


def current_memory_gb():
    """Resident set size, so a silent kill can be attributed after the fact."""
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024 ** 2
    except OSError:
        pass
    return float("nan")


def log(message):
    print(f"[{time.time() - _process_started:7.1f}s {current_memory_gb():5.2f}GB] {message}",
          flush=True)


class BackedCSRReader:
    """Read-only view of the CSR blocks of an h5ad, with indptr held in memory.

    rdcc_nbytes is raised well above the h5py default of 1 MB. With compressed
    chunks a 1 MB cache means a chunk decompressed for one row is evicted before
    the next row in that chunk is asked for, so the same chunk is decompressed
    many times over.
    """

    def __init__(self, h5ad_path, chunk_cache_bytes=512 * 1024 ** 2):
        self.handle = h5py.File(h5ad_path, "r",
                                rdcc_nbytes=chunk_cache_bytes, rdcc_nslots=100_003)
        group = self.handle["X"]
        self.indptr = group["indptr"][:]
        self.indices = group["indices"]
        self.data = group["data"]

    def describe_storage(self):
        return (f"chunks={self.data.chunks} compression={self.data.compression} "
                f"nnz={self.data.shape[0]:,} dtype={self.data.dtype}")

    def close(self):
        self.handle.close()


def iterate_rows_in_file_order(reader, sorted_rows, max_elements_per_read=8_000_000):
    """Yield (row, columns, values) for sorted_rows, reading contiguous slabs.

    The rows a screen needs are scattered, but they are dense enough that
    consecutive needed rows usually live within a few file rows of each other.
    Reading the whole span between them in one call and slicing locally costs a
    little extra bandwidth and saves an enormous number of HDF5 calls -- and,
    when the file is compressed, an enormous amount of repeated decompression.
    """
    indptr = reader.indptr
    start = 0
    while start < len(sorted_rows):
        slab_lo = int(indptr[sorted_rows[start]])
        end = start + 1
        while (end < len(sorted_rows)
               and int(indptr[sorted_rows[end] + 1]) - slab_lo <= max_elements_per_read):
            end += 1
        slab_hi = int(indptr[sorted_rows[end - 1] + 1])
        columns_slab = reader.indices[slab_lo:slab_hi]
        values_slab = reader.data[slab_lo:slab_hi]
        for row in sorted_rows[start:end]:
            lo = int(indptr[row]) - slab_lo
            hi = int(indptr[row + 1]) - slab_lo
            yield row, columns_slab[lo:hi], values_slab[lo:hi]
        start = end


def normalise(values, target_sum=1e4):
    """log1p counts-per-target_sum, over nonzeros only. log1p(0) = 0."""
    depth = float(values.sum())
    if depth <= 0:
        return None
    return np.log1p(values.astype(np.float32) * (target_sum / depth))


def read_dense_block(reader, rows, n_genes, target_sum=1e4):
    """Dense (kept_cells x n_genes) block. Used only for the 3,000 controls."""
    block = np.zeros((len(rows), n_genes), dtype=np.float32)
    kept_rows = []
    kept = 0
    for row, columns, values in iterate_rows_in_file_order(reader, np.sort(rows)):
        normalised = normalise(values, target_sum)
        if normalised is None:
            continue
        block[kept, columns] = normalised
        kept_rows.append(row)
        kept += 1
    return block[:kept], np.asarray(kept_rows)


def response_and_variance(perturbed, control_mean, control_variance, n_control):
    """Pseudobulk response, its standard error, and the resulting NSR.

    Under the isotropic squared-Euclidean geometry only variance traces enter --
    never a full covariance matrix -- so the standard error collapses to two
    scalars: se^2 = V_p/n_p + V_ctrl/n_ctrl.
    """
    n_perturbed = perturbed.shape[0]
    response = perturbed.mean(axis=0) - control_mean
    perturbed_variance = float(np.sum(perturbed.var(axis=0, ddof=1))) \
        if n_perturbed > 1 else 0.0
    if n_control <= 0:
        raise ValueError("n_control is zero -- no control cells reached the "
                         "variance estimate.")
    squared_error = perturbed_variance / n_perturbed + control_variance / n_control
    standard_error = float(np.sqrt(max(squared_error, 1e-12)))
    return response, standard_error, float(np.linalg.norm(response)) / standard_error


def nsr_from_accumulators(total_sum, sum_of_squared_norms, n_cells,
                          control_mean, control_variance, n_control):
    """Same quantity as response_and_variance, from streamed accumulators.

    perturbed_variance is the trace identity described in the module docstring.
    """
    mean = total_sum / n_cells
    perturbed_variance = ((sum_of_squared_norms - n_cells * float(mean @ mean))
                          / (n_cells - 1)) if n_cells > 1 else 0.0
    perturbed_variance = max(perturbed_variance, 0.0)
    squared_error = perturbed_variance / n_cells + control_variance / n_control
    standard_error = float(np.sqrt(max(squared_error, 1e-12)))
    response = mean - control_mean
    return response, standard_error, float(np.linalg.norm(response)) / standard_error


def split_half_rho(perturbed, control_mean, rng):
    """Across-gene correlation of responses from two independent cell halves.

    Direction agreement, not magnitude agreement. A relabelled control is a
    reproducibly NULL measurement, so a magnitude-disagreement rule would keep
    fakes; a direction rule drops them.
    """
    order = rng.permutation(perturbed.shape[0])
    midpoint = len(order) // 2
    if midpoint < 2:
        return np.nan
    first = perturbed[order[:midpoint]].mean(axis=0) - control_mean
    second = perturbed[order[midpoint:]].mean(axis=0) - control_mean
    if first.std() < 1e-12 or second.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(first, second)[0, 1])


def rho_from_accumulators(total_sum, first_half_sum, n_cells, n_first_half,
                          control_mean):
    """Same quantity as split_half_rho, from streamed accumulators.

    The half assignment was fixed before the pass, so the second half's sum is
    just the total minus the first half's -- no second accumulator needed.
    """
    n_second_half = n_cells - n_first_half
    if n_first_half < 2 or n_second_half < 2:
        return np.nan
    first = first_half_sum / n_first_half - control_mean
    second = (total_sum - first_half_sum) / n_second_half - control_mean
    if first.std() < 1e-12 or second.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(first, second)[0, 1])


def calibrate_null(control_block, control_mean, control_variance, cell_counts,
                   n_draws, rng):
    """Relabelled-control pseudo-perturbations: pure noise, zero effect.

    The control block is already resident from the reference-mean step, so the
    draws are pure arithmetic. The old version re-read them from disk.
    """
    n_control = control_block.shape[0]
    null_nsr, null_rho = [], []
    for _ in range(n_draws):
        size = int(rng.choice(cell_counts))
        size = min(size, n_control // 2)
        if size < 4:
            continue
        drawn = rng.choice(n_control, size=size, replace=False)
        block = control_block[np.sort(drawn)]
        _, _, nsr = response_and_variance(block, control_mean, control_variance,
                                          n_control)
        null_nsr.append(nsr)
        null_rho.append(split_half_rho(block, control_mean, rng))
    null_rho = [v for v in null_rho if np.isfinite(v)]
    return (float(np.quantile(null_nsr, 0.95)) if null_nsr else 1.0,
            float(np.quantile(null_rho, 0.95)) if null_rho else 0.0)


def rng_for_gene(seed, gene):
    """Deterministic per-perturbation stream. crc32, not hash(): hash() is
    salted per process and would make runs irreproducible."""
    return np.random.default_rng([seed, zlib.crc32(gene.encode())])


def plan_selection_rows(rows_by_gene, max_cells_screen, seed):
    """Choose which cells each perturbation contributes, and which half is A.

    Done up front so the reading pass can be a single ordered sweep, and so the
    subsample depends only on the gene name.
    """
    plan = {}
    for gene, rows in rows_by_gene.items():
        gene_rng = rng_for_gene(seed, gene)
        if len(rows) > max_cells_screen:
            rows = gene_rng.choice(rows, size=max_cells_screen, replace=False)
        rows = np.sort(rows)
        order = gene_rng.permutation(len(rows))
        in_first_half = np.zeros(len(rows), dtype=bool)
        in_first_half[order[: len(rows) // 2]] = True
        plan[gene] = (rows, in_first_half)
    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", required=True)
    parser.add_argument("--guide_calls", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--eval_set", required=True,
                        help="The frozen 200. Reconciliation is against this.")
    parser.add_argument("--old_screen", default=None,
                        help="screen_*_ranked.csv, to compare old vs new selection.")
    parser.add_argument("--dose_bins", type=int, default=2)
    parser.add_argument("--min_cells_per_bin", type=int, default=30,
                        help="Cells required in EVERY dose bin, evaluation half.")
    parser.add_argument("--min_cells_screen", type=int, default=12)
    parser.add_argument("--max_cells_screen", type=int, default=200)
    parser.add_argument("--max_perturbations", type=int, default=0,
                        help="Cap on perturbations screened, for pilots. 0 means "
                             "no cap, which is what the full screen wants. When "
                             "the cap fires it keeps the frozen set and fills the "
                             "remainder at random, so it changes WHICH "
                             "perturbations are screened, not just how many.")
    parser.add_argument("--n_control_cells", type=int, default=3000)
    parser.add_argument("--n_null_draws", type=int, default=300)
    parser.add_argument("--size_matched_null", action="store_true",
                        help="Cap null draw sizes at --max_cells_screen, matching "
                             "what the screen actually reads. The original sized "
                             "draws from uncapped per-gene counts; kept as default "
                             "so cutoffs stay comparable to previous runs.")
    parser.add_argument("--memory_budget_gb", type=float, default=16.0,
                        help="Accumulator budget. Below the requirement the pass "
                             "is split, which costs one extra sweep per split.")
    parser.add_argument("--max_elements_per_read", type=int, default=8_000_000,
                        help="Slab size for contiguous reads, in matrix nonzeros.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="nsr_rho_screen")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    adata = ad.read_h5ad(args.h5ad, backed="r")
    n_genes = int(adata.n_vars)
    obs = adata.obs
    barcode_index = pd.Index(obs.index.to_numpy())
    print(f"[load] {adata.n_obs:,} cells x {n_genes:,} genes", flush=True)

    frozen = set(pd.read_csv(args.eval_set)["gene_target"].astype(str))
    split = pd.read_csv(args.split)
    selection = set(split.loc[split["split"] == "selection", "cell_barcode"])
    evaluation = set(split.loc[split["split"] == "evaluation", "cell_barcode"])
    print(f"[load] frozen eval set: {len(frozen)} perturbations")
    print(f"[load] {len(selection):,} selection / {len(evaluation):,} evaluation cells")

    log("[dose] computing dose bins...")
    dose_bins = compute_dose_bins(obs, args.guide_calls, n_bins=args.dose_bins,
                                  min_cells_per_perturbation=2 * args.min_cells_per_bin,
                                  verbose=False)
    log(f"[dose] done; {int((dose_bins >= 0).sum()):,} cells assigned to a bin")

    genes = obs["gene_target"].astype(str).to_numpy()
    guides = obs["guide_target"].astype(str).to_numpy()
    log("[label] classifying control cells...")
    is_control = np.array([len(g.split("|")) >= 2
                           and all(p.startswith("non-targeting") for p in g.split("|"))
                           for g in guides])
    log(f"[label] {int(is_control.sum()):,} dual non-targeting cells")

    log("[label] matching split assignments...")
    in_selection = barcode_index.isin(selection)
    in_evaluation = barcode_index.isin(evaluation)
    log(f"[label] {int(in_selection.sum()):,} selection / "
        f"{int(in_evaluation.sum()):,} evaluation cells matched to the h5ad")

    # ---- row index, built once ------------------------------------------------
    # Replaces two (genes == gene) scans over all cells per perturbation.
    log("[index] grouping selection rows by perturbation...")
    started = time.time()
    selection_positions = np.flatnonzero(in_selection & ~is_control)
    selection_frame = pd.DataFrame({"row": selection_positions,
                                    "gene": genes[selection_positions]})
    rows_by_gene = {gene: group["row"].to_numpy()
                    for gene, group in selection_frame.groupby("gene", sort=False)}
    rows_by_gene.pop("Non-Targeting", None)

    log(f"[index] {len(rows_by_gene):,} perturbations have selection cells; "
        f"grouping evaluation rows by dose bin...")
    evaluation_positions = np.flatnonzero(in_evaluation & ~is_control)
    evaluation_frame = pd.DataFrame({"gene": genes[evaluation_positions],
                                     "dose_bin": dose_bins[evaluation_positions]})
    evaluation_cell_counts = evaluation_frame.groupby("gene").size()
    per_bin_counts = (evaluation_frame.groupby(["gene", "dose_bin"]).size()
                      .unstack(fill_value=0)
                      .reindex(columns=range(args.dose_bins), fill_value=0))
    smallest_bin = per_bin_counts.min(axis=1)
    log(f"[index] built in {time.time() - started:.0f}s "
        f"({len(rows_by_gene):,} perturbations with selection cells)")

    reader = BackedCSRReader(args.h5ad)
    print(f"[storage] {reader.describe_storage()}", flush=True)
    try:
        control_rows = np.where(is_control & in_selection)[0]
        if len(control_rows) == 0:
            # screen_perturbations.py only split PERTURBED cells, so controls
            # carry no split label. Assign them deterministically here so the
            # reference mean and the null calibration stay on selection cells.
            all_controls = np.where(is_control)[0]
            if len(all_controls) == 0:
                raise RuntimeError("No control cells found under the dual "
                                   "non-targeting rule -- check guide_target.")
            shuffled = rng.permutation(all_controls)
            control_rows = np.sort(shuffled[: len(shuffled) // 2])
            print(f"[control] split file has no controls; assigning "
                  f"{len(control_rows):,} of {len(all_controls):,} to the "
                  f"selection half by seed")
        control_rows = np.sort(rng.choice(control_rows,
                                          size=min(args.n_control_cells, len(control_rows)),
                                          replace=False))
        print(f"[control] reading {len(control_rows):,} selection-half controls...",
              flush=True)
        started = time.time()
        control_block, _ = read_dense_block(reader, control_rows, n_genes)
        control_mean = control_block.mean(axis=0)
        control_variance = float(np.sum(control_block.var(axis=0, ddof=1)))
        n_control = control_block.shape[0]
        print(f"[control] done in {time.time() - started:.0f}s "
              f"(total per-cell variance {control_variance:.1f})", flush=True)

        candidates = pd.Series({gene: len(rows) for gene, rows in rows_by_gene.items()})
        candidates = candidates[candidates >= args.min_cells_screen].sort_values(ascending=False)
        if args.max_perturbations and len(candidates) > args.max_perturbations:
            # sorted(), not list(set), because set iteration order over strings
            # depends on PYTHONHASHSEED and so varies between processes.
            keep = sorted(frozen & set(candidates.index))
            others = [g for g in candidates.index if g not in frozen]
            n_extra = max(0, args.max_perturbations - len(keep))
            keep += list(rng.choice(others, size=min(n_extra, len(others)), replace=False))
            candidates = candidates.loc[keep]
        print(f"[screen] {len(candidates):,} perturbations with "
              f">= {args.min_cells_screen} selection cells "
              f"({len(set(candidates.index) & frozen)} of them frozen)", flush=True)

        null_sizes = candidates.to_numpy()
        if args.size_matched_null:
            null_sizes = np.minimum(null_sizes, args.max_cells_screen)
        print(f"[null] calibrating against {args.n_null_draws} relabelled controls...",
              flush=True)
        started = time.time()
        nsr_cutoff, rho_cutoff = calibrate_null(
            control_block, control_mean, control_variance, null_sizes,
            args.n_null_draws, rng)
        print(f"[null] done in {time.time() - started:.0f}s -- "
              f"NSR p95 = {nsr_cutoff:.3f}, rho p95 = {rho_cutoff:.3f}", flush=True)
        del control_block

        # ---- the sequential pass ---------------------------------------------
        plan = plan_selection_rows({gene: rows_by_gene[gene] for gene in candidates.index},
                                   args.max_cells_screen, args.seed)

        bytes_per_perturbation = 2 * n_genes * 4
        max_perturbations_per_pass = max(
            1, int(args.memory_budget_gb * 1024 ** 3 // bytes_per_perturbation))
        gene_order = list(candidates.index)
        n_passes = int(np.ceil(len(gene_order) / max_perturbations_per_pass))
        print(f"[pass] accumulators need "
              f"{len(gene_order) * bytes_per_perturbation / 1024 ** 3:.1f} GB; "
              f"budget {args.memory_budget_gb:.1f} GB -> {n_passes} pass(es)",
              flush=True)

        records = []
        for pass_index in range(n_passes):
            batch = gene_order[pass_index * max_perturbations_per_pass:
                               (pass_index + 1) * max_perturbations_per_pass]
            slot_of_gene = {gene: slot for slot, gene in enumerate(batch)}

            total_sum = np.zeros((len(batch), n_genes), dtype=np.float32)
            first_half_sum = np.zeros((len(batch), n_genes), dtype=np.float32)
            sum_of_squared_norms = np.zeros(len(batch), dtype=np.float64)
            n_kept = np.zeros(len(batch), dtype=np.int64)
            n_kept_first_half = np.zeros(len(batch), dtype=np.int64)

            # Every needed row, in file order, tagged with its destination slot.
            all_rows, all_slots, all_is_first_half = [], [], []
            for gene in batch:
                rows, in_first_half = plan[gene]
                all_rows.append(rows)
                all_slots.append(np.full(len(rows), slot_of_gene[gene], dtype=np.int32))
                all_is_first_half.append(in_first_half)
            all_rows = np.concatenate(all_rows)
            all_slots = np.concatenate(all_slots)
            all_is_first_half = np.concatenate(all_is_first_half)
            file_order = np.argsort(all_rows, kind="stable")
            all_rows = all_rows[file_order]
            all_slots = all_slots[file_order]
            all_is_first_half = all_is_first_half[file_order]

            print(f"[pass {pass_index + 1}/{n_passes}] reading {len(all_rows):,} cells "
                  f"for {len(batch):,} perturbations", flush=True)
            started = time.time()
            cursor = 0
            for _, columns, values in iterate_rows_in_file_order(
                    reader, all_rows, args.max_elements_per_read):
                normalised = normalise(values)
                if normalised is not None:
                    slot = all_slots[cursor]
                    total_sum[slot, columns] += normalised
                    sum_of_squared_norms[slot] += float(normalised @ normalised)
                    n_kept[slot] += 1
                    if all_is_first_half[cursor]:
                        first_half_sum[slot, columns] += normalised
                        n_kept_first_half[slot] += 1
                cursor += 1
                if cursor % 200_000 == 0:
                    elapsed = time.time() - started
                    rate = cursor / max(elapsed, 1e-9)
                    print(f"[pass {pass_index + 1}/{n_passes}] {cursor:,}/{len(all_rows):,} "
                          f"cells  {rate:,.0f} cells/s  "
                          f"~{(len(all_rows) - cursor) / max(rate, 1e-9) / 60:.0f} min left",
                          flush=True)
            print(f"[pass {pass_index + 1}/{n_passes}] read in "
                  f"{(time.time() - started) / 60:.1f} min", flush=True)

            for slot, gene in enumerate(batch):
                if n_kept[slot] < args.min_cells_screen:
                    continue
                _, standard_error, nsr = nsr_from_accumulators(
                    total_sum[slot].astype(np.float64), sum_of_squared_norms[slot],
                    int(n_kept[slot]), control_mean, control_variance, n_control)
                rho = rho_from_accumulators(
                    total_sum[slot].astype(np.float64),
                    first_half_sum[slot].astype(np.float64),
                    int(n_kept[slot]), int(n_kept_first_half[slot]), control_mean)
                records.append({
                    "gene_target": str(gene),
                    "nsr": float(nsr),
                    "rho": float(rho),
                    "standard_error": float(standard_error),
                    "n_selection_cells": int(n_kept[slot]),
                    "n_evaluation_cells": int(evaluation_cell_counts.get(gene, 0)),
                    "min_cells_in_a_bin": int(smallest_bin.get(gene, 0)),
                    "in_frozen": bool(gene in frozen),
                })
            del total_sum, first_half_sum
    finally:
        reader.close()

    screen = pd.DataFrame(records)
    screen["passes_nsr"] = screen["nsr"] > nsr_cutoff
    screen["passes_rho"] = screen["rho"] > rho_cutoff
    screen["retained"] = screen["passes_nsr"] | screen["passes_rho"]
    screen["rho_rescued"] = screen["passes_rho"] & ~screen["passes_nsr"]
    screen["dose_valid"] = screen["min_cells_in_a_bin"] >= args.min_cells_per_bin
    screen.to_csv(f"{args.out}.csv", index=False)

    total = len(screen)
    frozen_rows = screen[screen["in_frozen"]]
    paper = screen[screen["in_frozen"] & screen["retained"] & screen["dose_valid"]]

    print(f"\n{'=' * 62}")
    print("Kalygina-style screen (NSR OR rho)")
    print("=" * 62)
    print(f"  cutoffs: NSR > {nsr_cutoff:.3f}, rho > {rho_cutoff:.3f}\n")
    print(f"  screened:                      {total:,}")
    print(f"  passes NSR:                    {int(screen['passes_nsr'].sum()):,}")
    print(f"  passes rho:                    {int(screen['passes_rho'].sum()):,}")
    print(f"  passes NSR OR rho:             {int(screen['retained'].sum()):,} "
          f"({screen['retained'].mean():.1%})")
    print(f"  rho-rescued:                   {int(screen['rho_rescued'].sum()):,}")
    print("\n  Kalygina report 22.4% retention on Orion (8,035 of 35,916).")

    print(f"\n  frozen eval set screened:      {len(frozen_rows):,}")
    print(f"    passes NSR OR rho:           {int(frozen_rows['retained'].sum()):,} "
          f"({frozen_rows['retained'].mean():.1%})")
    print(f"    dose-valid (>= {args.min_cells_per_bin} cells in every bin): "
          f"{int(frozen_rows['dose_valid'].sum()):,}")

    print(f"\n  *** EFFECTIVE SAMPLE SIZE: {len(paper):,} ***")
    print("      frozen AND passes screen AND dose-valid.")
    print("      This is the n every model-arm comparison is powered by --")
    print("      not the global retention count.")

    if len(paper) >= 10:
        print(f"\n  At the PCA pilot's gap SD of 0.596, n={len(paper)} resolves an "
              f"effect of\n  {2.8 * 0.596 / np.sqrt(len(paper)):.3f} at 80% power "
              f"(ground-truth dose effect ~0.073).")

    if args.old_screen:
        old = pd.read_csv(args.old_screen)
        old_strong = set(old.loc[old["effect_size_unbiased"] >= 2.5, "gene_target"])
        new_kept = set(screen.loc[screen["retained"], "gene_target"])
        print(f"\n  old effect-size screen (>= 2.5): {len(old_strong):,}")
        print(f"    old AND new:                   {len(old_strong & new_kept):,}")
        print(f"    old only:                      {len(old_strong - new_kept):,}")
        print(f"    new only:                      {len(new_kept - old_strong):,}")

    with open(f"{args.out}_summary.json", "w") as handle:
        json.dump({
            "nsr_cutoff": nsr_cutoff, "rho_cutoff": rho_cutoff,
            "screened": total, "retained": int(screen["retained"].sum()),
            "retained_fraction": float(screen["retained"].mean()),
            "rho_rescued": int(screen["rho_rescued"].sum()),
            "frozen_screened": len(frozen_rows),
            "frozen_retained": int(frozen_rows["retained"].sum()),
            "effective_sample_size": len(paper),
            "config": vars(args),
        }, handle, indent=2)
    print(f"\n[save] {args.out}.csv, {args.out}_summary.json")


if __name__ == "__main__":
    main()