#!/usr/bin/env python
import os as _os

_lr = int(_os.environ.get("LOCAL_RANK", "0"))
if "RANK" in _os.environ:
    _os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    _os.environ["CUDA_VISIBLE_DEVICES"] = str(_lr)

_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")

import os
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from scipy import sparse
import pandas as pd
import anndata as ad
from scipy.stats import rankdata

# ---------------- CONFIG ----------------
RUN_NAME = os.environ.get("RUN_NAME", "xatlas_scgpt_eval_tuplemean_verbose")
SAVE_DIR = Path(f"./save/{RUN_NAME}")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

H5AD_A = os.environ.get("H5AD_A")
H5AD_B = os.environ.get("H5AD_B")
LOAD_MODEL_DIR = os.environ.get("SCGPT_DIR")

MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "3000"))
EVAL_BSZ = int(os.environ.get("EVAL_BSZ", "256"))
MICRO_BSZ = int(os.environ.get("MICRO_BSZ", "16"))
SEED = int(os.environ.get("SEED", "42"))

SUBSAMPLE = int(os.environ.get("SUBSAMPLE", "-1"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "0"))
MIN_CTRL_PER_SAMPLE = int(os.environ.get("MIN_CTRL_PER_SAMPLE", "5"))

VAR_GENE_NAME = os.environ.get("VAR_GENE_NAME", "gene_name")
VAR_GENE_ID = os.environ.get("VAR_GENE_ID", "gene_id")

# StructuralIntegrity cellwise/batchwise options
# Paper says "log-transformed" profiles; keep default ON.
SI_LOG1P = int(os.environ.get("SI_LOG1P", "1"))

# DO NOT make assumptions: user stated these columns exist.
OBS_CELL_BARCODE = "cell_barcode"
OBS_SAMPLE = "sample"
OBS_GUIDE_TARGET = "guide_target"
OBS_GENE_TARGET = "gene_target"

# Batch-based logging; default ~20000 tuples worth of progress
LOG_EVERY = int(os.environ.get("LOG_EVERY", str(max(1, int(round(20000 / float(EVAL_BSZ)))))))

USE_AMP = int(os.environ.get("USE_AMP", "0"))
AMP_DTYPE = os.environ.get("AMP_DTYPE", "fp16")

COMPUTE_SPEARMAN = int(os.environ.get("COMPUTE_SPEARMAN", "1"))

PROFILE_FIRST_BATCH = int(os.environ.get("PROFILE_FIRST_BATCH", "1"))
PROFILE_CUDA_EVENTS = int(os.environ.get("PROFILE_CUDA_EVENTS", "1"))
PRINT_FIRST_BATCH_SAMPLES = int(os.environ.get("PRINT_FIRST_BATCH_SAMPLES", "1"))

# Verbosity controls
PERT_BUILD_LOG_EVERY_SAMPLES = int(os.environ.get("PERT_BUILD_LOG_EVERY_SAMPLES", "5"))
CTRL_MEAN_LOG_EVERY_SAMPLES = int(os.environ.get("CTRL_MEAN_LOG_EVERY_SAMPLES", "10"))
ROWSUM_HEARTBEAT_EVERY_CHUNKS = int(os.environ.get("ROWSUM_HEARTBEAT_EVERY_CHUNKS", "0"))  # 0 disables
# ---------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def init_distributed() -> bool:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        return True
    return False


_ddp = init_distributed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANK = dist.get_rank() if dist.is_initialized() else 0
WS = dist.get_world_size() if dist.is_initialized() else 1


def is_main():
    return RANK == 0


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _rss_gb() -> float:
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = float(line.split()[1])
                    return kb / 1024.0 / 1024.0
    except Exception:
        pass
    return float("nan")


def rlog(msg: str):
    print(f"[{RANK}/{WS}] {msg}", flush=True)


@contextmanager
def timed(msg: str, enabled: bool = True, sync_cuda: bool = False):
    if not enabled:
        yield
        return
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    rlog(f"[TIMER] {_now()} START {msg} | RSS={_rss_gb():.2f} GB")
    try:
        yield
    finally:
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        rlog(f"[TIMER] {_now()} END   {msg}  ({dt:.6f}s) | RSS={_rss_gb():.2f} GB")


def cuda_event_timing(enabled: bool):
    if not enabled or (not torch.cuda.is_available()):
        return None, None, False
    try:
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        return s, e, True
    except Exception:
        return None, None, False


if torch.cuda.is_available():
    rlog(f"GPU: {torch.cuda.get_device_name(0)} | VRAM(GB)={torch.cuda.get_device_properties(0).total_memory/1024**3:.2f}")

with timed("import scgpt / model class load", enabled=True):
    import scgpt as scg  # noqa: F401
    from scgpt.model.generation_model import TransformerGenerator
    from scgpt.tokenizer.gene_tokenizer import GeneVocab

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    from torch.backends.cuda import sdp_kernel
    sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)
except Exception:
    pass

PAD_TOKEN = "<pad>"
SPECIAL_TOKENS = [PAD_TOKEN, "<cls>", "<eoc>"]

with timed("load scgpt args.json + vocab + construct model", enabled=True):
    with open(Path(LOAD_MODEL_DIR) / "args.json", "r") as f:
        cfg = json.load(f)

    vocab = GeneVocab.from_file(Path(LOAD_MODEL_DIR) / "vocab.json")
    for s in SPECIAL_TOKENS:
        if s not in vocab:
            vocab.append_token(s)
    vocab.set_default_index(vocab[PAD_TOKEN])

    model = TransformerGenerator(
        len(vocab),
        cfg["embsize"],
        cfg["nheads"],
        cfg["d_hid"],
        cfg["nlayers"],
        nlayers_cls=cfg["n_layers_cls"],
        n_cls=1,
        vocab=vocab,
        dropout=0.0,
        pad_token=PAD_TOKEN,
        pad_value=0,
        pert_pad_id=0,
        use_fast_transformer=False,
        do_mvc=True,
    )

with timed("load weights best_model.pt", enabled=True):
    weights_path = Path(LOAD_MODEL_DIR) / "best_model.pt"
    rlog(f"Loading weights from {weights_path}")
    pre = torch.load(weights_path, map_location="cpu")
    ms = model.state_dict()
    pre_f = {k: v for k, v in pre.items() if (k in ms and v.shape == ms[k].shape)}
    ms.update(pre_f)
    model.load_state_dict(ms)

with timed("move model to device + wrap DDP (if enabled)", enabled=True):
    model.to(device)
    if _ddp:
        model = DDP(
            model,
            device_ids=[0],
            output_device=0,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

GENE_INFO_CSV = os.environ.get("GENE_INFO_CSV", "gene_info.csv")
ensg_to_symbol: Dict[str, str] = {}
with timed("load gene_info.csv mapping (optional)", enabled=True):
    try:
        gene_info_df = pd.read_csv(GENE_INFO_CSV)
        ensg_to_symbol = dict(
            zip(gene_info_df["feature_id"].astype(str), gene_info_df["feature_name"].astype(str))
        )
        if is_main():
            rlog(f"Loaded {len(ensg_to_symbol)} Ensembl→symbol mappings from {GENE_INFO_CSV}")
    except Exception as e:
        ensg_to_symbol = {}
        if is_main():
            rlog(f"No gene_info mapping used ({e})")


def _rows_sum_to_dense(X, idxs, G, chunk=2048):
    acc = np.zeros(G, dtype=np.float64)
    n = 0
    idxs = np.asarray(idxs, dtype=np.int64)

    hb = int(ROWSUM_HEARTBEAT_EVERY_CHUNKS)
    hb_stride = chunk * hb if hb > 0 else None

    for s in range(0, idxs.size, chunk):
        part = idxs[s: s + chunk]
        Xm = X[part]
        if sparse.issparse(Xm):
            Xm = Xm.tocsr()
            acc += np.asarray(Xm.sum(axis=0)).ravel()
        else:
            acc += np.asarray(Xm).sum(axis=0).ravel()
        n += int(part.size)

        if hb_stride is not None and (n % hb_stride == 0) and is_main():
            rlog(f"[ROWS-SUM] processed {n:,}/{idxs.size:,} rows | RSS={_rss_gb():.2f} GB")

    return acc, n


def _get_gene_names(adata):
    if VAR_GENE_NAME in adata.var.columns:
        raw = adata.var[VAR_GENE_NAME].astype(str).to_numpy()
    elif VAR_GENE_ID in adata.var.columns:
        raw = adata.var[VAR_GENE_ID].astype(str).to_numpy()
    else:
        raw = adata.var.index.astype(str).to_numpy()
    if ensg_to_symbol:
        return np.array([ensg_to_symbol.get(g, g) for g in raw], dtype=str)
    return raw


def pearson_exact_rowwise_torch(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x0 = x - x.mean(dim=1, keepdim=True)
    y0 = y - y.mean(dim=1, keepdim=True)
    num = (x0 * y0).sum(dim=1)
    den = torch.sqrt((x0 * x0).sum(dim=1) * (y0 * y0).sum(dim=1)).clamp_min(eps)
    vx = (x0 * x0).sum(dim=1)
    vy = (y0 * y0).sum(dim=1)
    out = num / den
    out = torch.where((vx <= eps) | (vy <= eps), torch.full_like(out, float("nan")), out)
    return out


def spearman_exact_batch(pred_np: np.ndarray, y_np: np.ndarray) -> np.ndarray:
    B = pred_np.shape[0]
    out = np.empty(B, dtype=np.float64)
    for i in range(B):
        rx = rankdata(pred_np[i], method="average")
        ry = rankdata(y_np[i], method="average")
        x0 = rx - rx.mean()
        y0 = ry - y0.mean() if False else (ry - ry.mean())
        denom = np.sqrt(np.dot(x0, x0) * np.dot(y0, y0))
        out[i] = (np.dot(x0, y0) / denom) if denom > 0 else np.nan
    return out.astype(np.float32)


class Agg:
    __slots__ = ("sum", "n")
    def __init__(self):
        self.sum = 0.0
        self.n = 0
    def add_sum_count(self, s: float, c: int):
        if c <= 0:
            return
        if np.isfinite(s):
            self.sum += float(s)
            self.n += int(c)
    def mean(self):
        return float(self.sum / self.n) if self.n > 0 else float("nan")


def avg_across_perturbations_unweighted(per_pert_output: Dict[str, Dict]) -> Dict[str, float]:
    keys = ["MAE", "MSE", "RMSE", "Pearson", "Spearman", "CosineLogFC", "StructuralIntegrity", "StructuralSignAgreement"]
    out = {}
    for k in keys:
        vals = []
        for v in per_pert_output.values():
            x = v.get(k, float("nan"))
            if isinstance(x, (int, float)) and np.isfinite(x):
                vals.append(float(x))
        out[k] = float(np.mean(vals)) if len(vals) > 0 else float("nan")
    out["n_perturbations_included"] = int(sum(
        1 for v in per_pert_output.values()
        if isinstance(v.get("MSE", float("nan")), (int, float)) and np.isfinite(v.get("MSE", float("nan")))
    ))
    return out


def avg_across_perturbations_weighted_by_controls(per_pert_output: Dict[str, Dict]) -> Dict[str, float]:
    keys = ["MAE", "MSE", "RMSE", "Pearson", "Spearman", "CosineLogFC", "StructuralIntegrity", "StructuralSignAgreement"]

    wsum = {k: 0.0 for k in keys}
    wtot = 0.0
    n_total = 0
    n_wpos = 0

    for v in per_pert_output.values():
        n_total += 1
        w = v.get("n_controls_for_sample", 0)
        try:
            w = float(w)
        except Exception:
            w = 0.0
        if w <= 0:
            continue
        n_wpos += 1
        for k in keys:
            x = v.get(k, float("nan"))
            if isinstance(x, (int, float)) and np.isfinite(x):
                wsum[k] += float(x) * w
        wtot += w

    out = {}
    for k in keys:
        out[k] = (wsum[k] / wtot) if wtot > 0 else float("nan")
    out["n_perturbations_total"] = int(n_total)
    out["n_perturbations_with_positive_weight"] = int(n_wpos)
    out["total_weight_controls"] = float(wtot)
    return out


def shard_indices(idxs: np.ndarray) -> np.ndarray:
    if WS <= 1:
        return idxs
    return idxs[(np.arange(idxs.size) % WS) == RANK]


class IndexDataset(torch.utils.data.Dataset):
    def __init__(self, indices: np.ndarray):
        self.indices = np.asarray(indices, dtype=np.int64)
    def __len__(self):
        return int(self.indices.size)
    def __getitem__(self, i):
        return int(self.indices[int(i)])


def _is_dual_nontargeting(guide_s: str) -> bool:
    gs = str(guide_s).split("|")
    return (len(gs) >= 2) and all(g.startswith("non-targeting") for g in gs)


def _compute_ctrl_means_per_sample(X, G, sample_arr, ctrl_idx) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    ctrl_means: Dict[str, np.ndarray] = {}
    ctrl_counts: Dict[str, int] = {}

    ctrl_samples = sample_arr[ctrl_idx]
    order = np.argsort(ctrl_samples, kind="mergesort")
    ctrl_idx_sorted = ctrl_idx[order]
    ctrl_samples_sorted = ctrl_samples[order]

    if ctrl_samples_sorted.size > 0:
        changes = np.where(ctrl_samples_sorted[1:] != ctrl_samples_sorted[:-1])[0] + 1
        starts = np.concatenate([[0], changes])
        ends = np.concatenate([changes, [ctrl_samples_sorted.size]])

        for i, (s0, s1) in enumerate(zip(starts, ends)):
            sid = str(ctrl_samples_sorted[s0])
            idxs_s = ctrl_idx_sorted[s0:s1]

            if is_main() and (i % CTRL_MEAN_LOG_EVERY_SAMPLES == 0):
                rlog(f"[CTRL-MEAN] {i+1}/{len(starts)} sample={sid} | ctrl_cells={idxs_s.size:,} | RSS={_rss_gb():.2f} GB")

            ssum, scnt = _rows_sum_to_dense(X, idxs_s, G)
            if scnt > 0:
                ctrl_means[sid] = (ssum / float(scnt)).astype(np.float32, copy=False)
                ctrl_counts[sid] = int(scnt)

    return ctrl_means, ctrl_counts


def _build_perturbation_groups_tuple_streamed(
    X, G,
    sample_arr: np.ndarray,
    guide_arr: np.ndarray,
    gene_arr: np.ndarray,
    pert_idx: np.ndarray,
    chunk=256,                 # smaller chunk lowers peak RSS
    log_every_samples=5,        # progress prints
):
    """
    Tuple aggregation:
    - Group perturbed cells by sample
    - Within each sample, group by (guide_target, gene_target)
    - Compute mean expression per (sample, guide_target, gene_target)
    """
    delim = "\x1f"

    ps = sample_arr[pert_idx].astype(str, copy=False)
    order_s = np.argsort(ps, kind="mergesort")
    pert_idx_sorted = pert_idx[order_s]
    ps_sorted = ps[order_s]

    if ps_sorted.size == 0:
        return [], [], [], [], [], []

    changes = np.where(ps_sorted[1:] != ps_sorted[:-1])[0] + 1
    starts = np.concatenate([[0], changes])
    ends = np.concatenate([changes, [ps_sorted.size]])

    keys_list: List[str] = []
    samples_list: List[str] = []
    guides_list: List[str] = []
    genes_list: List[str] = []
    n_cells_list: List[int] = []
    mean_expr_list: List[np.ndarray] = []

    n_samples = int(len(starts))
    t0 = time.time()
    tuples_before = 0

    for si, (s0, s1) in enumerate(zip(starts, ends)):
        sid = str(ps_sorted[s0])
        idxs_s = pert_idx_sorted[s0:s1]
        if idxs_s.size == 0:
            continue

        if is_main() and (si % max(1, log_every_samples) == 0):
            elapsed = time.time() - t0
            made = len(keys_list) - tuples_before
            tuples_before = len(keys_list)
            rlog(
                f"[PERT-BUILD] sample {si+1}/{n_samples} sample={sid} | "
                f"pert_cells_in_sample={idxs_s.size:,} | "
                f"tuples_total={len(keys_list):,} (+{made:,}) | "
                f"elapsed={elapsed:.1f}s | RSS={_rss_gb():.2f} GB"
            )

        g_s = guide_arr[idxs_s].astype(str, copy=False)
        ge_s = gene_arr[idxs_s].astype(str, copy=False)

        kg = np.fromiter((f"{g}{delim}{ge}" for g, ge in zip(g_s, ge_s)), dtype=object, count=idxs_s.size)

        order_kg = np.argsort(kg, kind="mergesort")
        idxs_s2 = idxs_s[order_kg]
        kg2 = kg[order_kg]

        ch2 = np.where(kg2[1:] != kg2[:-1])[0] + 1
        st2 = np.concatenate([[0], ch2])
        en2 = np.concatenate([ch2, [kg2.size]])

        for a0, a1 in zip(st2, en2):
            idxs = idxs_s2[a0:a1]
            if idxs.size == 0:
                continue

            ssum, scnt = _rows_sum_to_dense(X, idxs, G, chunk=chunk)
            if scnt <= 0:
                continue

            meanv = (ssum / float(scnt)).astype(np.float32, copy=False)

            gg = str(kg2[a0])
            parts = gg.split(delim)
            if len(parts) == 2:
                guide_t, gene_t = parts[0], parts[1]
            else:
                guide_t = str(guide_arr[idxs[0]])
                gene_t = str(gene_arr[idxs[0]])

            full_key = f"{sid}{delim}{guide_t}{delim}{gene_t}"

            keys_list.append(full_key)
            samples_list.append(sid)
            guides_list.append(guide_t)
            genes_list.append(gene_t)
            n_cells_list.append(int(scnt))
            mean_expr_list.append(meanv)

    if is_main():
        elapsed = time.time() - t0
        rlog(
            f"[PERT-BUILD] DONE | samples={n_samples} | "
            f"pert_cells_total={pert_idx.size:,} | tuples_total={len(keys_list):,} | "
            f"elapsed={elapsed:.1f}s | RSS={_rss_gb():.2f} GB"
        )

    return keys_list, samples_list, guides_list, genes_list, n_cells_list, mean_expr_list


def prepare_adata(h5_path: str) -> Dict:
    with timed(f"prepare_adata: open backed AnnData ({h5_path})", enabled=True):
        adata = ad.read_h5ad(h5_path, backed="r")
        rlog(f"Cells {adata.n_obs:,} | Genes {adata.n_vars:,}")

    for col in [OBS_SAMPLE, OBS_GUIDE_TARGET, OBS_GENE_TARGET]:
        if col not in adata.obs.columns:
            raise ValueError(f"Missing obs['{col}']")

    if OBS_CELL_BARCODE not in adata.obs.columns:
        rlog(f"Warning: obs['{OBS_CELL_BARCODE}'] missing; using adata.obs_names as cell ids.")
        cell_id_col = "__obs_names__"
    else:
        cell_id_col = OBS_CELL_BARCODE

    with timed("prepare_adata: gene name mapping + vocab coverage", enabled=True):
        gene_names = _get_gene_names(adata)
        mapped_mask = np.array([g in vocab for g in gene_names], dtype=bool)
        rlog(f"Vocab coverage: {int(mapped_mask.sum())}/{int(mapped_mask.size)} ({(mapped_mask.mean()*100.0):.2f}%)")
        gids_np = np.array([vocab[g] if g in vocab else vocab[PAD_TOKEN] for g in gene_names], dtype=np.int64)

    X = adata.X
    G = int(adata.n_vars)

    sample_arr = adata.obs[OBS_SAMPLE].astype(str).to_numpy()
    guide_arr = adata.obs[OBS_GUIDE_TARGET].astype(str).to_numpy()
    gene_arr = adata.obs[OBS_GENE_TARGET].astype(str).to_numpy()

    with timed("prepare_adata: build control mask from guide_target (dual non-targeting)", enabled=True):
        is_control = np.zeros(adata.n_obs, dtype=bool)
        for i in range(adata.n_obs):
            is_control[i] = _is_dual_nontargeting(guide_arr[i])

        if "pass_guide_filter" in adata.obs.columns:
            mask_good = (adata.obs["pass_guide_filter"].to_numpy() == 1)
        else:
            mask_good = np.ones(adata.n_obs, dtype=bool)

        is_perturbed = (~is_control) & mask_good

        ctrl_idx = np.where(is_control)[0].astype(np.int64)
        pert_idx = np.where(is_perturbed)[0].astype(np.int64)

        if ctrl_idx.size == 0:
            raise RuntimeError("No control cells found under dual non-targeting rule.")
        if pert_idx.size == 0:
            raise RuntimeError("No perturbed cells found after filters.")

        all_samples = np.unique(sample_arr)
        rlog(f"[DEBUG] TOTAL unique samples in dataset: {int(all_samples.size)}")

        uniq_ctrl, ctrl_counts_np = np.unique(sample_arr[is_control], return_counts=True)
        ctrl_count_map = dict(zip(uniq_ctrl.tolist(), ctrl_counts_np.tolist()))
        counts_all = np.fromiter((int(ctrl_count_map.get(s, 0)) for s in all_samples), dtype=np.int64, count=all_samples.size)
        n_ge = int((counts_all >= MIN_CTRL_PER_SAMPLE).sum())
        rlog(f"[DEBUG] Samples with >= {MIN_CTRL_PER_SAMPLE} controls (OUT OF ALL samples): {n_ge}/{int(all_samples.size)}")
        rlog("[DEBUG] First 5 samples overall + control counts:")
        for s in all_samples[:5]:
            rlog(f"    {s}: {int(ctrl_count_map.get(s, 0))}")

        rlog(f"[DEBUG] Example CONTROL guide_target: {guide_arr[is_control][:5].tolist()}")
        rlog(f"[DEBUG] Example NON-CONTROL guide_target: {guide_arr[~is_control][:5].tolist()}")

    with timed(f"prepare_adata: global control mean over {ctrl_idx.size:,} cells", enabled=True):
        sum_global, n_global = _rows_sum_to_dense(X, ctrl_idx, G, chunk=2048)
        global_mean = (sum_global / float(n_global)).astype(np.float32, copy=False)

    with timed("prepare_adata: per-sample control means (dual non-targeting controls)", enabled=True):
        ctrl_means, ctrl_counts = _compute_ctrl_means_per_sample(X, G, sample_arr, ctrl_idx)
        good_samples = sum(1 for c in ctrl_counts.values() if c >= MIN_CTRL_PER_SAMPLE)
        rlog(f"Samples with >= {MIN_CTRL_PER_SAMPLE} controls (among samples with ANY controls): {good_samples:,}/{len(ctrl_counts):,}")

    with timed("prepare_adata: build perturbation tuples (guide,gene,sample) and mean pert expression", enabled=True):
        keys_list, samples_list, guides_list, genes_list, n_cells_list, mean_expr_list = _build_perturbation_groups_tuple_streamed(
            X, G, sample_arr, guide_arr, gene_arr, pert_idx,
            chunk=256,
            log_every_samples=PERT_BUILD_LOG_EVERY_SAMPLES,
        )
        n_perts = int(len(keys_list))
        rlog(f"Built {n_perts:,} perturbation-tuples (guide,gene,sample) from {pert_idx.size:,} perturbed cells")
        if n_perts == 0:
            raise RuntimeError("No perturbation tuples constructed (unexpected).")

    pert_ids = np.arange(len(keys_list), dtype=np.int64)
    if SUBSAMPLE > 0 and SUBSAMPLE < pert_ids.size:
        pert_ids = np.random.choice(pert_ids, size=SUBSAMPLE, replace=False).astype(np.int64)

    gene_name_to_idx = {g: i for i, g in enumerate(gene_names)}
    target_idx_per_pert = np.full(len(keys_list), -1, dtype=np.int64)
    missing = 0
    for i, gname in enumerate(genes_list):
        gi = int(gene_name_to_idx.get(str(gname), -1))
        target_idx_per_pert[i] = gi
        if gi < 0:
            missing += 1
    rlog(f"[DEBUG] target gene missing from var for {missing:,}/{len(keys_list):,} tuples")

    # --------- NEW: cell -> tuple id mapping, grouped by sample (for cellwise SI) ---------
    delim = "\x1f"
    key_to_pid = {k: i for i, k in enumerate(keys_list)}

    # For each perturbed cell, map to pid (tuple). Also group cells by sample.
    pert_cell_pid = np.full(adata.n_obs, -1, dtype=np.int64)
    pert_cells_by_sample: Dict[str, List[int]] = defaultdict(list)

    # Build keys for perturbed cells and map.
    for ci in pert_idx.tolist():
        sid = str(sample_arr[ci])
        g = str(guide_arr[ci])
        ge = str(gene_arr[ci])
        k = f"{sid}{delim}{g}{delim}{ge}"
        pid = int(key_to_pid.get(k, -1))
        if pid >= 0:
            pert_cell_pid[ci] = pid
            pert_cells_by_sample[sid].append(int(ci))

    # Convert lists to numpy arrays for faster slicing.
    pert_cells_by_sample_np: Dict[str, np.ndarray] = {}
    for sid, lst in pert_cells_by_sample.items():
        pert_cells_by_sample_np[sid] = np.asarray(lst, dtype=np.int64)

    baseline_cache: Dict[str, np.ndarray] = {"__GLOBAL__": global_mean}

    return dict(
        adata=adata,
        X=X,
        G=G,
        gids_np=gids_np,
        gene_names=gene_names,

        sample_arr=sample_arr,
        guide_arr=guide_arr,
        gene_arr=gene_arr,

        global_mean=global_mean,
        ctrl_means=ctrl_means,
        ctrl_counts=ctrl_counts,
        baseline_cache=baseline_cache,

        pert_keys=keys_list,
        pert_samples=samples_list,
        pert_guides=guides_list,
        pert_genes=genes_list,
        pert_n_cells=n_cells_list,
        pert_mean_expr=mean_expr_list,
        pert_target_idx=target_idx_per_pert,

        eval_pert_ids=pert_ids,
        cell_id_col=cell_id_col,
        pert_idx_size=int(pert_idx.size),
        ctrl_idx_size=int(ctrl_idx.size),

        # NEW
        pert_cell_pid=pert_cell_pid,
        pert_cells_by_sample=pert_cells_by_sample_np,
    )


def choose_sample_ids(y_full_t: torch.Tensor, target_idx_t: torch.Tensor, max_seq_len: int) -> torch.Tensor:
    B, G = y_full_t.shape

    nonzero_mask = (y_full_t != 0).any(dim=0)
    candidate_ids = nonzero_mask.nonzero(as_tuple=False).view(-1)
    if candidate_ids.numel() == 0:
        candidate_ids = torch.arange(G, device=y_full_t.device)

    target_ids = target_idx_t[target_idx_t >= 0]
    target_ids = torch.unique(target_ids) if target_ids.numel() > 0 else torch.empty(0, dtype=torch.long, device=y_full_t.device)

    if target_ids.numel() >= max_seq_len:
        perm = torch.randperm(target_ids.numel(), device=y_full_t.device)
        return target_ids[perm[:max_seq_len]]

    if target_ids.numel() > 0:
        keep = torch.ones(G, dtype=torch.bool, device=y_full_t.device)
        keep[target_ids] = False
        cand = candidate_ids[keep[candidate_ids]]
    else:
        cand = candidate_ids

    need = max_seq_len - int(target_ids.numel())
    if cand.numel() > need:
        perm = torch.randperm(cand.numel(), device=y_full_t.device)
        fill = cand[perm[:need]]
    else:
        fill = cand

    sample_ids = torch.cat([target_ids, fill], dim=0)
    if sample_ids.numel() > max_seq_len:
        sample_ids = sample_ids[:max_seq_len]

    if target_ids.numel() > 0:
        if not torch.all(torch.isin(target_ids, sample_ids)):
            raise RuntimeError("choose_sample_ids: missing at least one target gene id in sample_ids (bug).")

    return sample_ids


def build_baseline_sub_and_pflags_sub_for_perts(
    pert_ids: np.ndarray,
    sample_ids_np: np.ndarray,
    pert_samples: List[str],
    pert_target_idx: np.ndarray,
    ctrl_means: Dict[str, np.ndarray],
    ctrl_counts: Dict[str, int],
    global_mean: np.ndarray,
    baseline_cache: Dict[str, np.ndarray],
    min_ctrl_per_sample: int,
) -> Tuple[np.ndarray, np.ndarray]:
    B = int(pert_ids.size)
    S = int(sample_ids_np.size)

    baseline_sub = np.zeros((B, S), dtype=np.float32)
    pflags_sub = np.zeros((B, S), dtype=np.float32)

    row_target_idx = pert_target_idx[pert_ids].astype(np.int64, copy=False)

    for j, pid in enumerate(pert_ids.tolist()):
        sid = str(pert_samples[pid])

        bc = baseline_cache.get(sid, None)
        if bc is None:
            n_ctrl = int(ctrl_counts.get(sid, 0))
            if n_ctrl >= min_ctrl_per_sample and sid in ctrl_means:
                bc = ctrl_means[sid]
            else:
                bc = global_mean
            baseline_cache[sid] = bc

        baseline_sub[j, :] = bc[sample_ids_np]

        gi = int(row_target_idx[j])
        if gi >= 0:
            hits = np.where(sample_ids_np == gi)[0]
            if hits.size > 0:
                pflags_sub[j, hits[0]] = 1.0

    return baseline_sub, pflags_sub


def forward_microbatched(unwrapped, tokens_sub, vals_sub, pflags_sub, padmask, micro_bsz: int):
    B = tokens_sub.shape[0]
    outs = []

    if USE_AMP and torch.cuda.is_available():
        autocast_dtype = torch.bfloat16 if AMP_DTYPE.lower() == "bf16" else torch.float16
        autocast_ctx = torch.autocast(device_type="cuda", dtype=autocast_dtype)
    else:
        class _Null:
            def __enter__(self): return None
            def __exit__(self, *args): return False
        autocast_ctx = _Null()

    for s in range(0, B, micro_bsz):
        e = min(B, s + micro_bsz)
        with autocast_ctx:
            out = unwrapped(
                tokens_sub[s:e],
                vals_sub[s:e],
                pflags_sub[s:e].long(),
                src_key_padding_mask=padmask[s:e],
                CLS=False, CCE=False, MVC=False, ECS=False,
                do_sample=False,
            )
            outs.append(out["mlm_output"].detach())
    return torch.cat(outs, dim=0)


def _dense_rows_for_indices(X, row_idxs: np.ndarray, col_idxs: np.ndarray):
    """
    Read X[row_idxs, col_idxs] into a dense float32 array (handles sparse/backed).
    """
    Xm = X[row_idxs][:, col_idxs]
    if sparse.issparse(Xm):
        Xm = Xm.tocsr()
        return np.asarray(Xm.toarray(), dtype=np.float32)
    return np.asarray(Xm, dtype=np.float32)


def compute_structural_integrity_cellwise_batchwise(
    *,
    tag: str,
    unwrapped,
    prep: Dict,
):
    """
    Implements the requested SI:
    - batch = sample (sid)
    - rows = perturbed CELLS in that sample
    - Ypred row repeats the tuple-level prediction for the tuple the cell belongs to
    - centering is log1p(expr) - log1p(control_mean_for_sample)
    - SD = (1/B) sum_b (1/nb) || (Ypred_center - Yact_center) ||_F
    - SDmax = (2/B) sum_b (1/nb) || Yact_center ||_F
    - SI = 1 - SD/SDmax
    Unweighted across batches.
    """
    X = prep["X"]
    G = prep["G"]
    gids_np = prep["gids_np"]

    pert_samples = prep["pert_samples"]
    pert_mean_expr = prep["pert_mean_expr"]
    pert_target_idx = prep["pert_target_idx"]

    ctrl_means = prep["ctrl_means"]
    ctrl_counts = prep["ctrl_counts"]
    global_mean = prep["global_mean"]
    baseline_cache = prep["baseline_cache"]

    pert_cell_pid = prep["pert_cell_pid"]
    pert_cells_by_sample = prep["pert_cells_by_sample"]

    gids_1d = torch.as_tensor(gids_np, dtype=torch.long, device=device)

    # Group tuple ids by sample
    tuples_by_sample: Dict[str, List[int]] = defaultdict(list)
    for pid, sid in enumerate(pert_samples):
        tuples_by_sample[str(sid)].append(int(pid))

    samples = sorted(set(pert_cells_by_sample.keys()))
    if is_main():
        rlog(f"[SI-CELL] computing cellwise/batchwise SI over {len(samples)} samples (batches). SI_LOG1P={SI_LOG1P}")

    sum_sd = 0.0
    sum_sdmax = 0.0
    B_used = 0

    # Iterate over samples/batches
    for bi, sid in enumerate(samples):
        cell_idxs = pert_cells_by_sample.get(sid, None)
        if cell_idxs is None or cell_idxs.size == 0:
            continue

        # All tuple ids in this sample
        pids = tuples_by_sample.get(sid, None)
        if pids is None or len(pids) == 0:
            continue

        pids_np = np.asarray(pids, dtype=np.int64)
        nb = int(cell_idxs.size)

        # Select baseline for this sample (same logic as before)
        bc = baseline_cache.get(sid, None)
        if bc is None:
            n_ctrl = int(ctrl_counts.get(sid, 0))
            if n_ctrl >= MIN_CTRL_PER_SAMPLE and sid in ctrl_means:
                bc = ctrl_means[sid]
            else:
                bc = global_mean
            baseline_cache[sid] = bc

        # Build y_full over tuples (mean expression vectors) for choose_sample_ids (to include targets)
        y_full_np = np.zeros((pids_np.size, G), dtype=np.float32)
        for j, pid in enumerate(pids_np.tolist()):
            y_full_np[j, :] = pert_mean_expr[pid]
        y_full_t = torch.from_numpy(y_full_np).to(device)

        target_idx_t = torch.from_numpy(pert_target_idx[pids_np].astype(np.int64, copy=False)).to(device)

        # One shared gene subset for this SAMPLE (sid)
        sample_ids = choose_sample_ids(y_full_t, target_idx_t, MAX_SEQ_LEN)
        sample_ids_np = sample_ids.detach().cpu().numpy().astype(np.int64, copy=False)
        S = int(sample_ids_np.size)

        # Predict each tuple in this sample once (cache by pid)
        # Baseline values for these genes
        base_vec = bc[sample_ids_np].astype(np.float32, copy=False)

        # Build baseline_sub and pflags for these tuple ids (same helper)
        baseline_sub_np, pflags_sub_np = build_baseline_sub_and_pflags_sub_for_perts(
            pids_np, sample_ids_np,
            pert_samples,
            pert_target_idx,
            ctrl_means,
            ctrl_counts,
            global_mean,
            baseline_cache,
            MIN_CTRL_PER_SAMPLE,
        )

        tokens_sub = gids_1d[sample_ids].unsqueeze(0).expand(pids_np.size, -1).contiguous()
        vals_sub = torch.from_numpy(baseline_sub_np).to(device, non_blocking=True)
        pflags_sub = torch.from_numpy(pflags_sub_np).to(device, non_blocking=True)
        padmask = (tokens_sub == vocab[PAD_TOKEN])

        with torch.inference_mode():
            pred_sub_t = forward_microbatched(unwrapped, tokens_sub, vals_sub, pflags_sub, padmask, MICRO_BSZ)
        pred_np = pred_sub_t.detach().float().cpu().numpy().astype(np.float32, copy=False)  # shape (#tuples, S)

        # Build pid -> pred vector map for this sample
        pid_to_pred = {int(pid): pred_np[j] for j, pid in enumerate(pids_np.tolist())}

        # Now accumulate Frobenius pieces over CELLS in this sample
        sum_sq_diff = 0.0
        sum_sq_ycent = 0.0

        # Precompute centered baseline in log space if needed
        if SI_LOG1P:
            base_log = np.log1p(np.clip(base_vec, 0.0, None)).astype(np.float32, copy=False)
        else:
            base_log = base_vec

        # Stream cells in chunks to limit memory
        chunk = 1024
        for s0 in range(0, nb, chunk):
            s1 = min(nb, s0 + chunk)
            cpart = cell_idxs[s0:s1]

            # Actual expression for these cells on selected genes
            y_act = _dense_rows_for_indices(X, cpart, sample_ids_np)  # (chunk, S)

            # Map each cell -> its tuple pid, then get pred row (repeat per tuple)
            pids_cells = pert_cell_pid[cpart]
            # Filter out any unexpected -1
            if np.any(pids_cells < 0):
                keep = (pids_cells >= 0)
                cpart = cpart[keep]
                y_act = y_act[keep]
                pids_cells = pids_cells[keep]
                if cpart.size == 0:
                    continue

            # Build pred matrix for this chunk
            y_pred = np.stack([pid_to_pred[int(pid)] for pid in pids_cells.tolist()], axis=0).astype(np.float32, copy=False)

            if SI_LOG1P:
                y_act_t = np.log1p(np.clip(y_act, 0.0, None)).astype(np.float32, copy=False)
                y_pred_t = np.log1p(np.clip(y_pred, 0.0, None)).astype(np.float32, copy=False)
            else:
                y_act_t = y_act
                y_pred_t = y_pred

            # Center by batch control (broadcast)
            y_act_c = y_act_t - base_log[None, :]
            y_pred_c = y_pred_t - base_log[None, :]

            d = (y_pred_c - y_act_c)
            sum_sq_diff += float(np.sum(d * d, dtype=np.float64))
            sum_sq_ycent += float(np.sum(y_act_c * y_act_c, dtype=np.float64))

        # Frobenius norms
        sd_b = (np.sqrt(sum_sq_diff) / float(nb)) if nb > 0 else float("nan")
        sdmax_b = (2.0 * np.sqrt(sum_sq_ycent) / float(nb)) if nb > 0 else float("nan")

        if np.isfinite(sd_b) and np.isfinite(sdmax_b) and sdmax_b > 1e-12:
            sum_sd += float(sd_b)
            sum_sdmax += float(sdmax_b)
            B_used += 1

        if is_main() and (bi % 25 == 0):
            rlog(f"[SI-CELL] sample {bi+1}/{len(samples)} sid={sid} nb={nb} S={S} partial_B={B_used} | RSS={_rss_gb():.2f} GB")

    if B_used <= 0:
        return float("nan"), dict(B_used=0)

    SD = sum_sd / float(B_used)
    SDmax = sum_sdmax / float(B_used)
    SI = 1.0 - (SD / SDmax) if SDmax > 1e-12 else float("nan")
    return float(SI), dict(B_used=int(B_used), SD=float(SD), SDmax=float(SDmax), SI_LOG1P=int(SI_LOG1P))


def run_eval(tag: str, h5_path: str):
    unwrapped = model.module if isinstance(model, DDP) else model
    unwrapped.eval()

    prep = prepare_adata(h5_path)

    G = prep["G"]
    gids_np = prep["gids_np"]

    pert_keys = prep["pert_keys"]
    pert_samples = prep["pert_samples"]
    pert_n_cells = prep["pert_n_cells"]
    pert_mean_expr = prep["pert_mean_expr"]
    pert_target_idx = prep["pert_target_idx"]

    ctrl_means = prep["ctrl_means"]
    ctrl_counts = prep["ctrl_counts"]
    global_mean = prep["global_mean"]
    baseline_cache = prep["baseline_cache"]

    eval_pert_ids = shard_indices(prep["eval_pert_ids"])

    rlog(
        f"{tag}: eval tuples={eval_pert_ids.size:,} (SUBSAMPLE={SUBSAMPLE}) | "
        f"ctrl_cells={prep['ctrl_idx_size']:,} pert_cells={prep['pert_idx_size']:,} | RSS={_rss_gb():.2f} GB"
    )

    with timed("build DataLoader", enabled=True):
        ds = IndexDataset(eval_pert_ids)
        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=EVAL_BSZ,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=False,
        )

    gids_1d = torch.as_tensor(gids_np, dtype=torch.long, device=device)

    per_pert = defaultdict(lambda: {
        "n_cells_in_tuple": 0,
        "mae": Agg(),
        "mse": Agg(),
        "rmse": Agg(),
        "pearson": Agg(),
        "spearman": Agg(),
        "cos_logfc": Agg(),
        "sign_agree": Agg(),
        "sd": 0.0,
        "sdmax": 0.0,
    })

    eps = 1e-8

    global_perts = 0
    # Running (online) summaries across evaluated perturbation-tuples
    # Weighted by number of control cells in the tuple's sample (n_controls_for_sample).
    run_wsum = {k: 0.0 for k in ["MAE","MSE","RMSE","Pearson","Spearman","CosineLogFC","StructuralIntegrity","StructuralSignAgreement"]}
    run_wtot = 0.0
    run_n_total = 0
    run_n_wpos = 0
    t0 = time.time()

    with torch.inference_mode():
        for bi, batch_pids in enumerate(dl):
            prof = bool(PROFILE_FIRST_BATCH and (bi == 0))
            pids = np.asarray(batch_pids.tolist(), dtype=np.int64)
            B = int(pids.size)

            if prof:
                rlog(f"==== FIRST BATCH PROFILING START ({tag}) B={B} G={G} MAX_SEQ_LEN={MAX_SEQ_LEN} MICRO_BSZ={MICRO_BSZ} ====")
                if torch.cuda.is_available():
                    rlog(f"cuda mem alloc={torch.cuda.memory_allocated()/1e9:.3f} GB | reserved={torch.cuda.memory_reserved()/1e9:.3f} GB")

            with timed("batch0: stack tuple mean expr into y_full_np (CPU)", enabled=prof):
                y_full_np = np.zeros((B, G), dtype=np.float32)
                for j, pid in enumerate(pids.tolist()):
                    y_full_np[j, :] = pert_mean_expr[pid]

            with timed("batch0: CPU→GPU y_full", enabled=prof):
                y_full_t = torch.from_numpy(y_full_np).to(device, non_blocking=True)

            with timed("batch0: choose_sample_ids", enabled=prof):
                target_idx_t = torch.from_numpy(
                    pert_target_idx[pids].astype(np.int64, copy=False)
                ).to(device, non_blocking=True)
                sample_ids = choose_sample_ids(y_full_t, target_idx_t, MAX_SEQ_LEN)

            with timed("batch0: sample_ids → CPU", enabled=prof):
                sample_ids_np = sample_ids.detach().cpu().numpy().astype(np.int64, copy=False)
                S = int(sample_ids_np.size)

            with timed("batch0: build baseline_sub + pflags_sub (CPU)", enabled=prof):
                baseline_sub_np, pflags_sub_np = build_baseline_sub_and_pflags_sub_for_perts(
                    pids, sample_ids_np,
                    pert_samples,
                    pert_target_idx,
                    ctrl_means,
                    ctrl_counts,
                    global_mean,
                    baseline_cache,
                    MIN_CTRL_PER_SAMPLE,
                )

            with timed("batch0: build tokens_sub/vals_sub/pflags_sub + padmask (GPU)", enabled=prof):
                tokens_sub = gids_1d[sample_ids].unsqueeze(0).expand(B, -1).contiguous()
                vals_sub = torch.from_numpy(baseline_sub_np).to(device, non_blocking=True)
                pflags_sub = torch.from_numpy(pflags_sub_np).to(device, non_blocking=True)
                padmask = (tokens_sub == vocab[PAD_TOKEN])

            start_evt, end_evt, use_evt = cuda_event_timing(bool(prof and PROFILE_CUDA_EVENTS))
            if use_evt:
                start_evt.record()

            with timed("batch0: forward_microbatched", enabled=prof, sync_cuda=False):
                pred_sub_t = forward_microbatched(
                    unwrapped,
                    tokens_sub,
                    vals_sub,
                    pflags_sub,
                    padmask,
                    MICRO_BSZ
                )

            if use_evt:
                end_evt.record()
                torch.cuda.synchronize()
                ms = start_evt.elapsed_time(end_evt)
                rlog(f"[CUDA-EVENT] forward_microbatched: {ms:.3f} ms (first batch)")

            with timed("batch0: gather y_sub for metrics (GPU)", enabled=prof):
                y_sub_t = y_full_t[:, sample_ids]

            with timed("batch0: compute metrics (GPU)", enabled=prof):
                diff = pred_sub_t - y_sub_t
                mae_t = diff.abs().mean(dim=1)
                mse_t = (diff * diff).mean(dim=1)
                rmse_t = torch.sqrt(mse_t.clamp_min(0.0))

                pearson_t = pearson_exact_rowwise_torch(pred_sub_t, y_sub_t)

                if COMPUTE_SPEARMAN:
                    pred_np2 = pred_sub_t.detach().float().cpu().numpy()
                    y_np2 = y_sub_t.detach().float().cpu().numpy()
                    spearman_np = spearman_exact_batch(pred_np2, y_np2)
                else:
                    spearman_np = np.full((B,), np.nan, dtype=np.float32)

                # CosineLogFC (PerturBench-style): cosine between logFC vectors relative to baseline
                base_t = vals_sub
                logfc_pred = torch.log1p(pred_sub_t.clamp_min(0.0)) - torch.log1p(base_t.clamp_min(0.0))
                logfc_true = torch.log1p(y_sub_t.clamp_min(0.0))   - torch.log1p(base_t.clamp_min(0.0))

                num = (logfc_pred * logfc_true).sum(dim=1)
                den = torch.sqrt((logfc_pred * logfc_pred).sum(dim=1) * (logfc_true * logfc_true).sum(dim=1)).clamp_min(eps)
                cos_t = num / den

                # Old per-tuple "SI pieces" kept unchanged (we will ADD the new SI separately)
                sd_t = (diff * diff).sum(dim=1)
                sdmax_t = (y_sub_t * y_sub_t).sum(dim=1)

                # StructuralSignAgreement: sign agreement on logFC
                sign_pred = torch.sign(logfc_pred)
                sign_true = torch.sign(logfc_true)
                signagree_t = (sign_pred == sign_true).float().mean(dim=1)

            with timed("batch0: move metrics to CPU", enabled=prof):
                mae_np = mae_t.detach().float().cpu().numpy()
                mse_np = mse_t.detach().float().cpu().numpy()
                rmse_np = rmse_t.detach().float().cpu().numpy()
                pearson_np = pearson_t.detach().float().cpu().numpy()
                cos_np = cos_t.detach().float().cpu().numpy()
                sd_np = sd_t.detach().float().cpu().numpy()
                sdmax_np = sdmax_t.detach().float().cpu().numpy()
                signagree_np = signagree_t.detach().float().cpu().numpy()

            with timed("batch0: aggregate per tuple (CPU)", enabled=prof):
                for j, pid in enumerate(pids.tolist()):
                    k = pert_keys[pid]
                    a = per_pert[k]
                    a["n_cells_in_tuple"] = int(pert_n_cells[pid])

                    a["mae"].add_sum_count(float(mae_np[j]), 1)
                    a["mse"].add_sum_count(float(mse_np[j]), 1)
                    a["rmse"].add_sum_count(float(rmse_np[j]), 1)
                    a["pearson"].add_sum_count(float(pearson_np[j]), 1)
                    if COMPUTE_SPEARMAN:
                        a["spearman"].add_sum_count(float(spearman_np[j]), 1)
                    a["cos_logfc"].add_sum_count(float(cos_np[j]), 1)
                    a["sign_agree"].add_sum_count(float(signagree_np[j]), 1)

                    a["sd"] += float(sd_np[j])
                    a["sdmax"] += float(sdmax_np[j])

                    # Online weighted summary (by n_controls_for_sample = ctrl_counts[sample])
                    sid = str(pert_samples[pid])
                    w = float(ctrl_counts.get(sid, 0))
                    run_n_total += 1
                    if w > 0:
                        run_n_wpos += 1
                        run_wsum["MAE"] += float(mae_np[j]) * w
                        run_wsum["MSE"] += float(mse_np[j]) * w
                        run_wsum["RMSE"] += float(rmse_np[j]) * w
                        if np.isfinite(pearson_np[j]): run_wsum["Pearson"] += float(pearson_np[j]) * w
                        if COMPUTE_SPEARMAN and np.isfinite(spearman_np[j]): run_wsum["Spearman"] += float(spearman_np[j]) * w
                        if np.isfinite(cos_np[j]): run_wsum["CosineLogFC"] += float(cos_np[j]) * w
                        if np.isfinite(sd_np[j]) and np.isfinite(sdmax_np[j]):
                            si_old = (1.0 - float(sd_np[j]) / float(sdmax_np[j])) if float(sdmax_np[j]) > 1e-8 else float("nan")
                            if np.isfinite(si_old): run_wsum["StructuralIntegrity"] += float(si_old) * w
                        if np.isfinite(signagree_np[j]): run_wsum["StructuralSignAgreement"] += float(signagree_np[j]) * w
                        run_wtot += w

                global_perts += B

            if (bi % LOG_EVERY) == 0:
                elapsed = time.time() - t0
                rate = global_perts / elapsed if elapsed > 0 else float("nan")
                remaining = (eval_pert_ids.size - global_perts)
                eta = remaining / rate if rate > 0 else float("nan")

                if run_wtot > 0:
                    w_mae = run_wsum["MAE"] / run_wtot
                    w_mse = run_wsum["MSE"] / run_wtot
                    w_rmse = run_wsum["RMSE"] / run_wtot
                    w_p = run_wsum["Pearson"] / run_wtot
                    w_s = run_wsum["Spearman"] / run_wtot
                    w_cos = run_wsum["CosineLogFC"] / run_wtot
                    w_si = run_wsum["StructuralIntegrity"] / run_wtot
                    w_sa = run_wsum["StructuralSignAgreement"] / run_wtot
                else:
                    w_mae = w_mse = w_rmse = w_p = w_s = w_cos = w_si = w_sa = float("nan")

                rlog(
                    f"{tag} | batch {bi} | "
                    f"tuples {global_perts:,}/{eval_pert_ids.size:,} | "
                    f"rate {rate:.2f} tuples/s | ETA {eta/60.0:.1f} min | "
                    f"B={EVAL_BSZ} micro={MICRO_BSZ} S={S} | "
                    f"wctrl(MAE={w_mae:.4f} MSE={w_mse:.4f} RMSE={w_rmse:.4f} "
                    f"P={w_p:.4f} S={w_s:.4f} CosLFC={w_cos:.4f} SI(old)={w_si:.4f} SA={w_sa:.4f} "
                    f"n={run_n_total} n_wpos={run_n_wpos} wtot={run_wtot:.0f}) | "
                    f"RSS={_rss_gb():.2f} GB"
                )

            if prof:
                rlog(f"==== FIRST BATCH PROFILING END ({tag}) ====")

    # Build final per-pert output dict (one row per perturbation tuple)
    per_pert_output: Dict[str, Dict] = {}
    for k, a in per_pert.items():
        delim = "\x1f"
        parts = str(k).split(delim)
        if len(parts) == 3:
            sid, guide_t, gene_t = parts
        else:
            sid, guide_t, gene_t = "", "", ""

        sd = float(a["sd"])
        sdmax = float(a["sdmax"])
        si_old = (1.0 - sd / sdmax) if sdmax > 1e-8 else float("nan")

        n_ctrl = int(ctrl_counts.get(str(sid), 0))

        per_pert_output[k] = {
            "sample": str(sid),
            "guide_target": str(guide_t),
            "gene_target": str(gene_t),

            "n_cells_in_tuple": int(a["n_cells_in_tuple"]),
            "n_controls_for_sample": int(n_ctrl),

            "MAE": a["mae"].mean(),
            "MSE": a["mse"].mean(),
            "RMSE": a["rmse"].mean(),
            "Pearson": a["pearson"].mean(),
            "Spearman": a["spearman"].mean() if COMPUTE_SPEARMAN else float("nan"),
            "CosineLogFC": a["cos_logfc"].mean(),
            "StructuralIntegrity": float(si_old),  # kept for backward compat
            "StructuralSignAgreement": a["sign_agree"].mean(),
        }

    if is_main():
        out_json = SAVE_DIR / f"{tag}_per_perturbation_metrics.json"
        with open(out_json, "w") as f:
            json.dump(per_pert_output, f, indent=2)
        rlog(f"Wrote per-perturbation metrics: {out_json}")

        # Existing summaries (unchanged): unweighted + weighted_by_controls
        avg_unw = avg_across_perturbations_unweighted(per_pert_output)
        avg_w = avg_across_perturbations_weighted_by_controls(per_pert_output)

        # NEW: compute cellwise/batchwise SI exactly as requested and store it
        si_cell, si_meta = compute_structural_integrity_cellwise_batchwise(
            tag=tag,
            unwrapped=unwrapped,
            prep=prep,
        )
        rlog(f"[SI-CELL] StructuralIntegrity_cellwise_batchwise = {si_cell:.6f} | meta={si_meta}")

        out_sum = SAVE_DIR / f"{tag}_summary.json"
        with open(out_sum, "w") as f:
            json.dump(
                {
                    "unweighted": avg_unw,
                    "weighted_by_controls": avg_w,
                    "structural_integrity_cellwise_batchwise": {
                        "SI": float(si_cell),
                        **si_meta,
                    },
                },
                f,
                indent=2,
            )
        rlog(f"Wrote summary metrics: {out_sum}")
        rlog(f"[SUMMARY] unweighted: {avg_unw}")
        rlog(f"[SUMMARY] weighted_by_controls: {avg_w}")


def main():
    if H5AD_A:
        run_eval("A", H5AD_A)
    if H5AD_B:
        run_eval("B", H5AD_B)

    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()

