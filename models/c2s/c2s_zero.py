#!/usr/bin/env python
import os
import argparse
import json
from collections import defaultdict

import numpy as np
import anndata
from scipy import sparse as sp
from scipy.stats import pearsonr, spearmanr
import torch
import torch.distributed as dist
from tqdm import tqdm

import cell2sentence as cs
from cell2sentence import utils as cs_utils
from transformers import AutoModelForCausalLM

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def merge_pert_results(dst, src):
    for gene, d in src.items():
        if gene not in dst:
            dst[gene] = {
                "pred_sum": d["pred_sum"].copy(),
                "actual_sum": d["actual_sum"].copy(),
                "ctrl_sum": d["ctrl_sum"].copy(),
                "count": int(d["count"]),
            }
        else:
            dst[gene]["pred_sum"] += d["pred_sum"]
            dst[gene]["actual_sum"] += d["actual_sum"]
            dst[gene]["ctrl_sum"] += d["ctrl_sum"]
            dst[gene]["count"] += int(d["count"])

def merge_struct_acc(dst, src):
    for batch_name, d in src.items():
        if batch_name not in dst:
            dst[batch_name] = {
                "sum_sq_diff": float(d["sum_sq_diff"]),
                "sum_sq_act": float(d["sum_sq_act"]),
                "n_cells": int(d["n_cells"]),
            }
        else:
            dst[batch_name]["sum_sq_diff"] += float(d["sum_sq_diff"])
            dst[batch_name]["sum_sq_act"]  += float(d["sum_sq_act"])
            dst[batch_name]["n_cells"]     += int(d["n_cells"])


def make_pert_entry(n_genes: int):
    return {
        "pred_sum": np.zeros(n_genes, dtype=np.float64),
        "actual_sum": np.zeros(n_genes, dtype=np.float64),
        "ctrl_sum": np.zeros(n_genes, dtype=np.float64),
        "count": 0,
    }

def init_pert_results(n_genes: int):
    # plain dict, checkpoint-safe
    return {}


def make_struct_entry():
    return {"sum_sq_diff": 0.0, "sum_sq_act": 0.0, "n_cells": 0}

def init_struct_acc():
    return {}


def torch_save(obj, path: str):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

def save_checkpoint(path: str, cursor: int, pert_results, struct_acc, rank=None, world_size=None, shard_len=None):
    if path is None:
        return
    state = {
        "cursor": int(cursor),
        "rank": int(rank) if rank is not None else None,
        "world_size": int(world_size) if world_size is not None else None,
        "shard_len": int(shard_len) if shard_len is not None else None,
        "pert_results": pert_results,
        "struct_acc": struct_acc,
    }

    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)   # atomic on POSIX


def load_checkpoint(path: str, expected_rank=None, expected_world_size=None, expected_shard_len=None):
    if path is None or not os.path.exists(path):
        return None
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)

        if not isinstance(state, dict):
            print(f"[ckpt] Invalid checkpoint (not a dict): {path}", flush=True)
            return None
        for k in ("cursor", "pert_results", "struct_acc"):
            if k not in state:
                print(f"[ckpt] Invalid checkpoint (missing {k}): {path}", flush=True)
                return None

        if expected_rank is not None and state.get("rank") is not None and int(state["rank"]) != int(expected_rank):
            print(f"[ckpt] Rank mismatch: ckpt_rank={state.get('rank')} expected={expected_rank}. Ignoring {path}", flush=True)
            return None
        if expected_world_size is not None and state.get("world_size") is not None and int(state["world_size"]) != int(expected_world_size):
            print(f"[ckpt] World size mismatch: ckpt_ws={state.get('world_size')} expected={expected_world_size}. Ignoring {path}", flush=True)
            return None
        if expected_shard_len is not None and state.get("shard_len") is not None and int(state["shard_len"]) != int(expected_shard_len):
            print(f"[ckpt] Shard len mismatch: ckpt_len={state.get('shard_len')} expected={expected_shard_len}. Ignoring {path}", flush=True)
            return None

        return state
    except Exception as e:
        print(f"[ckpt] Failed to load checkpoint {path}: {e}", flush=True)
        return None

    
#multi GPU data parallelism
def get_dist_info():
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    return rank, world_size, local_rank

#Get gene expression for the particular cell
def get_expression_row(adata, idx):
    x = adata.X[idx]
    if sp.issparse(x):
        x = x.toarray()
    return np.asarray(x).ravel().astype(np.float32)


def structural_from_acc(struct_acc):
    batches = [v for v in struct_acc.values() if v["n_cells"] > 0]
    if not batches:
        return {"structural_integrity": float("nan"), "struct_distance": 0.0, "struct_distance_max": 0.0}
    num_batches = len(batches)
    struct_distance = 0.0
    struct_distance_max = 0.0

    for v in batches:
        n = v["n_cells"]
        frob_diff = float(np.sqrt(v["sum_sq_diff"]))
        frob_act  = float(np.sqrt(v["sum_sq_act"]))
        struct_distance     += (1.0 / num_batches) * (1.0 / n) * frob_diff
        struct_distance_max += (1.0 / num_batches) * (1.0 / n) * frob_act

    structural_integrity = 1.0 - (struct_distance / struct_distance_max) if struct_distance_max > 0 else float("nan")

    return {
        "structural_integrity": structural_integrity,
        "struct_distance": struct_distance,
        "struct_distance_max": struct_distance_max,
    }

def evaluate_correlations(n_genes, pert_results):
    pert_pearson_list = []
    pert_spearman_list = []
    pert_cosine_list = []
    pert_mse_list = []

    epsilon = 1e-8

    logfc_preds = np.zeros(n_genes, dtype=np.float64)
    logfc_actuals = np.zeros(n_genes, dtype=np.float64)

    #compute correlations
    for pert_gene, data in tqdm(pert_results.items(), desc = "[eval] computing correlations"):
        if data['count'] == 0:
            continue

        #compute mean
        mean_pred = data['pred_sum'] / data['count']
        mean_actual = data['actual_sum'] / data['count']
        mean_ctrl = data['ctrl_sum'] / data['count']

        #Log fold changes
        np.log((mean_pred + epsilon) / (mean_ctrl + epsilon), out = logfc_preds)
        np.log((mean_actual + epsilon) / (mean_ctrl + epsilon), out = logfc_actuals)

        #Pearson
        try:
            pr = pearsonr(logfc_preds, logfc_actuals)[0]
            if np.isfinite(pr):
                pert_pearson_list.append(pr)
        except:
            pass

    
        #Spearman
        try:
            sr = spearmanr(logfc_preds, logfc_actuals)[0]
            if np.isfinite(sr):
                pert_spearman_list.append(sr)
        except:
            pass

        cos_sim = np.dot(logfc_preds, logfc_actuals) / (np.linalg.norm(logfc_preds) * np.linalg.norm(logfc_actuals) + epsilon)
        if np.isfinite(cos_sim):
            pert_cosine_list.append(cos_sim)

        #MSE
        mse_logfc = np.mean((logfc_preds - logfc_actuals) ** 2)
        if np.isfinite(mse_logfc):
            pert_mse_list.append(mse_logfc)

    
    pert_pearson_mean = float(np.mean(pert_pearson_list)) if pert_pearson_list else float('nan')
    pert_spearman_mean = float(np.mean(pert_spearman_list)) if pert_spearman_list else float('nan')
    pert_cosine_mean = float(np.mean(pert_cosine_list)) if pert_cosine_list else float('nan')
    pert_mse_mean = float(np.mean(pert_mse_list)) if pert_mse_list else float('nan')

    print(f"\n[eval] Results:")
    print(f"  Pearson (LogFC):  {pert_pearson_mean:.4f}")
    print(f"  Spearman (LogFC): {pert_spearman_mean:.4f}")
    print(f"  Cosine (LogFC):   {pert_cosine_mean:.4f}")
    print(f"  MSE (LogFC):      {pert_mse_mean:.4f}")

    if pert_pearson_mean < 0.1:
        print(" WARNING: Very low per-perturbation correlation - model may not be learning perturbation effects")
    
    if len([p for p in pert_pearson_list if p < 0]) > len(pert_pearson_list) * 0.1:
        print("WARNING: >10% of perturbations have negative correlation - check model predictions")

    print(f"\n[eval] Found {len(pert_results)} unique perturbations")

    return pert_mse_mean, pert_pearson_mean, pert_spearman_mean, pert_cosine_mean

#fit gene ranks to gene exprssion values, both in log space
def fit_logrank_linear_model(adata, max_cells=128, random_state=1234):
    #get random number generator
    r_generator = np.random.default_rng(random_state)

    n = adata.n_obs
    
    print('adata.n_obs', adata.n_obs)

    if max_cells > 0 and n > max_cells:
        idxs = r_generator.choice(n, size=max_cells, replace=False)
    else:
        idxs = np.arange(n)

    print('idxs size', len(idxs))
    print('idxs first five', idxs[0:5])

    epsilon = 1e-8
    all_x = []
    all_y = []

    for idx in tqdm(idxs, desc="[rank-fit]"):
        row = get_expression_row(adata, idx)

        #boolean array of non zero indices
        non_zero_idxs = row > 0
        if not np.any(non_zero_idxs):
            continue

        #expr will only contain nonzero gene expressions
        expr = row[non_zero_idxs]

        order = np.argsort(-expr)

        print('len(order)', len(order))
        print('order', order)
        #make expressions in descending order. More expressed at the top
        expr = expr[order]

        ranks = np.arange(1, len(expr) + 1)

        print('len ranks', len(ranks))
        print('ranks', ranks)
        all_x.append(np.log(ranks))
        all_y.append(np.log(expr + epsilon))


    print('all_x', all_x)
    print('all_y', all_y)

    X = np.concatenate(all_x)
    Y = np.concatenate(all_y)

    print('X', X)
    print('Y', Y)

    slope, intercept = np.polyfit(X, Y, 1)
    return float(slope), float(intercept)

def streamed_generate(
    perturbed_idxs, batch_size, sample_field, baseline_by_batch, args,
    vocab_list, adata, csmodel, hf_backend, slope, intercept, n_genes,
    write_jsonl_path=None,
    rank=0, world_size=1,
    ckpt_path=None, ckpt_batch_stride=200, resume=True,
    prompt_perturbation: str | None = None,
    only_first_batch_debug: bool = True
):

    epsilon = 1e-8


    #resume from checkpoint
    cursor = 0

    pert_results = init_pert_results(n_genes)
    struct_acc = init_struct_acc()
    log_base_cache = {}

    vocab_set = set(vocab_list)

    ctrl_sentence_cache = {}
    gene_order_cache = {}

    if resume and ckpt_path is not None:
        state = load_checkpoint(ckpt_path, expected_rank=rank, expected_world_size=world_size, expected_shard_len=len(perturbed_idxs))


        if state is not None:
            cursor = int(state.get("cursor", 0))

            pert_results = state["pert_results"]
            struct_acc = state["struct_acc"]
            print(f"Rank {rank}, Resuming from cursor={cursor} (shard_len={len(perturbed_idxs)})")
            print(f"[Resume] rank={rank} cursor={cursor} shard_len={len(perturbed_idxs)} ckpt={ckpt_path}", flush=True)


    fout = None
    if write_jsonl_path is not None:
        import gzip
        fout = gzip.open(write_jsonl_path, "wt") if write_jsonl_path.endswith(".gz") else open(write_jsonl_path, "w")

    next_ckpt_at = cursor + ckpt_batch_stride * batch_size

    for b in tqdm(range(cursor, len(perturbed_idxs), batch_size), desc="[inference] minibatches"):
        batch_cells = perturbed_idxs[b:b + batch_size]

        X_expression_block = adata.X[batch_cells]  # (B, G)
        if sp.issparse(X_expression_block):
            X_expression_block = X_expression_block.toarray()
        X_expression_block = np.asarray(X_expression_block, dtype=np.float32)

        batch_prompts = []
        batch_meta = []

        for idx in batch_cells:
            batch_name = str(adata.obs.iloc[idx][sample_field])

            baseline = baseline_by_batch[batch_name]


            #cache gene order
            if batch_name not in gene_order_cache:
                gene_order_cache[batch_name] = np.argsort(-baseline)[:args.top_k_genes]

            #cache control sentence
            if batch_name not in ctrl_sentence_cache:
                ctrl_sentence_cache[batch_name] = " ".join(
                    vocab_list[g] for g in gene_order_cache[batch_name]
                )

            ctrl_sentence = ctrl_sentence_cache[batch_name]

            true_pert_gene = str(adata.obs.iloc[idx]["gene_target"])

            # The gene token we *write into the prompt* (real vs null ablation)
            prompt_gene = prompt_perturbation if prompt_perturbation is not None else true_pert_gene

            prompt = (
                f"Given the following cell sentence consisting of {args.top_k_genes} control genes, "
                f"predict the cell sentence after applying the perturbation {prompt_gene}.\n"
                f"Control cell sentence: {ctrl_sentence}.\n\nPerturbed cell sentence:"
            )

            batch_prompts.append(prompt)
            # Store BOTH: the true gene (for grouping/metrics) and the prompt gene (for debugging)
            batch_meta.append((idx, batch_name, true_pert_gene, prompt_gene, ctrl_sentence))

        with torch.inference_mode():
            preds = csmodel.generate_from_prompt_batched(
                model=hf_backend,
                prompt_list=batch_prompts,
                max_num_tokens=args.max_new_tokens,    
                do_sample=False
        )

        if b == 0:
            print("\n[Validation] First batch predictions:")
            print(f"  Batch size: {len(batch_prompts)}")
            print(f"  Sample prompt (first 300 chars):")
            print(f"    {batch_prompts[0][:300]}...")

        if rank == 0 and b == 0 and only_first_batch_debug:
            print("\n  Sample prediction (raw):")
            print(f"    {str(preds[0])[:400]}...")

        # consume predictions immediately
        for i, (pred_sentence, meta) in enumerate(zip(preds, batch_meta)):
            # drop common prefixes like "Cell 1:" or "Cell 2:"
            pred_sentence = pred_sentence.split(":", 1)[1].strip() if pred_sentence.lower().startswith("cell") and ":" in pred_sentence else pred_sentence

            idx, batch_name, true_pert_gene, prompt_gene, ctrl_sentence = meta


            y_ctrl = baseline_by_batch[batch_name]
            y_actual = X_expression_block[i]

            proc, _ = cs_utils.post_process_generated_cell_sentences(
                str(pred_sentence),
                vocab_list,
                "NA_GENE",
            )

            y_pred = cs_utils.reconstruct_expression_from_cell_sentence(
                cell_sentence_str=" ".join(proc),
                delimiter=" ",
                vocab_list=vocab_list,
                slope=slope,
                intercept=intercept,
            )

            # ---- per-perturbation accumulation ----
            # Group by TRUE perturbation gene for paired comparison (real vs null)
            pr = pert_results.setdefault(true_pert_gene, make_pert_entry(n_genes))

            pr["pred_sum"] += y_pred
            pr["actual_sum"] += y_actual
            pr["ctrl_sum"] += y_ctrl
            pr["count"] += 1

            # ---- structural accumulation (streaming Frobenius) ----
            # logFC relative to chosen baseline for this batch

        
            if batch_name not in log_base_cache:
                log_base_cache[batch_name] = np.log(baseline_by_batch[batch_name] + epsilon)
            log_base = log_base_cache[batch_name]

            v_pred = np.log(y_pred + epsilon) - log_base
            v_act  = np.log(y_actual + epsilon) - log_base
            d = v_pred - v_act

            sa = struct_acc.setdefault(batch_name, make_struct_entry())
            sa["sum_sq_diff"] += float(np.dot(d, d))
            sa["sum_sq_act"]  += float(np.dot(v_act, v_act))
            sa["n_cells"] += 1
            
            # ---- optional compact logging ----
            if fout is not None:
                rec = {
                    "idx": int(idx),
                    "sample": batch_name,
                    "pert_gene": true_pert_gene,
                    # WARNING: writing raw pred is huge; only do if you really need it
                    # "pred": pred_sentence,
                }
                fout.write(json.dumps(rec) + "\n")

        # ---- advance cursor ----
        cursor = b + len(batch_cells)

        # ---- checkpoint periodically ----
        if ckpt_path is not None and cursor >= next_ckpt_at:
            save_checkpoint(
                ckpt_path,
                cursor=cursor,  # use your computed cursor, not b+batch_size
                pert_results=pert_results,
                struct_acc=struct_acc,
                rank=rank,
                world_size=world_size,
                shard_len=len(perturbed_idxs),
            )
            next_ckpt_at += ckpt_batch_stride * batch_size

    if fout is not None:
        fout.close()

    return pert_results, struct_acc, len(ctrl_sentence_cache), len(gene_order_cache)


def zero_shot_inference(h5ad_path, args, hf_model, csmodel, rank, world_size, device):

    EVAL_SUBSET_TOTAL = 50         # total target across all ranks
    NULL_TOKEN = "CTRL_NULL"          # perturb token used in null run
    SUBSET_SEED = args.seed           # keep deterministic

    #Load in backed mode
    adata = anndata.read_h5ad(h5ad_path, backed = 'r')

    if rank == 0:
        print("Rank is 0")

    if "pass_guide_filter" in adata.obs.columns:
        pass_mask = adata.obs["pass_guide_filter"].astype(bool)

    else:
        pass_mask = np.ones(adata.n_obs, dtype=bool)    

    #Get control cells
    control_set = {g.strip() for g in args.control_genes.split(",")}
    gt = adata.obs["gene_target"].astype(str)

    is_control = (gt == "Non-Targeting") | gt.isin(control_set)

    ctrl_idxs = np.where(is_control & pass_mask)[0]
    perturbed_idxs = np.where((~is_control) & pass_mask)[0]


    # --------------------------------------------------
    # LIMIT ZERO-SHOT EVAL TO FIXED GLOBAL SUBSET
    # --------------------------------------------------
    if EVAL_SUBSET_TOTAL is not None:
        rng = np.random.default_rng(SUBSET_SEED)

        if len(perturbed_idxs) > EVAL_SUBSET_TOTAL:
            perturbed_idxs = rng.choice(
                perturbed_idxs,
                size=EVAL_SUBSET_TOTAL,
                replace=False,
            )

        perturbed_idxs = np.sort(perturbed_idxs)  # for deterministic sharding


    print("CTRL/SAFE/NTC present counts:\n", gt[gt.isin(control_set)].value_counts())
    print("Total mapped from control_set:", int(gt.isin(control_set).sum()))

    print("Raw Non-Targeting:", int((gt == "Non-Targeting").sum()))
    print("Control_set matches:", int(gt.isin(control_set).sum()))
    print("Total control by your rule:", int(((gt == "Non-Targeting") | gt.isin(control_set)).sum()))
          
    vocab_list = list(adata.var_names)
    perturbed_idxs = perturbed_idxs.tolist()
    perturbed_idxs = perturbed_idxs[rank::world_size]

    print(f"[Rank {rank}] Shard size: {len(perturbed_idxs)}")

    baseline_by_batch = {}
    baseline_control_counts = {}

    sample_field = "sample"

    for batch_name in adata.obs.loc[pass_mask, sample_field].unique():
        idxs = np.where(is_control & (adata.obs[sample_field].to_numpy() == batch_name))[0]

        if len(idxs) == 0:
            continue

        Xm = adata.X[idxs]

        if sp.issparse(Xm):
            baseline_by_batch[batch_name] = np.asarray(Xm.mean(axis=0)).ravel().astype(np.float32)
        else:
            baseline_by_batch[batch_name] = Xm.mean(axis=0).astype(np.float32)

        baseline_control_counts[batch_name] = len(idxs)


    print("[Sanity] baseline_by_batch entries:", len(baseline_by_batch))
    some_key = next(iter(baseline_by_batch))
    print("[Sanity] example sample key:", some_key, "control_count:", baseline_control_counts[some_key])

    hf_backend = hf_model
    hf_backend.eval()

    #generate predictions
    #slope, intercept = fit_logrank_linear_model(adata, max_cells=args.rank_fit_cells, random_state=args.seed)

    if rank == 0:
        slope, intercept = fit_logrank_linear_model(
            adata, max_cells=args.rank_fit_cells, random_state=args.seed
        )
    else:
        slope, intercept = 0.0, 0.0

    if dist.is_initialized():
        t = torch.tensor([slope, intercept], dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")
        dist.broadcast(t, src=0)
        slope, intercept = float(t[0].item()), float(t[1].item())
        dist.barrier()

    if rank == 0:
        print(f"[rank-fit] slope={slope:.6f} intercept={intercept:.6f}", flush=True)



    # stream inference + metrics accumulation
    if rank == 0:
        print("Start computing per-perturbation metrics")
        print(f"[Sanity] perturbed_idxs: {len(perturbed_idxs)} ctrl_idxs: {len(ctrl_idxs)}", flush=True)
        print("[Sanity] n_genes:", len(vocab_list), "top_k:", args.top_k_genes, "max_new_tokens:", args.max_new_tokens)


    # ---- checkpoint path ----
    dataset_tag = os.path.splitext(os.path.basename(h5ad_path))[0]
    ckpt_dir = os.path.join(args.save_dir, "ckpts", dataset_tag, f"rank{rank}")
    os.makedirs(ckpt_dir, exist_ok=True)

    n_genes=len(vocab_list)

    ckpt_path_real = os.path.join(ckpt_dir, f"ckpt_real.rank{rank}.pt")
    ckpt_path_null = os.path.join(ckpt_dir, f"ckpt_null.rank{rank}.pt")


    pert_results, struct_acc, n_ctrl_cached, n_orders_cached = streamed_generate(perturbed_idxs=perturbed_idxs, batch_size=args.batch_size,
        sample_field=sample_field, baseline_by_batch=baseline_by_batch, 
        args=args, vocab_list=vocab_list, adata=adata,
        csmodel=csmodel, hf_backend=hf_backend,
        slope=slope, intercept=intercept,
        n_genes=n_genes,
        write_jsonl_path=None,  # or args.save_dir + "/preds.jsonl.gz" if you truly want logs
        rank=rank,
        world_size=world_size,
        ckpt_path=ckpt_path_real,
        ckpt_batch_stride=args.ckpt_batch_stride,
        resume=args.resume,
    )

    # ------------------------
    # PASS 2: NULL perturbation token
    # ------------------------
    pert_null, struct_null, n_ctrl_cached_n, n_orders_cached_n = streamed_generate(
        perturbed_idxs=perturbed_idxs,
        batch_size=args.batch_size,
        sample_field=sample_field,
        baseline_by_batch=baseline_by_batch,
        args=args,
        vocab_list=vocab_list,
        adata=adata,
        csmodel=csmodel,
        hf_backend=hf_backend,
        slope=slope,
        intercept=intercept,
        n_genes=n_genes,
        write_jsonl_path=None,
        rank=rank,
        world_size=world_size,
        ckpt_path=ckpt_path_null,
        resume=args.resume,
        prompt_perturbation=NULL_TOKEN,       # <-- NULL
        only_first_batch_debug=False,         # don’t spam a second time
    )


    partial_dir = os.path.join(args.save_dir, "partials", dataset_tag)
    os.makedirs(partial_dir, exist_ok=True)
    partial_path = os.path.join(partial_dir, f"partial.rank{rank}.pt")

    partial_state = {
        "rank": rank,
        "world_size": world_size,
        "n_genes": len(vocab_list),
        "pert_results": pert_results,
        "struct_acc": struct_acc,
        "pert_null": pert_null,
        "struct_null": struct_null,
    }

    # atomic save
    tmp = partial_path + ".tmp"
    torch.save(partial_state, tmp)
    os.replace(tmp, partial_path)

    if dist.is_initialized():
        dist.barrier()
        
    # ========================================
    # Gather results from all ranks
    # ========================================
    if rank == 0:
        partial_dir = os.path.join(args.save_dir, "partials", dataset_tag)

        merged_pert = {}
        merged_null = {}
        merged_struct = {}
        merged_struct_null = {}

        for r in range(world_size):
            p = os.path.join(partial_dir, f"partial.rank{r}.pt")

            # PyTorch 2.6+ note: weights_only defaults True; for our own files, set False.
            state = torch.load(p, map_location="cpu", weights_only=False)

            merge_pert_results(merged_pert, state["pert_results"])
            merge_pert_results(merged_null, state["pert_null"])
            merge_struct_acc(merged_struct, state["struct_acc"])
            merge_struct_acc(merged_struct_null, state["struct_null"])

        # Now compute metrics on merged results (rank0 only)
        pert_mse_mean, pert_pearson_mean, pert_spearman_mean, pert_cosine_mean = evaluate_correlations(
            len(vocab_list), merged_pert
        )
        _, pearson_null, spearman_null, cosine_null = evaluate_correlations(
            len(vocab_list), merged_null
        )

        ss = structural_from_acc(merged_struct)

        print("\n" + "="*80)
        print("NULL ABLATION COMPARISON (ZERO-SHOT)")
        print("="*80)
        print(f"Pearson(LogFC):  real={pert_pearson_mean:.4f}  null={pearson_null:.4f}  Δ={pert_pearson_mean - pearson_null:.4f}")
        print(f"Spearman(LogFC): real={pert_spearman_mean:.4f}  null={spearman_null:.4f}  Δ={pert_spearman_mean - spearman_null:.4f}")
        print(f"Cosine(LogFC):   real={pert_cosine_mean:.4f}  null={cosine_null:.4f}  Δ={pert_cosine_mean - cosine_null:.4f}")
        print("="*80 + "\n")

        # Use merged_* for final summary / return
        pert_results = merged_pert
        pert_null = merged_null
        struct_acc = merged_struct
    else:
        # other ranks: do nothing
        pert_mse_mean = pert_pearson_mean = pert_spearman_mean = pert_cosine_mean = float("nan")


    if rank == 0:
        print(f"[Sanity] Cached control sentences: {n_ctrl_cached}")
        print(f"[Sanity] Cached gene orders: {n_orders_cached}")
        print("top_k_genes:", args.top_k_genes)
        print("max_new_tokens:", args.max_new_tokens)
        print("batch_size:", args.batch_size)
        print("world_size:", world_size)
        
        ss = structural_from_acc(struct_acc)

        # ==========================================================
        # Final Summary
        # ==========================================================
        print("\n" + "="*80)
        print("FINAL RESULTS SUMMARY")
        print("="*80)
        print(f"File: {h5ad_path}")
        
        total_eval = sum(v["n_cells"] for v in struct_acc.values())
        print(f"Cells evaluated: {total_eval}")
        print(f"Unique perturbations: {len(pert_results)}")

        print(f"\nPer-perturbation metrics (MAIN RESULTS):")
        print(f"  Pearson (LogFC):  {pert_pearson_mean:.4f}")
        print(f"  Spearman (LogFC): {pert_spearman_mean:.4f}")
        print(f"  Cosine (LogFC):   {pert_cosine_mean:.4f}")
        print(f"  MSE (LogFC):      {pert_mse_mean:.4f}")
        print(f"\nStructural metrics:")
        print(f"  Structural distance :  {ss['struct_distance']}")
        print(f"  Structural Integrity :  {ss['structural_integrity']}")
        print("="*80 + "\n")

        return {
            "file": h5ad_path,
            "n_cells": int(adata.n_obs),
            "n_perturbations": len(pert_results),
            
            # Per-perturbation metrics (main results)
            "Pearson_LogFC": pert_pearson_mean,
            "Spearman_LogFC": pert_spearman_mean,
            "Cosine_LogFC": pert_cosine_mean,
            "MSE_LogFC": pert_mse_mean,
            
            # Metadata
            "rank_to_expression_slope": slope,
            "rank_to_expression_intercept": intercept,
            "top_k_genes": args.top_k_genes,

            #structural integrity scores
            "structural_distance": ss['struct_distance'],
            "structural_integrity": ss['structural_integrity']
        }
    else:
        # Non-zero ranks return minimal info
        return {
            "file": h5ad_path,
            "rank": rank,
            "note": "Non-zero rank; see rank 0 for aggregated results"
        }


def parse_args():
    p = argparse.ArgumentParser()
    # p.add_argument("--h5ad_A", required=True)
    # p.add_argument("--h5ad_B", required=True)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--control_genes", default="CTRL,SAFE,NTC")
    p.add_argument("--model_name_or_path", default="vandijklab/C2S-Scale-Pythia-1b-pt")
    p.add_argument("--top_k_genes", type=int, default=1800)
    p.add_argument("--rank_fit_cells", type=int, default=50)
    p.add_argument("--max_new_tokens", type=int, default=320)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt_path", type=str, default=None,
               help="Base path for per-rank checkpoint files (rank suffix will be appended).")
    p.add_argument("--ckpt_batch_stride", type=int, default=200,
                help="Save checkpoint every N minibatches (per rank).")
    p.add_argument("--resume", action="store_true",
                help="If set, resume from existing checkpoint(s) if present.")



    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)


    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)



    rank, world_size, local_rank = get_dist_info()

    if torch.cuda.is_available():
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        # If Slurm restricted visibility to 1 GPU, always use cuda:0
        if vis and ("," not in vis):
            device = torch.device("cuda:0")
        else:
            device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    #claude block
    if world_size > 1 and torch.distributed.is_available() and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    print(f"[Rank {rank}] WORLD_SIZE={world_size} LOCAL_RANK={local_rank} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} device={device}")

    csmodel = cs.CSModel(
        model_name_or_path=args.model_name_or_path,
        save_dir=args.save_dir,
        save_name="pretrained_only",
    )

    hf_model = AutoModelForCausalLM.from_pretrained(
        csmodel.model_name_or_path,
        dtype=torch.float16,
        attn_implementation="sdpa",
        device_map = None
    )
    
    hf_model.to(device)
    hf_model.eval()

    print("HF model device:", next(hf_model.parameters()).device)

    summary_A = zero_shot_inference("/research/projects/rna_lab/x-atlas/HCT116_filtered_dual.h5ad", args, hf_model, csmodel, rank, world_size, device)
    #summary_B = zero_shot_inference("/research/projects/rna_lab/x-atlas/HEK293T_filtered_dual_guide_cells.h5ad", args, hf_model, csmodel, rank, world_size, device)

    # with open(os.path.join(args.save_dir, f"H5AD_A_summary_rank{rank}.json"), "w") as f:
    #     json.dump(summary_A, f, indent=2)
    # with open(os.path.join(args.save_dir, f"H5AD_B_summary_rank{rank}.json"), "w") as f:
    #     json.dump(summary_B, f, indent=2)

if dist.is_initialized(): 
        dist.destroy_process_group()


if __name__ == "__main__":
    main()