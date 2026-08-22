import gc
import numpy as np
import pandas as pd
import torch
import anndata as ad
from scipy import sparse

H5AD = "/research/projects/rna_lab/x-atlas/HCT116_filtered_dual.h5ad"
SPLIT = "split_hct116_v2.csv"

C2S_FILE = "save/c2s_v2/HCT116_filtered_dual_per_pert_real_null.pt"

STATE_RESULTS = [
    "runs/HCT116/batch_000000/results.csv",
    "runs/HCT116/batch_000001/results.csv",
]
STATE_H5AD = "runs/HCT116/batch_000000/infer_celleval.h5ad"

EPS = 1e-4
CHUNK = 512
MAX_CTRL_PER_SAMPLE_HALF = 500
N_PERM = 2000

rng = np.random.default_rng(42)


# ============================================================
# 1. Reconstruct exact 731 perturbation / 756 gene test space
# ============================================================

print("[setup] loading C2S metadata", flush=True)

obj = torch.load(C2S_FILE, map_location="cpu", weights_only=False)
real = obj["pert_results"]
c2s_key = {str(k).upper(): k for k in real}

state_targets = set()

for p in STATE_RESULTS:
    df = pd.read_csv(p, usecols=["perturbation"])
    state_targets.update(
        df["perturbation"].astype(str).str.upper()
    )

targets = sorted(set(c2s_key) & state_targets)

adata = ad.read_h5ad(H5AD, backed="r")

s = ad.read_h5ad(STATE_H5AD, backed="r")
state_features = list(map(str, s.var_names))
s.file.close()

hct_genes = list(map(str, adata.var_names))
hct_index = {g.upper(): i for i, g in enumerate(hct_genes)}

common_features = [
    g for g in state_features
    if g.upper() in hct_index
]

common_idx = np.array(
    [hct_index[g.upper()] for g in common_features],
    dtype=int,
)

# Same 60th percentile control-expression filter
C = np.empty(
    (len(targets), len(common_features)),
    dtype=np.float32,
)

for i, t in enumerate(targets):
    d = real[c2s_key[t]]
    C[i] = d["ctrl_sum"][common_idx] / d["count"]

ctrl_expr = C.mean(axis=0)
threshold = np.quantile(ctrl_expr, 0.60)

target_set = set(targets)

keep = ctrl_expr >= threshold
keep &= np.array([
    g.upper() not in target_set
    for g in common_features
])

selected_features = np.asarray(common_features)[keep]
gene_idx = common_idx[keep]

print("[setup] targets =", len(targets), flush=True)
print("[setup] genes =", len(gene_idx), flush=True)

assert len(targets) == 731
assert len(gene_idx) == 756

# Free C2S completely before touching the large expression matrix.
del C, ctrl_expr, obj, real, c2s_key
gc.collect()

print("[setup] C2S freed", flush=True)


# ============================================================
# 2. Map split to H5AD
# ============================================================

split = pd.read_csv(
    SPLIT,
    usecols=["cell_barcode", "split"],
)

split["split"] = split["split"].astype("category")

rows = adata.obs_names.get_indexer(
    split["cell_barcode"].astype(str)
)

valid = rows >= 0

split = split.loc[valid].copy()
split["row_idx"] = rows[valid]

# Pull metadata once.
rr = split["row_idx"].to_numpy()

split["gene_target"] = (
    adata.obs.iloc[rr]["gene_target"]
    .astype(str)
    .str.upper()
    .to_numpy()
)

split["sample"] = (
    adata.obs.iloc[rr]["sample"]
    .astype(str)
    .to_numpy()
)

print("[split] mapped =", len(split), flush=True)


# ============================================================
# 3. Safe expression reader
#
# Critical difference:
#   adata.X[rows, :]
# NOT
#   adata[rows, genes].X
# ============================================================

def read_small_chunk(rows):
    rows = np.asarray(rows, dtype=int)

    # rows are already sorted before this function is called
    X = adata.X[rows, :]

    if sparse.issparse(X):
        X = X[:, gene_idx].toarray()
    else:
        X = np.asarray(X, dtype=np.float32)
        X = X[:, gene_idx]

    X = np.asarray(X, dtype=np.float32)

    totals = (
        adata.obs.iloc[rows]["total_counts"]
        .to_numpy(dtype=np.float32)
    )

    totals[totals <= 0] = 1.0
    X /= totals[:, None]

    return X


# ============================================================
# 4. Stream perturbation cells directly into group sums
# ============================================================

target_to_i = {t: i for i, t in enumerate(targets)}

pert = split[
    split["gene_target"].isin(targets)
].copy()

half_code = (
    pert["split"]
    .astype(str)
    .map({"selection": 0, "evaluation": 1})
)

pert = pert[half_code.notna()].copy()

half_code = (
    pert["split"]
    .astype(str)
    .map({"selection": 0, "evaluation": 1})
    .astype(int)
    .to_numpy()
)

ti = (
    pert["gene_target"]
    .map(target_to_i)
    .astype(int)
    .to_numpy()
)

# group = target*2 + half
group_id = 2 * ti + half_code

pert_rows = pert["row_idx"].to_numpy(dtype=int)

order = np.argsort(pert_rows)
pert_rows = pert_rows[order]
group_id = group_id[order]

n_groups = len(targets) * 2

pert_sum = np.zeros(
    (n_groups, len(gene_idx)),
    dtype=np.float64,
)

pert_n = np.zeros(n_groups, dtype=np.int64)

print(
    f"[pert] streaming {len(pert_rows):,} cells",
    flush=True,
)

for start in range(0, len(pert_rows), CHUNK):

    end = min(start + CHUNK, len(pert_rows))

    rows_chunk = pert_rows[start:end]
    gids = group_id[start:end]

    X = read_small_chunk(rows_chunk)

    for g in np.unique(gids):
        m = gids == g
        pert_sum[g] += X[m].sum(axis=0)
        pert_n[g] += m.sum()

    if start % (CHUNK * 50) == 0:
        print(
            f"[pert] {end:,}/{len(pert_rows):,}",
            flush=True,
        )

    del X


# ============================================================
# 5. Independent selection/evaluation control means
#
# Build a separate control estimate for each
# (sample, half), also streamed.
# ============================================================

is_control = split["gene_target"].str.contains(
    r"NON[-_ ]?TARGET",
    regex=True,
    na=False,
)

controls = split[is_control].copy()

controls = controls[
    controls["split"].astype(str).isin(
        ["selection", "evaluation"]
    )
]

samples = sorted(pert["sample"].unique())
sample_to_i = {s: i for i, s in enumerate(samples)}

ctrl_rows = []
ctrl_group = []

for half, hc in [("selection", 0), ("evaluation", 1)]:

    ch = controls[
        controls["split"].astype(str) == half
    ]

    for sample in samples:

        x = ch[ch["sample"] == sample]

        if len(x) == 0:
            continue

        rows_s = x["row_idx"].to_numpy(dtype=int)

        if len(rows_s) > MAX_CTRL_PER_SAMPLE_HALF:
            rows_s = rng.choice(
                rows_s,
                MAX_CTRL_PER_SAMPLE_HALF,
                replace=False,
            )

        sid = sample_to_i[sample]
        gid = 2 * sid + hc

        ctrl_rows.extend(rows_s.tolist())
        ctrl_group.extend([gid] * len(rows_s))


ctrl_rows = np.asarray(ctrl_rows, dtype=int)
ctrl_group = np.asarray(ctrl_group, dtype=int)

order = np.argsort(ctrl_rows)
ctrl_rows = ctrl_rows[order]
ctrl_group = ctrl_group[order]

n_ctrl_groups = 2 * len(samples)

ctrl_sum = np.zeros(
    (n_ctrl_groups, len(gene_idx)),
    dtype=np.float64,
)
ctrl_n = np.zeros(n_ctrl_groups, dtype=np.int64)

print(
    f"[ctrl] streaming {len(ctrl_rows):,} sampled controls",
    flush=True,
)

for start in range(0, len(ctrl_rows), CHUNK):

    end = min(start + CHUNK, len(ctrl_rows))

    rc = ctrl_rows[start:end]
    gids = ctrl_group[start:end]

    X = read_small_chunk(rc)

    for g in np.unique(gids):
        m = gids == g
        ctrl_sum[g] += X[m].sum(axis=0)
        ctrl_n[g] += m.sum()

    del X


# Means by (sample, half)
ctrl_mean = np.zeros_like(ctrl_sum)

ok = ctrl_n > 0

ctrl_mean[ok] = (
    ctrl_sum[ok] /
    ctrl_n[ok, None]
)

global_ctrl = {}

for hc in [0, 1]:
    m = np.arange(n_ctrl_groups) % 2 == hc
    m &= ctrl_n > 0

    global_ctrl[hc] = (
        ctrl_sum[m].sum(axis=0) /
        ctrl_n[m].sum()
    )


# ============================================================
# 6. Construct matched logFC response for each target-half
# ============================================================

Y = np.zeros(
    (len(targets), 2, len(gene_idx)),
    dtype=np.float64,
)

valid_target = np.ones(len(targets), dtype=bool)

# Need sample composition per target-half
for i, t in enumerate(targets):

    for half, hc in [("selection", 0), ("evaluation", 1)]:

        gid = 2 * i + hc

        if pert_n[gid] < 5:
            valid_target[i] = False
            continue

        mu = pert_sum[gid] / pert_n[gid]

        g = pert[
            (pert["gene_target"] == t) &
            (pert["split"].astype(str) == half)
        ]

        sample_counts = g["sample"].value_counts()

        baseline = np.zeros(
            len(gene_idx),
            dtype=np.float64,
        )

        total = sample_counts.sum()

        for sample, n in sample_counts.items():

            sid = sample_to_i[sample]
            cgid = 2 * sid + hc

            if ctrl_n[cgid] > 0:
                c = ctrl_mean[cgid]
            else:
                c = global_ctrl[hc]

            baseline += (n / total) * c

        Y[i, hc] = np.log(
            (mu + EPS) /
            (baseline + EPS)
        )


Y = Y[valid_target]
final_targets = np.asarray(targets)[valid_target]

Ys = Y[:, 0]
Ye = Y[:, 1]

print(
    "[final] perturbations =",
    len(final_targets),
    flush=True,
)


# ============================================================
# 7. Generic-response projection
# ============================================================

generic = Ys.mean(axis=0)


def remove_generic(M, g):
    Mc = M - M.mean(axis=1, keepdims=True)
    gc = g - g.mean()

    beta = (Mc @ gc) / (gc @ gc)

    return Mc - beta[:, None] * gc[None, :]


Ys = remove_generic(Ys, generic)
Ye = remove_generic(Ye, generic)


def normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    return X / n


A = normalize(Ys)
B = normalize(Ye)

sim = A @ B.T

P = len(final_targets)

diag = np.diag(sim)

true_rank = (
    1 +
    np.sum(sim > diag[:, None], axis=1)
)

top1 = np.mean(true_rank <= 1)
top10 = np.mean(true_rank <= 10)
median_rank = np.median(true_rank)
mrr = np.mean(1 / true_rank)


# ============================================================
# 8. Permutation test
# ============================================================

order = np.argsort(-sim, axis=1)

ranks = np.empty_like(order, dtype=np.int32)

ranks[
    np.arange(P)[:, None],
    order
] = np.arange(1, P + 1)[None, :]

null_mrr = np.empty(N_PERM)

for b in range(N_PERM):
    perm = rng.permutation(P)
    rr = ranks[np.arange(P), perm]
    null_mrr[b] = np.mean(1 / rr)

p_mrr = (
    1 + np.sum(null_mrr >= mrr)
) / (N_PERM + 1)


# ============================================================
# 9. Results
# ============================================================

print()
print("========== OBSERVED POSITIVE CONTROL ==========")
print("n =", P)
print("genes =", len(gene_idx))
print("top1 =", top1, f"({np.sum(true_rank <= 1)}/{P})")
print("top10 =", top10, f"({np.sum(true_rank <= 10)}/{P})")
print("median rank =", median_rank)
print("MRR =", mrr)
print("permutation p MRR =", p_mrr)

print()
print("Chance:")
print("top1 =", 1 / P)
print("top10 =", 10 / P)
print("median rank ~", (P + 1) / 2)
print(
    "null MRR 95% =",
    np.quantile(null_mrr, [0.025, 0.975]),
)


pd.DataFrame([{
    "n_perturbations": P,
    "n_genes": len(gene_idx),
    "top1": top1,
    "top10": top10,
    "median_rank": median_rank,
    "MRR": mrr,
    "perm_p_MRR": p_mrr,
}]).to_csv(
    "observed_identity_positive_control_summary.csv",
    index=False,
)

pd.DataFrame({
    "perturbation": final_targets,
    "true_rank": true_rank,
}).to_csv(
    "observed_identity_positive_control_ranks.csv",
    index=False,
)

adata.file.close()