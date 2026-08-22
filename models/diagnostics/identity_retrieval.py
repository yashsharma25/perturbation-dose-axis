import numpy as np
import pandas as pd
import torch
import anndata as ad

from scipy.stats import rankdata


C2S_FILE = "save/c2s_v2/HCT116_filtered_dual_per_pert_real_null.pt"

STATE_BATCHES = [
    "runs/HCT116/batch_000000",
    "runs/HCT116/batch_000001",
]

H5AD = "/research/projects/rna_lab/x-atlas/HCT116_filtered_dual.h5ad"

EPS = 1e-4
N_PERM = 2000
RNG = np.random.default_rng(42)


# ------------------------------------------------------------
# 1. Load C2S
# ------------------------------------------------------------

print("[load] C2S")
obj = torch.load(C2S_FILE, map_location="cpu", weights_only=False)

real = obj["pert_results"]
null = obj["pert_null"]

c2s_key = {k.upper(): k for k in real}


# ------------------------------------------------------------
# 2. Find STATE targets
# ------------------------------------------------------------

state_targets = set()

for b in STATE_BATCHES:
    x = pd.read_csv(f"{b}/results.csv", usecols=["perturbation"])
    state_targets.update(x["perturbation"].astype(str).str.upper())

targets = sorted(set(c2s_key) & state_targets)

print("[targets] C2S:", len(c2s_key))
print("[targets] STATE:", len(state_targets))
print("[targets] common:", len(targets))

if len(targets) < 700:
    raise RuntimeError("Unexpectedly small C2S/STATE target intersection")

target_to_i = {t: i for i, t in enumerate(targets)}


# ------------------------------------------------------------
# 3. Establish a COMMON gene space
#
# STATE's 2000 genes are available directly from the H5AD.
# Map those genes onto the 38,606-gene C2S vectors.
# ------------------------------------------------------------

print("[genes] reading STATE feature set")

a_state = ad.read_h5ad(
    f"{STATE_BATCHES[0]}/infer_celleval.h5ad",
    backed="r",
)
state_features = list(map(str, a_state.var_names))
a_state.file.close()

print("[genes] STATE features:", len(state_features))


print("[genes] mapping to X-Atlas gene order")

a = ad.read_h5ad(H5AD, backed="r")
hct_genes = list(map(str, a.var_names))
a.file.close()

if len(hct_genes) != obj["n_genes"]:
    raise RuntimeError(
        f"H5AD has {len(hct_genes)} genes but C2S vectors have {obj['n_genes']}"
    )

hct_index = {g.upper(): i for i, g in enumerate(hct_genes)}

common_features = [
    g for g in state_features
    if g.upper() in hct_index
]

c2s_gene_idx = np.array(
    [hct_index[g.upper()] for g in common_features],
    dtype=int,
)

print("[genes] shared C2S/STATE:", len(common_features))


# ------------------------------------------------------------
# 4. Build C2S predicted / actual / null expression matrices
# ------------------------------------------------------------

P = len(targets)
G = len(common_features)

actual_mean = np.zeros((P, G), dtype=np.float64)
ctrl_mean   = np.zeros((P, G), dtype=np.float64)
c2s_pred    = np.zeros((P, G), dtype=np.float64)
c2s_null    = np.zeros((P, G), dtype=np.float64)

for i, t in enumerate(targets):

    k = c2s_key[t]

    d = real[k]
    dn = null[k]

    n = d["count"]
    nn = dn["count"]

    actual_mean[i] = d["actual_sum"][c2s_gene_idx] / n
    ctrl_mean[i]   = d["ctrl_sum"][c2s_gene_idx] / n
    c2s_pred[i]    = d["pred_sum"][c2s_gene_idx] / n
    c2s_null[i]    = dn["pred_sum"][c2s_gene_idx] / nn


# ------------------------------------------------------------
# 5. Restrict to reasonably expressed genes.
#
# This mirrors the existing benchmark's use of a control-
# expression filter and prevents near-zero denominators from
# dominating log fold change.
#
# Also REMOVE every perturbation target gene from the feature
# space, so identity retrieval cannot be won trivially by
# predicting suppression of the target itself.
# ------------------------------------------------------------

control_expression = ctrl_mean.mean(axis=0)
threshold = np.quantile(control_expression, 0.60)

target_gene_set = set(targets)

keep = control_expression >= threshold

keep &= np.array([
    g.upper() not in target_gene_set
    for g in common_features
])

selected_features = np.asarray(common_features)[keep]

actual_mean = actual_mean[:, keep]
ctrl_mean   = ctrl_mean[:, keep]
c2s_pred    = c2s_pred[:, keep]
c2s_null    = c2s_null[:, keep]

print("[genes] after expression filter + target-gene exclusion:",
      len(selected_features))


# ------------------------------------------------------------
# 6. Convert C2S to response vectors
# ------------------------------------------------------------

def logfc(x, ctrl):
    x = np.maximum(x, 0)
    ctrl = np.maximum(ctrl, 0)

    return np.log(
        (x + EPS) /
        (ctrl + EPS)
    )


Y = logfc(actual_mean, ctrl_mean)
X_c2s = logfc(c2s_pred, ctrl_mean)
X_null = logfc(c2s_null, ctrl_mean)


# ------------------------------------------------------------
# 7. Build STATE predictions over the SAME targets and genes
#
# Read CSVs in chunks because each is ~18.7 million rows.
# Average the two STATE batches.
# ------------------------------------------------------------

feature_to_j = {
    g.upper(): j
    for j, g in enumerate(selected_features)
}

G2 = len(selected_features)

state_sum = np.zeros((P, G2), dtype=np.float64)
state_n   = np.zeros((P, G2), dtype=np.int16)

print("[STATE] streaming prediction files")

for batch in STATE_BATCHES:

    path = f"{batch}/pred_de.csv"
    print(" ", path)

    for chunk in pd.read_csv(
        path,
        usecols=[
            "target",
            "feature",
            "target_mean",
            "reference_mean",
        ],
        chunksize=500_000,
    ):

        t = chunk["target"].astype(str).str.upper()
        f = chunk["feature"].astype(str).str.upper()

        ti = t.map(target_to_i)
        gj = f.map(feature_to_j)

        mask = ti.notna() & gj.notna()

        if not mask.any():
            continue

        ii = ti[mask].astype(int).to_numpy()
        jj = gj[mask].astype(int).to_numpy()

        tm = chunk.loc[mask, "target_mean"].to_numpy(dtype=float)
        rm = chunk.loc[mask, "reference_mean"].to_numpy(dtype=float)

        tm = np.maximum(tm, 0)
        rm = np.maximum(rm, 0)

        response = np.log(
            (tm + EPS) /
            (rm + EPS)
        )

        np.add.at(state_sum, (ii, jj), response)
        np.add.at(state_n,   (ii, jj), 1)


missing = state_n == 0

print("[STATE] missing target/gene entries:", missing.sum())
print("[STATE] entries with two batches:", np.sum(state_n == 2))

if missing.any():

    bad_rows = np.where(missing.any(axis=1))[0]

    print("[STATE] dropping targets with missing values:",
          len(bad_rows))

    good_rows = ~missing.any(axis=1)

    targets = np.asarray(targets)[good_rows].tolist()

    Y       = Y[good_rows]
    X_c2s   = X_c2s[good_rows]
    X_null  = X_null[good_rows]

    state_sum = state_sum[good_rows]
    state_n   = state_n[good_rows]

state = state_sum / state_n

P = len(targets)

print("[final] perturbations:", P)
print("[final] genes:", state.shape[1])


# ------------------------------------------------------------
# 8. Remove the generic response direction
#
# Same basic idea as the perturbation-specific metric in the
# paper: center each response across genes, then project out
# the generic observed response direction.
# ------------------------------------------------------------

generic = Y.mean(axis=0)


def remove_generic(M, g):

    Mc = M - M.mean(axis=1, keepdims=True)
    gc = g - g.mean()

    denom = gc @ gc

    if denom <= 1e-15:
        raise RuntimeError("Generic direction has zero variance")

    beta = (Mc @ gc) / denom

    return Mc - beta[:, None] * gc[None, :]


Y_specific     = remove_generic(Y, generic)
c2s_specific   = remove_generic(X_c2s, generic)
null_specific  = remove_generic(X_null, generic)
state_specific = remove_generic(state, generic)


# ------------------------------------------------------------
# 9. Perturbation identity retrieval
# ------------------------------------------------------------

def normalize_rows(X):
    norm = np.linalg.norm(X, axis=1, keepdims=True)

    zero = norm[:, 0] < 1e-12

    if zero.any():
        print("[warning] near-zero response vectors:", zero.sum())

    norm[norm < 1e-12] = 1.0

    return X / norm


Yn = normalize_rows(Y_specific)


def retrieval(name, X):

    Xn = normalize_rows(X)

    # rows = predicted perturbations
    # cols = observed perturbation identities
    similarity = Xn @ Yn.T

    # rank 1 = best observed response
    rank_matrix = np.vstack([
        rankdata(-row, method="average")
        for row in similarity
    ])

    true_ranks = rank_matrix[
        np.arange(P),
        np.arange(P)
    ]

    top1 = np.mean(true_ranks <= 1)
    top10 = np.mean(true_ranks <= 10)
    median_rank = np.median(true_ranks)
    mrr = np.mean(1.0 / true_ranks)

    # permutation null: scramble which observed identity is
    # considered correct for each prediction.
    null_mrr = np.empty(N_PERM)
    null_top10 = np.empty(N_PERM)

    rows = np.arange(P)

    for b in range(N_PERM):
        perm = RNG.permutation(P)
        rr = rank_matrix[rows, perm]

        null_mrr[b] = np.mean(1.0 / rr)
        null_top10[b] = np.mean(rr <= 10)

    p_mrr = (
        1 + np.sum(null_mrr >= mrr)
    ) / (N_PERM + 1)

    p_top10 = (
        1 + np.sum(null_top10 >= top10)
    ) / (N_PERM + 1)

    out = {
        "model": name,
        "n_perturbations": P,
        "n_genes": X.shape[1],
        "top1": top1,
        "top10": top10,
        "median_rank": median_rank,
        "MRR": mrr,
        "perm_p_MRR": p_mrr,
        "perm_p_top10": p_top10,
    }

    return out, true_ranks


results = []
rank_table = pd.DataFrame({
    "perturbation": targets
})

for name, X in [
    ("C2S_real_token", c2s_specific),
    ("C2S_null_token", null_specific),
    ("STATE_zero_shot", state_specific),
]:

    result, ranks = retrieval(name, X)

    results.append(result)

    rank_table[name + "_rank"] = ranks


summary = pd.DataFrame(results)

print()
print("================ RESULTS ================")
print(summary.to_string(index=False))

print()
print("Chance expectations:")
print("Top-1 =", 1 / P)
print("Top-10 =", min(10 / P, 1))
print("Median rank ≈", (P + 1) / 2)

summary.to_csv("identity_retrieval_summary.csv", index=False)
rank_table.to_csv("identity_retrieval_ranks.csv", index=False)

print()
print("saved:")
print("  identity_retrieval_summary.csv")
print("  identity_retrieval_ranks.csv")