import anndata as ad
import pandas as pd
import torch
import numpy as np
import scipy as sp
from itertools import islice


#load STATE genes
# ckpt = torch.load("/research/projects/rna_lab/perturb_benchmark/state/checkpoints/rpe/final.ckpt", map_location="cpu")
# hparams = ckpt["hyper_parameters"]
# state_genes_name = hparams["gene_names"]

hct116_adata = ad.read_h5ad('/research/projects/rna_lab/x-atlas/HCT116_filtered_dual.h5ad', backed = 'r')


# ---------------------------------------------------------
# 2. Identify control cells
#    Control = BOTH guides are non-targeting
# ---------------------------------------------------------
is_control = (
    hct116_adata.obs["guide_target"]
    .str.split("|")
    .apply(lambda gs: all(g.startswith("non-targeting") for g in gs))
)


# ---------------------------------------------------------
# 3. Count control cells per sample (batch)
# ---------------------------------------------------------
control_counts_per_sample = (
    hct116_adata.obs[is_control]
    .groupby("sample")
    .size()
    .rename("n_control_cells_in_sample")
)

# ---------------------------------------------------------
# 4. Attach control counts to every cell
# ---------------------------------------------------------
hct116_adata.obs["n_control_cells_in_sample"] = (
    hct116_adata.obs["sample"]
    .map(control_counts_per_sample)
    .fillna(0)
    .astype(int)
)

# ---------------------------------------------------------
# 5. Compute number of controls per perturbation
#    (exclude control perturbations themselves)
# ---------------------------------------------------------
controls_per_perturbation = (
    hct116_adata.obs
    .loc[~is_control]
    .groupby("guide_target")["n_control_cells_in_sample"]
    .first()   # safe: perturbations are batch-specific
    .rename("n_controls")
    .reset_index()
)

# ---------------------------------------------------------
# 6. Sanity check: ensure perturbations do not span batches
# ---------------------------------------------------------
perturbation_batch_check = (
    hct116_adata.obs
    .loc[~is_control]
    .groupby("guide_target")["sample"]
    .nunique()
)

print("Batch count per perturbation:")
print(perturbation_batch_check.value_counts())

# ---------------------------------------------------------
# 7. Inspect result
# ---------------------------------------------------------
print("\nControls per perturbation:")
print(controls_per_perturbation)


# How many perturbations have at least 5 control cells?
n_perturbations_with_5_controls = (
    controls_per_perturbation["n_controls"] >= 5
).sum()

# Total number of perturbations
total_perturbations = controls_per_perturbation.shape[0]

print(f"Perturbations with ≥5 control cells: {n_perturbations_with_5_controls}")
print(f"Total perturbations: {total_perturbations}")
print(f"Fraction: {n_perturbations_with_5_controls / total_perturbations:.3f}")


# One row per sample with its control count
controls_per_sample = (
    hct116_adata.obs[['sample', 'n_control_cells_in_sample']]
    .drop_duplicates()
)

# How many samples have more than 5 control cells?
n_samples_with_gt5_controls = (
    controls_per_sample['n_control_cells_in_sample'] > 5
).sum()

# Total number of samples
total_samples = controls_per_sample.shape[0]

print(f"Samples with >5 control cells: {n_samples_with_gt5_controls}")
print(f"Total samples: {total_samples}")
print(f"Fraction: {n_samples_with_gt5_controls / total_samples:.3f}")



# Count unique perturbations = unique (guide_target, sample) pairs
n_unique_perturbations = (
    hct116_adata.obs.loc[~is_control, ["guide_target", "sample"]]
    .drop_duplicates()
    .shape[0]
)

print(f"Number of unique perturbations: {n_unique_perturbations}")

# Number of unique perturbed genes (exclude controls)
n_unique_perturbed_genes = (
    hct116_adata.obs
    .loc[hct116_adata.obs["gene_target"] != "non-targeting", "gene_target"]
    .nunique()
)

print(f"Number of unique perturbed genes: {n_unique_perturbed_genes}")


#hct116_adata = hct116_adata_big.X[:100, :]
#print(hct116_adata.shape)
#print(type(hct116_adata))

def normalize(name: str) -> str:
    """Normalize gene names by removing '-1', '-2', '-3' suffixes."""

    # Split only once at the end
    if name.count('-') > 0:
        base, suffix = name.rsplit('-', 1)
        if suffix.isdigit():
            return base
    return name


def create_cell_gene_matrix_batch(adata, state_to_xatlas_cols, batch_indices):
    X_state = np.zeros((len(batch_indices), len(state_to_xatlas_cols)), dtype=np.float32) 
    
    dest_indices = []
    source_indices = []
    for destination_col_idx, source_col_idx in enumerate(state_to_xatlas_cols):
        if source_col_idx is not None:
            dest_indices.append(destination_col_idx)
            source_indices.append(source_col_idx)

    X_batch_HVG = adata.X[batch_indices,:][:, source_indices]
    X_state[:, dest_indices] = X_batch_HVG.toarray()
    
    return X_state



def main():
    #load xatlas gene_metadata
    gene_df = pd.read_parquet("/research/projects/rna_lab/x-atlas/gene_metadata.parquet")

    # Add normalized names
    gene_df["norm_name"] = gene_df["gene_name"].apply(normalize)


    #get gene names from the gene_metadata
    xatlas_gene_names = gene_df["gene_name"].astype('string').tolist()
    xatlas_gene_ids = gene_df["gene_token_id"].tolist()


    name_to_col = {}
    missing_genes = []

    for col, gene_name in enumerate(hct116_adata.var_names):
        name_to_col[normalize(gene_name)] = col


    #convert STATE gene names to tokens so that we know which ones are in xatlas
    #array of state gene token ids from x_atlas 
    state_in_xatlas_cols = []
    for state_gene in state_genes_name:
        g_norm = normalize(state_gene)
        col_id = name_to_col.get(g_norm)

        if col_id is None:                        
            state_in_xatlas_cols.append(None)
            missing_genes.append(state_gene)


        else:
            state_in_xatlas_cols.append(col_id)

    # print("Mapped STATE genes:", sum(c is not None for c in state_in_xatlas_cols))
    # print("Missing STATE genes:", len(missing_genes))

    # if missing_genes:
    #     print("Example missing genes:", missing_genes[:10])

    # mapped = sum(t is not None for t in state_in_xatlas_cols)
    # missing = sum(t is None for t in state_in_xatlas_cols)

    # print("Mapped tokens:", mapped)
    # print("Missing tokens:", missing)

    # hct116_adata = ad.read_h5ad('/research/projects/rna_lab/x-atlas/HCT116_filtered_dual.h5ad', backed = 'r')
    #hek293T_adata = ad.read_h5ad('/research/projects/rna_lab/x-atlas/HEK293T_filtered_dual_guide_cells.h5ad')
    # example_batch = [0, 4096]
    # X_state = create_cell_gene_matrix_batch(hct116_adata, state_in_xatlas_cols ,example_batch)

    # print("Batch output shape:", X_state.shape)
    # print("First 5x5 block:\n", X_state[:5, :5])

    def iter_batches(n_obs, batch_size):
        for start in range(0, n_obs, batch_size):
            end = min(start + batch_size, n_obs)
            yield start, end

    BATCH_SIZE = 1024  # keep modest for now


    print("Finished streaming all batches.")
    # Take one batch
    batch_indices = list(range(0, 1024))
    X_test = create_cell_gene_matrix_batch(
        hct116_adata,
        state_in_xatlas_cols,
        batch_indices
    )

    # Build a tiny AnnData for testing STATE
    test_adata = ad.AnnData(
        X=X_test,
        obs=hct116_adata.obs.iloc[batch_indices].copy(),
        var=pd.DataFrame(index=state_genes_name)
    )

    test_adata.write_h5ad("state_step4_test.h5ad")

#main()