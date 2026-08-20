import anndata as ad
import pandas as pd
import torch
import numpy as np
import scipy as sp
from itertools import islice
from collections import defaultdict

#load STATE genes
ckpt = torch.load("/research/projects/rna_lab/perturb_benchmark/state/checkpoints/rpe/best.ckpt", map_location="cpu")
hparams = ckpt["hyper_parameters"]
state_genes_name = hparams["gene_names"]

#hct116_adata = ad.read_h5ad('/research/projects/rna_lab/x-atlas/HCT116_filtered_dual.h5ad', backed = 'r')

def normalize(name: str) -> str:
    """Normalize gene names by removing '-1', '-2', '-3' suffixes."""

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
    hct116_adata = ad.read_h5ad('/research/projects/rna_lab/x-atlas/HCT116_filtered_dual.h5ad', backed = 'r')
    #hek293T_adata = ad.read_h5ad('/research/projects/rna_lab/x-atlas/HEK293T_filtered_dual_guide_cells.h5ad')

    #create a map from normalised genename in adata to its index in adata
    xatlas_name_to_col = defaultdict(list)
    missing_genes = []


    for col, gene_name in enumerate(hct116_adata.var_names):
        key = normalize(gene_name) 
        xatlas_name_to_col[key].append(col)

    #convert STATE gene names to columns so that we know which ones are in xatlas
    #array of state gene column ids from x_atlas 

    state_in_xatlas_cols = []
    for state_gene in state_genes_name:
        #return [] if the name_to_col entry doesnt exist
        cols = xatlas_name_to_col.get(normalize(state_gene), [])

        col = next((c for c in cols if hct116_adata.var_names[c] == state_gene), 
                    cols[0] if cols else None)
        if col is None:                        
            missing_genes.append(state_gene)
        
        state_in_xatlas_cols.append(col)

    # print("Mapped STATE genes:", sum(c is not None for c in state_in_xatlas_cols))
    # print("Missing STATE genes:", len(missing_genes))

    # if missing_genes:
    #     print("Example missing genes:", missing_genes[:10])

    # mapped = sum(t is not None for t in state_in_xatlas_cols)
    # missing = sum(t is None for t in state_in_xatlas_cols)

    # print("Mapped cols:", mapped)
    # print("Missing cols:", missing)

    

    def iter_batches(n_obs, batch_size):
        for start in range(0, n_obs, batch_size):
            end = min(start + batch_size, n_obs)
            yield start, end

    batch_size = 1024


    print("Finished streaming all batches.")
    # Take one batch
    batch_indices = iter_batches(hct116_adata.n_obs, batch_size)
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

    test_adata.write_h5ad("state_aligned.h5ad")

main()


# state emb transform \
#   --model-folder /research/projects/rna_lab/perturb_benchmark/state/checkpoints/se/ \
#   --input /research/projects/rna_lab/perturb_benchmark/state/state_aligned.h5ad \
#   --output /research/projects/rna_lab/perturb_benchmark/state/state_embeddings.h5ad