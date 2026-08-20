import anndata as ad
import pandas as pd
import torch
import numpy as np
import scipy as sp
from itertools import islice
from collections import defaultdict
import argparse

def normalize(name: str) -> str:
    """Normalize gene names by removing '-1', '-2', '-3' suffixes."""

    if name.count('-') > 0:
        base, suffix = name.rsplit('-', 1)
        if suffix.isdigit():
            return base
    return name


def create_cell_gene_matrix_batch(adata, state_to_xatlas_cols, start, end):
    X_state = np.zeros((end-start, len(state_to_xatlas_cols)), dtype=np.float32) 
    
    dest_indices = []
    source_indices = []
    for destination_col_idx, source_col_idx in enumerate(state_to_xatlas_cols):
        if source_col_idx is not None:
            dest_indices.append(destination_col_idx)
            source_indices.append(source_col_idx)

    X_batch_HVG = adata.X[start:end,:][:, source_indices]
    X_state[:, dest_indices] = X_batch_HVG.toarray()
    
    return X_state



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', type=int, required = True)
    ap.add_argument('--end', type=int, required = True)
    ap.add_argument('--xatlas' ,required=True)
    ap.add_argument('--ckpt' ,required=True)
    ap.add_argument('--output',required=True)


    args = ap.parse_args()

    #load STATE genes
    #ckpt = torch.load("/research/projects/rna_lab/perturb_benchmark/state/checkpoints/rpe/best.ckpt", map_location="cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    hparams = ckpt["hyper_parameters"]
    state_genes_name = hparams["gene_names"]

    #load xatlas gene_metadata
    #xatlas_adata = ad.read_h5ad('/research/projects/rna_lab/x-atlas/HCT116_filtered_dual.h5ad', backed = 'r')
    xatlas_adata = ad.read_h5ad(args.xatlas, backed = 'r')

    #hek293T_adata = ad.read_h5ad('/research/projects/rna_lab/x-atlas/HEK293T_filtered_dual_guide_cells.h5ad')

    #create a map from normalised genename in adata to its index in adata
    xatlas_name_to_col = defaultdict(list)
    missing_genes = []


    for col, gene_name in enumerate(xatlas_adata.var_names):
        key = normalize(gene_name) 
        xatlas_name_to_col[key].append(col)

    #convert STATE gene names to columns so that we know which ones are in xatlas
    #array of state gene column ids from x_atlas 

    state_in_xatlas_cols = []
    for state_gene in state_genes_name:
        #return [] if the name_to_col entry doesnt exist
        cols = xatlas_name_to_col.get(normalize(state_gene), [])

        col = next((c for c in cols if xatlas_adata.var_names[c] == state_gene), 
                    cols[0] if cols else None)
        if col is None:                        
            missing_genes.append(state_gene)
        
        state_in_xatlas_cols.append(col)

    print("Finished streaming all batches.")

    X_test = create_cell_gene_matrix_batch(
        xatlas_adata,
        state_in_xatlas_cols,
        args.start,
        args.end
    )

    adata = ad.AnnData(
        X=X_test,
        obs=xatlas_adata.obs.iloc[args.start : args.end].copy(),
        var=pd.DataFrame(index=state_genes_name)
    )

    src = adata.obs["gene_target"].astype(str)

    # Normalize: lowercase, strip whitespace
    norm = src.str.strip().str.lower()

    # If anything contains "non-targeting" (case-insensitive), force exact token
    norm = norm.where(~norm.str.contains("non-targeting", regex=False), "non-targeting")

    adata.obs["gene"] = norm

    adata.obs['gem_group'] = adata.obs['sample'].astype(str)


    adata.write_h5ad(args.output)

main()

# state emb transform \
#   --model-folder /research/projects/rna_lab/perturb_benchmark/state/checkpoints/se/ \
#   --input /research/projects/rna_lab/perturb_benchmark/state/state_aligned.h5ad \
#   --output /research/projects/rna_lab/perturb_benchmark/state/state_embeddings.h5ad
