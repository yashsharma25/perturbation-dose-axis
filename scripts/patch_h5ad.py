import anndata as ad
import pandas as pd


#load the h5ad file
adata = ad.read_h5ad('/research/projects/rna_lab/perturb_benchmark/state/state_embeddings.h5ad')

adata.obs['gene'] = adata.obs['gene_target'].astype(str)

adata.obs.loc[adata.obs['gene'] == 'Non-Targeting', 'gene'] = 'non-targeting'

adata.obs['gem_group'] = adata.obs['sample'].astype(str)

adata.write_h5ad("/research/projects/rna_lab/perturb_benchmark/state/fixed_state_embeddings.h5ad")


# state tx infer \
# --model-dir /research/projects/rna_lab/perturb_benchmark/state/checkpoints/rpe  \
# --checkpoint /research/projects/rna_lab/perturb_benchmark/state/checkpoints/rpe/best.ckpt \
# --adata /research/projects/rna_lab/perturb_benchmark/state/state_embeddings_fixpert.h5ad \
# --pert-col gene \
# --embed-key X_state \
# --output state_predictions.h5ad

# We want to know which metrics did STATE compute
# Then we want to add the structural integrity metrics
# And then we want to check QC
