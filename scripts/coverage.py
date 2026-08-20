import anndata as ad
import pickle

DATA = "/research/projects/rna_lab/x-atlas/HEK293T_filtered_dual_guide_cells.h5ad" 
PERT_COL = "gene_target"
CONTROL = "non-targeting"

# load model pert vocab
with open("/research/projects/rna_lab/perturb_benchmark/state/checkpoints/rpe/var_dims.pkl", "rb") as f:
    vd = dict(pickle.load(f))
model_perts = set(map(str, vd["pert_names"]))

adata = ad.read_h5ad(DATA, backed="r")  # obs is accessible

perts = adata.obs[PERT_COL].astype(str)
non_ctl = perts != CONTROL

dataset_unique = set(perts[non_ctl].unique())
in_vocab = dataset_unique & model_perts

unique_cov = 100 * len(in_vocab) / max(1, len(dataset_unique))
cell_cov = 100 * (non_ctl & perts.isin(list(model_perts))).sum() / max(1, non_ctl.sum())

print("Unique perts:", len(dataset_unique))
print("In-vocab perts:", len(in_vocab))
print("Unique coverage %:", unique_cov)
print("Cell-weighted coverage %:", cell_cov)

# cell-eval run \
#     -ap /research/projects/rna_lab/perturb_benchmark/state/xatlas_state_gene_predictions.h5ad \
#     -ar /research/projects/rna_lab/perturb_benchmark/state/xatlas_state_embeddings.h5ad \
#     --num-threads 64 \
#     --profile full