import anndata as ad
import numpy as np
import torch
from state.tx.models.state_transition import StateTransitionPerturbationModel

CKPT = "/research/projects/rna_lab/perturb_benchmark/state/checkpoints/rpe/best.ckpt"
INFER_H5AD = "/research/projects/rna_lab/perturb_benchmark/state/state_predictions.h5ad"
OUT_H5AD = "/research/projects/rna_lab/perturb_benchmark/state/state_pred_gene.h5ad"

# Load infer output (real X + predicted latent in obsm["X_state"])
adata = ad.read_h5ad(INFER_H5AD)
Z = torch.tensor(np.asarray(adata.obsm["X_state"]), dtype=torch.float32)

# Load model and decode latent -> genes (2000)
model = StateTransitionPerturbationModel.load_from_checkpoint(CKPT, map_location="cpu")
model.eval()

with torch.no_grad():
    # gene_decoder maps 2058 -> 2000
    pred = model.gene_decoder(Z).cpu().numpy().astype(np.float32)

# Write predicted genes into .X of a new AnnData (keep obs/var identical)
ap = ad.AnnData(X=pred, obs=adata.obs.copy(), var=adata.var.copy())
ap.var_names = adata.var_names.copy()
ap.write_h5ad(OUT_H5AD)

print("Saved:", OUT_H5AD)
print("pred shape:", pred.shape, "min/mean/max:", float(pred.min()), float(pred.mean()), float(pred.max()))



cd /research/projects/rna_lab/perturb_benchmark/state/
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
setsid nohup python -u nsr_rho_screen_2.py \
  --h5ad /research/projects/rna_lab/x-atlas/HEK293T_filtered_dual_guide_cells.h5ad \
  --guide_calls /research/projects/rna_lab/x-atlas/HEK293T_filtered_guide_calls_per_cell.csv \
  --split split_hek_v2.csv --eval_set eval_set_v2.csv \
  --memory_budget_gb 8 --out nsr_rho_hek \
  > nsr_rho_hek.log 2> nsr_rho_hek.err < /dev/null &
echo $! > nsr_rho_hek.pid