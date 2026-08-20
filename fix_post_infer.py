# fix_for_celleval.py
import anndata as ad
import numpy as np
import argparse
from pathlib import Path
import torch
from state.tx.models.state_transition import StateTransitionPerturbationModel

CKPT = "/research/projects/rna_lab/perturb_benchmark/state/checkpoints/rpe/best.ckpt"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_file' ,required=True)
    ap.add_argument('--pred_file', required=True)

    args = ap.parse_args()

    REAL_IN = Path(args.input_file)
    REAL_OUT = REAL_IN.with_name(REAL_IN.stem + "_celleval.h5ad")

    PRED_In = Path(args.pred_file)
    PRED_OUT = PRED_In.with_name(PRED_In.stem + "_celleval.h5ad")

    real = ad.read_h5ad(args.input_file)
    pred_adata = ad.read_h5ad(args.pred_file)


    CONTROL = "non-targeting"

    # Hard sanity checks (you already did these, but keep them so the fix is safe)
    if real.n_obs != pred_adata.n_obs:
        raise ValueError(f"n_obs mismatch: real={real.n_obs}, pred_adata={pred_adata.n_obs}")
    if real.n_vars != pred_adata.n_vars:
        raise ValueError(f"n_vars mismatch: real={real.n_vars}, pred_adata={pred_adata.n_vars}")
    if not np.array_equal(real.var_names.values, pred_adata.var_names.values):
        raise ValueError("var_names differ between real and pred_adata (must be identical and same order)")

    # We want a single canonical perturbation label column: target_gene
    # Prefer 'gene' because it contains exact 'non-targeting' in your pred_adata file.
    if "gene" not in pred_adata.obs.columns:
        raise ValueError(f"'gene' column not found in pred_adata.obs. Found: {list(pred_adata.obs.columns)}")

    # Create in pred_adata
    pred_adata.obs["target_gene"] = pred_adata.obs["gene"].astype(str).str.strip()

    real.obs["target_gene"] = pred_adata.obs["target_gene"].values

    # Normalize control label to exactly "non-targeting" (just in case of capitalization drift)
    real.obs["target_gene"] = real.obs["target_gene"].astype(str).str.strip()
    pred_adata.obs["target_gene"] = pred_adata.obs["target_gene"].astype(str).str.strip()

    # Force only the control label to canonical spelling (do NOT lowercase everything globally)
    real.obs.loc[real.obs["target_gene"].str.lower() == CONTROL, "target_gene"] = CONTROL
    pred_adata.obs.loc[pred_adata.obs["target_gene"].str.lower() == CONTROL, "target_gene"] = CONTROL

    # Final check: control exists in both
    for name, adata in [("REAL", real), ("PRED", pred_adata)]:
        present = (adata.obs["target_gene"] == CONTROL).any()
        if not present:
            vals = np.unique(adata.obs["target_gene"].values)[:25]
            raise ValueError(f"{name}: control '{CONTROL}' not present. First unique labels: {vals}")


    #Decode latent genes
    Z = torch.tensor(np.asarray(pred_adata.obsm["X_state"]), dtype=torch.float32)

    # Load model and decode latent -> genes (2000)
    model = StateTransitionPerturbationModel.load_from_checkpoint(CKPT, map_location="cpu")
    model.eval()

    with torch.no_grad():
        # gene_decoder maps 2058 -> 2000
        pred_array = model.gene_decoder(Z).cpu().numpy().astype(np.float32)

    # Write predicted genes into .X of a new AnnData (keep obs/var identical)
    ap = ad.AnnData(X=pred_array, obs=pred_adata.obs.copy(), var=pred_adata.var.copy())
    ap.var_names = pred_adata.var_names.copy()

    real.write_h5ad(REAL_OUT)
    ap.write_h5ad(PRED_OUT)

    print("Wrote:")
    print("  REAL:", REAL_OUT)
    print("  PRED:", PRED_OUT)
    print("Control count (real):", int((real.obs["target_gene"] == CONTROL).sum()))
    print("Control count (pred_adata):", int((pred_adata.obs["target_gene"] == CONTROL).sum()))

if __name__ == "__main__":
    main()