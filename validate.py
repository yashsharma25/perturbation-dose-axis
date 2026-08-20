import scanpy as sc
import pandas as pd

REAL_H5AD = "/research/projects/rna_lab/perturb_benchmark/state/runs/HCT116/batch_000000/emb_celleval.h5ad"
PRED_H5AD = "/research/projects/rna_lab/perturb_benchmark/state/runs/HCT116/batch_000000/infer_celleval.h5ad"

PERT_COL = "target_gene"   # or "gene", same idea
CONTROL = "non-targeting"

real = sc.read_h5ad(REAL_H5AD)
pred = sc.read_h5ad(PRED_H5AD)

for name, adata in [("REAL", real), ("PRED", pred)]:
    print(f"\n===== {name} =====")
    vc = adata.obs[PERT_COL].astype(str).str.lower().value_counts()

    print("total cells:", adata.n_obs)
    print("unique perturbations:", vc.shape[0])
    print("control cells:", int(vc.get(CONTROL, 0)))

    print("\nsmallest counts:")
    print(vc.sort_values().head(20).to_string())

    print("\ncount distribution:")
    print(vc.describe())

    print("\n# perturbations with only 1 cell:", int((vc == 1).sum()))
    print("# perturbations with <= 2 cells:", int((vc <= 2).sum()))
    print("# perturbations with <= 5 cells:", int((vc <= 5).sum()))
    print("# perturbations with >= 10 cells:", int((vc >= 10).sum()))