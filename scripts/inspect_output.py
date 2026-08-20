import anndata as ad
import numpy as np
import pandas as pd

adata = ad.read_h5ad("/research/projects/rna_lab/perturb_benchmark/state/runs/HCT116/batch_000000/infer.h5ad")

print("X shape:", adata.X.shape, "dtype:", getattr(adata.X, "dtype", None))
print("pred shape:", adata.layers["pred"].shape, "dtype:", adata.layers["pred"].dtype)

# 1) Are predictions non-trivial?
pred = adata.layers["pred"]
print("pred min/mean/max:", float(pred.min()), float(pred.mean()), float(pred.max()))
print("fraction exactly zero:", float(np.mean(pred == 0.0)))

# 2) Do controls vs non-controls differ in bulk? (quick signal check)
ctl_mask = (adata.obs["gene"].astype(str).values == "non-targeting")
mu_ctl = pred[ctl_mask].mean(axis=0)
mu_all = pred.mean(axis=0)
print("mean abs diff (all vs control):", float(np.mean(np.abs(mu_all - mu_ctl))))


counts = adata.obs["gene"].value_counts()
print(counts.head(30))
print("num unique perts:", counts.shape[0])
print("controls:", int((adata.obs["gene"]=="non-targeting").sum()))