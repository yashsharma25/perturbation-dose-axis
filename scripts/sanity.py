import anndata as ad
import numpy as np
import sys

path = sys.argv[1]
a = ad.read_h5ad(path)

def arr(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)

X = arr(a.X)

print("FILE:", path)
print("shape:", a.shape, "X dtype:", X.dtype)
print("X min/mean/max:", float(X.min()), float(X.mean()), float(X.max()))
print("fraction exactly zero:", float((X == 0).mean()))
print("obs cols:", list(a.obs.columns))
print("obsm keys:", list(a.obsm.keys()))
for k in a.obsm.keys():
    B = arr(a.obsm[k])
    print(f"  obsm[{k}] shape={B.shape} dtype={B.dtype} min/mean/max={B.min():.4g}/{B.mean():.4g}/{B.max():.4g}")

# quick perturbation check if present
for col in ["gene", "gene_target", "guide_target"]:
    if col in a.obs:
        vals = a.obs[col].astype(str)
        print(f"{col}: unique={vals.nunique()}, has 'non-targeting' exactly? {('non-targeting' in set(vals))}")
