import anndata as ad
import numpy as np

H5AD_PATH = "/research/projects/rna_lab/perturb_benchmark/state/runs/HCT116/batch_000000/infer_celleval.h5ad"

adata = ad.read_h5ad(H5AD_PATH)

print("\n================ BASIC INFO ================")
print("Shape (cells x genes):", adata.shape)
print("X dtype:", adata.X.dtype if adata.X is not None else None)

print("\n================ OBS (per cell metadata) ================")
print("obs columns:", list(adata.obs.columns))
print("obs head:\n", adata.obs.head())

print("\n================ VAR (per gene metadata) ================")
print("var columns:", list(adata.var.columns))
print("var head:\n", adata.var.head())

print("\n================ LAYERS ================")
print("Available layers:", list(adata.layers.keys()))
for k in adata.layers.keys():
    arr = adata.layers[k]
    print(f"Layer '{k}' shape:", arr.shape, "dtype:", arr.dtype)

print("\n================ OBSM (embeddings) ================")
print("obsm keys:", list(adata.obsm.keys()))
for k in adata.obsm.keys():
    print(f"obsm['{k}'] shape:", adata.obsm[k].shape)

print("\n================ VARM ================")
print("varm keys:", list(adata.varm.keys()))

print("\n================ OBSP ================")
print("obsp keys:", list(adata.obsp.keys()))

print("\n================ VARP ================")
print("varp keys:", list(adata.varp.keys()))

print("\n================ UNS (misc / metrics likely here) ================")
print("uns keys:", list(adata.uns.keys()))

for k, v in adata.uns.items():
    print(f"\nuns['{k}'] type:", type(v))
    if isinstance(v, dict):
        print("  dict keys:", list(v.keys()))
    elif isinstance(v, (list, tuple, np.ndarray)):
        print("  length:", len(v))
    else:
        print("  value:", v)

print("\n================ SAMPLE DATA CHECK ================")
if "pred" in adata.layers:
    print("First row pred first 5 genes:", adata.layers["pred"][0, :5])

if adata.X is not None:
    print("First row X first 5 genes:", adata.X[0, :5])

print("\nInspection complete.")