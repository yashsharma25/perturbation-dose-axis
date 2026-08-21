import scanpy as sc
import pandas as pd

# Load the h5ad file
adata = sc.read_h5ad('/research/projects/rna_lab/perturb_benchmark/state/temp_aligned.h5ad', backed = 'r')

print("=== AnnData Shape ===")
print(f"n_obs (cells): {adata.n_obs}")
print(f"n_vars (genes): {adata.n_vars}")

print("\n=== obs columns ===")
print(adata.obs.columns.tolist())

print("\n=== Cell types ===")
if 'cell_type' in adata.obs.columns:
    print(adata.obs['cell_type'].value_counts())
else:
    print("No 'cell_type' column found")

print("\n=== Perturbations/genes ===")
if 'gene' in adata.obs.columns:
    print(adata.obs['gene'].value_counts().head(20))
else:
    print("No 'gene' column found")

print("\n=== Available layers ===")
print(list(adata.layers.keys()) if adata.layers else "No layers")

print("\n=== Available obsm keys ===")
print(list(adata.obsm.keys()) if adata.obsm else "No obsm")

print("\n=== var columns ===")
print(adata.var.columns.tolist())\

print("adata.var_names", adata.var_names)

# for g in adata.var_names:
#     print(g)


print("Shape of adata.obsm['X_state']", adata.obsm['X_state'].shape)

print("First 5 elements")
X = adata.obsm["X_state"]
print(X[:5, :5]) 


print("X shape:", adata.X.shape)
print(adata.X[:5, :5])
