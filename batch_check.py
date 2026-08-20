import anndata as ad
import numpy as np

real = ad.read_h5ad("/research/projects/rna_lab/perturb_benchmark/state/runs/HCT116/batch_000000/emb_celleval.h5ad")
pred = ad.read_h5ad("/research/projects/rna_lab/perturb_benchmark/state/runs/HCT116/batch_000000/infer_celleval.h5ad")


print("=== REAL ===")
print("shape:", real.shape)
print("X first row first 5:", np.asarray(real.X[0]).ravel()[:5])
print("obsm keys:", list(real.obsm.keys()))
print("layers keys:", list(real.layers.keys()))
print("uns keys:", list(real.uns.keys()))

print("\n=== PRED ===")
print("shape:", pred.shape)
print("X first row first 5:", np.asarray(pred.X[0]).ravel()[:5])
print("obsm keys:", list(pred.obsm.keys()))
print("layers keys:", list(pred.layers.keys()))
print("uns keys:", list(pred.uns.keys()))

print("\nX identical:", np.allclose(np.asarray(real.X), np.asarray(pred.X)))

for k in pred.obsm.keys():
    arr = pred.obsm[k]
    print(f"pred obsm[{k}] shape:", arr.shape)
    print(f"pred obsm[{k}] first row first 5:", np.asarray(arr[0]).ravel()[:5])

for k in pred.uns.keys():
    val = pred.uns[k]
    print(f"uns key: {k}, type: {type(val)}")