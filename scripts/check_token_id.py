import pandas as pd
import anndata as ad

gene_df = pd.read_parquet("/research/projects/rna_lab/x-atlas/gene_metadata.parquet")
adata = ad.read_h5ad("/research/projects/rna_lab/x-atlas/HCT116_filtered_dual.h5ad", backed="r")

print("len(var) =", len(adata.var))
print("max(token_id) =", gene_df.gene_token_id.max())

# gene_df = gene_df.sort_values("gene_token_id")
# print((gene_df["gene_name"].values == adata.var_names.values).all())

gene_df_sorted = gene_df.sort_values("gene_token_id")
names1 = gene_df_sorted["gene_name"].astype(str).values
names2 = adata.var_names.astype(str).values

for i, (a, b) in enumerate(zip(names1, names2)):
    if a != b:
        print("Mismatch at index:", i)
        print("gene_df_sorted:", a)
        print("adata.var_names:", b)
        break
else:
    print("No mismatches detected. All aligned.")


mismatch_indices = [i for i, (a, b) in enumerate(zip(names1, names2)) if a != b]
print("Total mismatches:", len(mismatch_indices))
print("Example indices:", mismatch_indices[:10])

for m in mismatch_indices:
    print(f"{names1[m]}, {names2[m]}")