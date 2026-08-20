import anndata as ad, math

x = "/research/projects/rna_lab/x-atlas/HCT116_filtered_dual.h5ad"
bs = 16384
adata = ad.read_h5ad(x, backed="r")
print("n_obs", adata.n_obs)
print("num_batches", math.ceil(adata.n_obs/bs))

# n_obs 3409169
# num_batches 209

# Now next step is that we create the driver file that calls all the scripts we want
# Then we fix the align part