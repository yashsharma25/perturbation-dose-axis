import anndata
adata = anndata.read_h5ad("/research/projects/rna_lab/x-atlas/HCT116_filtered_dual.h5ad", backed="r")
import re
guide_like = [v for v in adata.var_names if re.search(r'P1P2|non-targeting|sgRNA|guide', str(v), re.I)]
print(len(adata.var_names), guide_like[:10])
print(adata.var.columns.tolist())