import scanpy as sc

name_to_col = {}
collisions = {}

hct116_adata = sc.read_h5ad('/research/projects/rna_lab/x-atlas/HCT116_filtered_dual.h5ad', backed = 'r')

def normalize(name: str) -> str:
    """Normalize gene names by removing '-1', '-2', '-3' suffixes."""

    # Split only once at the end
    if name.count('-') > 0:
        base, suffix = name.rsplit('-', 1)
        if suffix.isdigit():
            return base
    return name


print("Length of varnams=", len(hct116_adata.var_names))

for col, gene_name in enumerate(hct116_adata.var_names):
    k = normalize(gene_name)
    if k in name_to_col:
        collisions.setdefault(k, []).append(col)
    else:
        name_to_col[k] = col

print("Normalized collisions:", len(collisions))
