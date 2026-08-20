import anndata as ad
import pandas as pd
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_file' ,required=True)

    args = ap.parse_args()

    #IN = "/research/projects/rna_lab/perturb_benchmark/state/state_embeddings.h5ad"
    #OUT = "/research/projects/rna_lab/perturb_benchmark/state/state_embeddings_fixpert.h5ad"

    IN = Path(args.input_file)
    OUT = IN.with_name(IN.stem + "_fixed.h5ad")

    adata = ad.read_h5ad(IN)

    # Create a canonical pert column `gene` (what you'll pass as --pert-col)
    src = adata.obs["gene_target"].astype(str)

    # Normalize: lowercase, strip whitespace
    norm = src.str.strip().str.lower()

    # If anything contains "non-targeting" (case-insensitive), force exact token
    norm = norm.where(~norm.str.contains("non-targeting", regex=False), "non-targeting")

    adata.obs["gene"] = norm

    adata.obs['gem_group'] = adata.obs['sample'].astype(str)

    adata.write_h5ad(OUT)
    print("Wrote:", OUT)
    print("Has exact control?", ("non-targeting" in set(adata.obs["gene"])))

if __name__ == "__main__":
    main()