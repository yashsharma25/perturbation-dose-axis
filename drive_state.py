import os
import subprocess
import argparse
import pathlib
import anndata as ad

def require_finish(path: pathlib.Path):
    if not path.exists():
        raise FileNotFoundError(f"{path} missing")

    try:
        ad.read_h5ad(path, backed = 'r')

    except Exception as e:
        raise RuntimeError(f"{path} exists but is not valid h5ad") from e
    return

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xatlas' ,required=True)
    ap.add_argument('--start', type=int, required = True)
    ap.add_argument('--end', type=int, required = True)
    ap.add_argument('--ckpt' ,required=True)
    ap.add_argument('--output',required=True)

    args = ap.parse_args()

    output_dir = pathlib.Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    aligned_file = output_dir / "aligned_.h5ad"
    emb_file = output_dir / "emb.h5ad"
    infer_file = output_dir / "infer.h5ad"

    #create the directory
    subprocess.run(["python", "xatlas_align.py",
                    "--start", str(args.start),
                    "--end", str(args.end),
                    "--xatlas", args.xatlas,
                    "--ckpt", args.ckpt,
                    "--output",  str(aligned_file)],
                    check=True
                    )


    require_finish(aligned_file)

    if not emb_file.exists():
        with open(output_dir / "emb.log", "w") as f:
            subprocess.run(["state", "emb", "transform",
                            "--config", "/research/projects/rna_lab/perturb_benchmark/state/xatlas_emb_infer.yaml",  
                            "--model-folder", "/research/projects/rna_lab/perturb_benchmark/state/checkpoints/se/",
                            "--input", str(aligned_file),
                            "--output", str(emb_file)],
                            stdout=f, stderr=subprocess.STDOUT, check=True
                            ) 

    require_finish(emb_file)

    #call the function to make the embeddings compatible for infer. Cant we do that before generating the embeddings
    if not infer_file.exists():
        subprocess.run(["state", "tx", "infer",
            "--model-dir", "/research/projects/rna_lab/perturb_benchmark/state/checkpoints/rpe",
            "--checkpoint", "/research/projects/rna_lab/perturb_benchmark/state/checkpoints/rpe/best.ckpt",
            "--adata", str(emb_file),
            "--pert-col", "gene_target",
            "--embed-key", "X_state",
            "--output", str(infer_file)],
            check=True
        )

    require_finish(infer_file)

    post_infer_emb = emb_file.with_name(emb_file.stem + "_celleval.h5ad")
    post_infer_pred = infer_file.with_name(infer_file.stem + "_celleval.h5ad")

    with open(output_dir / "post_infer_fix.log", "w") as f:
        subprocess.run(["python", "fix_post_infer.py",
                        "--input_file",  str(emb_file),
                        "--pred_file", str(infer_file)],
                        stdout=f, stderr=subprocess.STDOUT, check=True
                        )
        
    require_finish(post_infer_emb)
    require_finish(post_infer_pred)

    with open(output_dir / "cell-eval.log", "w") as f:
        subprocess.run(["cell-eval", "run",
        "-ap", str(post_infer_pred),
        "-ar", str(post_infer_emb),
        "--outdir", str(output_dir),
        "--num-threads", "64",
        "--profile", "full"],
        stdout=f, stderr=subprocess.STDOUT, check=True)


    #delete the intermediate files
    os.remove(emb_file)
    os.remove(aligned_file)
    os.remove(emb_file)
    os.remove(infer_file)


if __name__ == "__main__":
    main()


# 1. Align state and xatlas genes
# 2. Pass it to SE - output will be an embeddings file that ST can use
# 3. Do ST infer with the embeddings file as input
# 4. Check where in the output contains predictions
# 5. Pass to cell-eval