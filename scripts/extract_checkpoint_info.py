"""
Extract metadata from checkpoint to help with inference.
"""
import torch
import pickle

ckpt_path = "checkpoints/rpe/final.ckpt"
data_module_path = "checkpoints/rpe/data_module.torch"

print("Loading checkpoint...")
ckpt = torch.load(ckpt_path, map_location="cpu")

print("\n=== Checkpoint Keys ===")
print(ckpt.keys())

print("\n=== Hyperparameters ===")
if "hyper_parameters" in ckpt:
    hparams = ckpt["hyper_parameters"]
    for key in sorted(hparams.keys()):
        print(f"{key}: {hparams[key]}")

print("\n=== Gene names (first 10) ===")
if "hyper_parameters" in ckpt and "gene_names" in ckpt["hyper_parameters"]:
    gene_names = ckpt["hyper_parameters"]["gene_names"]
    print(f"Total genes: {len(gene_names)}")
    print(gene_names[:10])

print("\n=== Loading data_module.torch ===")
try:
    data_module = torch.load(data_module_path, map_location="cpu")
    print(f"Type: {type(data_module)}")
    if hasattr(data_module, '__dict__'):
        print("\nData module attributes:")
        for attr in dir(data_module):
            if not attr.startswith('_'):
                print(f"  {attr}")
except Exception as e:
    print(f"Error: {e}")
