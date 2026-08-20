# import h5py

# path = "/research/projects/rna_lab/perturb_benchmark/state/xatlas_state_predictions.npy"

# with h5py.File(path, "r") as f:
#     def walk(name, obj):
#         if isinstance(obj, h5py.Dataset):
#             print(name, obj.shape, obj.dtype)
#         else:
#             print(name)
#     f.visititems(walk)


import h5py
import numpy as np

path = "/research/projects/rna_lab/perturb_benchmark/state/xatlas_state_predictions.npy"

with h5py.File(path, "r") as f:
    # replace this with the dataset name you see in Step 1
    dset = f["X"]
    print("shape:", dset.shape, "dtype:", dset.dtype)
    print(dset[:5, :5])
