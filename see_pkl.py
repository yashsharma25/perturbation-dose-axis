import pickle

with open("/research/projects/rna_lab/perturb_benchmark/state/checkpoints/rpe/var_dims.pkl", "rb") as f:
    data = pickle.load(f)

for d in data.items():
    print(d)