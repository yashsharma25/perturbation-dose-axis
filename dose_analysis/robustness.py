import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

rng = np.random.default_rng(42)

def summarize(path, metric, label):
    df = pd.read_csv(path)
    s = df[df["stratum"] == "strong"]

    w = s.pivot(index="gene_target", columns="bin", values=metric).dropna()
    gaps = (w[1] - w[0]).to_numpy()

    boot = np.array([
        np.median(rng.choice(gaps, len(gaps), replace=True))
        for _ in range(20000)
    ])

    print("\n", label)
    print("n =", len(gaps))
    print("low =", w[0].median())
    print("high =", w[1].median())
    print("median gap =", np.median(gaps))
    print("95% CI =", np.quantile(boot, [.025, .975]))
    print("fraction positive =", np.mean(gaps > 0))
    print("Wilcoxon two-sided =", wilcoxon(gaps).pvalue)

summarize(
    "envelope_dose_finegrid.csv",
    "specific_signal",
    "10x8 matched, perturbation-specific"
)

summarize(
    "envelope_dose_finegrid.csv",
    "total_reproducibility",
    "10x8 matched, total reproducibility"
)

summarize(
    "envelope_dose.csv",
    "specific_signal",
    "earlier/default-grid matched, perturbation-specific"
)
