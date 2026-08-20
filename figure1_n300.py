import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# ============================================================
# Paths
# ============================================================
RESID_DOSE_CSV = "dose_validation_hct116_min_resid_n300.csv"
OUT_PNG = "figure1_main.png"
OUT_PDF = "figure1_main.pdf"

# ============================================================
# Style
# ============================================================
BLUE = "#0072B2"
POINT_BLUE = "#4C9ED9"
DOSE_COLORS = ["#DEEBF7", "#9ECAE1", "#4292C6", "#08519C"]

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
})


def add_panel_label(ax, label):
    ax.text(
        -0.12, 1.07, label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
    )


# ============================================================
# Data preparation
# ============================================================
def compute_target_spearman(df):
    """One Spearman rho per target across four residualized-dose quartiles."""
    rows = []

    for gene, sub in df.groupby("gene_target"):
        sub = sub.sort_values("dose_bin")

        # Require all four quartiles and a nonconstant target-expression vector.
        if sub["dose_bin"].nunique() != 4:
            continue
        if sub["mean_dose"].nunique() < 2:
            continue
        if sub["relative_target_expression"].nunique() < 2:
            continue

        rho, _ = spearmanr(
            sub["mean_dose"],
            sub["relative_target_expression"],
        )

        if np.isfinite(rho):
            rows.append({"gene_target": gene, "rho": rho})

    return pd.DataFrame(rows)


def load_figure1_data():
    resid = pd.read_csv(RESID_DOSE_CSV)
    rho = compute_target_spearman(resid)

    knockdown_by_bin = [
        100 * (
            1
            - resid.loc[
                resid["dose_bin"] == dose_bin,
                "relative_target_expression",
            ].to_numpy()
        )
        for dose_bin in range(4)
    ]

    medians = [np.median(values) for values in knockdown_by_bin]
    n_per_bin = [len(values) for values in knockdown_by_bin]

    return {
        "rho": rho,
        "median_rho": rho["rho"].median(),
        "n_negative": int((rho["rho"] < 0).sum()),
        "fraction_negative": float((rho["rho"] < 0).mean()),
        "knockdown_by_bin": knockdown_by_bin,
        "knockdown_medians": medians,
        "n_per_bin": n_per_bin,
    }


# ============================================================
# Panel A: schematic
# ============================================================
def plot_panel_a(ax):
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(
        0.50, 0.88,
        "Same nominal perturbation",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.42",
            facecolor="white",
            edgecolor="#555555",
            linewidth=1.0,
        ),
    )

    ax.annotate(
        "", xy=(0.27, 0.68), xytext=(0.44, 0.82),
        arrowprops=dict(arrowstyle="->", color="#666666", linewidth=1.3),
    )
    ax.annotate(
        "", xy=(0.73, 0.68), xytext=(0.56, 0.82),
        arrowprops=dict(arrowstyle="->", color="#666666", linewidth=1.3),
    )

    ax.text(
        0.25, 0.62,
        "Lower realized dose",
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.38",
            facecolor=DOSE_COLORS[0],
            edgecolor=BLUE,
            linewidth=1.0,
        ),
    )

    ax.text(
        0.75, 0.62,
        "Higher realized dose",
        ha="center",
        va="center",
        color="white",
        bbox=dict(
            boxstyle="round,pad=0.38",
            facecolor=DOSE_COLORS[3],
            edgecolor=BLUE,
            linewidth=1.0,
        ),
    )

    ax.annotate(
        "", xy=(0.25, 0.40), xytext=(0.25, 0.55),
        arrowprops=dict(arrowstyle="->", color="#666666", linewidth=1.3),
    )
    ax.annotate(
        "", xy=(0.75, 0.40), xytext=(0.75, 0.55),
        arrowprops=dict(arrowstyle="->", color="#666666", linewidth=1.3),
    )

    ax.text(0.25, 0.34, "Weaker target\nknockdown", ha="center", va="center")
    ax.text(0.75, 0.34, "Stronger target\nknockdown", ha="center", va="center")

    ax.annotate(
        "", xy=(0.42, 0.18), xytext=(0.28, 0.27),
        arrowprops=dict(arrowstyle="->", color="#888888", linewidth=1.0),
    )
    ax.annotate(
        "", xy=(0.58, 0.18), xytext=(0.72, 0.27),
        arrowprops=dict(arrowstyle="->", color="#888888", linewidth=1.0),
    )

    ax.text(
        0.50, 0.12,
        "Different dose-conditioned\ntranscriptional responses",
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.36",
            facecolor="#F7F7F7",
            edgecolor="#777777",
            linewidth=0.9,
        ),
    )

    ax.set_title("Realized dose varies within a nominal perturbation", pad=10)
    add_panel_label(ax, "A")


# ============================================================
# Panel B: distribution of target-level correlations
# ============================================================
def plot_panel_b(ax, rho, median_rho):
    values = rho["rho"].to_numpy()

    bp = ax.boxplot(
        [values],
        positions=[0],
        widths=0.36,
        patch_artist=True,
        showfliers=False,
        whis=1.5,
        medianprops=dict(color="black", linewidth=1.7),
        whiskerprops=dict(color="#666666", linewidth=1.0),
        capprops=dict(color="#666666", linewidth=1.0),
    )
    bp["boxes"][0].set_facecolor("#DCEEF8")
    bp["boxes"][0].set_edgecolor(BLUE)
    bp["boxes"][0].set_linewidth(1.1)

    rng = np.random.default_rng(0)
    x = rng.normal(loc=0.0, scale=0.055, size=len(values))
    ax.scatter(x, values, s=13, color=POINT_BLUE, alpha=0.50, linewidths=0, zorder=3)

    ax.axhline(0, color="#555555", linewidth=0.9, linestyle="--")
    ax.set_xlim(-0.38, 0.38)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks([0])
    ax.set_xticklabels([f"Residualized dose\nmedian $\\rho$ = {median_rho:.3f}"])
    ax.set_ylabel(r"Spearman $\rho$(dose, target expression)")
    ax.set_title("Dose is associated with target suppression", pad=10)
    add_panel_label(ax, "B")


# ============================================================
# Panel C: boxplots by dose quartile
# ============================================================
def plot_panel_c(ax, knockdown_by_bin, medians):
    bp = ax.boxplot(
        knockdown_by_bin,
        positions=np.arange(4),
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        whis=1.5,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(color="#666666", linewidth=1.0),
        capprops=dict(color="#666666", linewidth=1.0),
    )

    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(DOSE_COLORS[i])
        box.set_edgecolor(BLUE)
        box.set_linewidth(1.0)

    for i, median in enumerate(medians):
        ax.text(
            i, 104, f"{median:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(["Q1\nlowest", "Q2", "Q3", "Q4\nhighest"])
    ax.set_ylim(-55, 112)
    ax.set_yticks([-50, -25, 0, 25, 50, 75, 100])
    ax.set_xlabel("Depth-residualized sgRNA dose quartile")
    ax.set_ylabel("Target knockdown relative to control (%)")
    ax.set_title("Higher realized dose gives stronger target knockdown", pad=10)
    add_panel_label(ax, "C")


# ============================================================
# Assemble figure
# ============================================================
def make_figure1():
    data = load_figure1_data()

    fig, axes = plt.subplots(
        1, 3,
        figsize=(14.4, 4.4),
        gridspec_kw={"width_ratios": [1.05, 0.85, 1.1]},
    )

    plot_panel_a(axes[0])
    plot_panel_b(
        axes[1],
        data["rho"],
        data["median_rho"],
    )
    plot_panel_c(
        axes[2],
        data["knockdown_by_bin"],
        data["knockdown_medians"],
    )

    fig.tight_layout(w_pad=2.2)
    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")
    print(
        f"Defined target-level rho: n={len(data['rho'])}; "
        f"median={data['median_rho']:.3f}; "
        f"negative={data['n_negative']}/{len(data['rho'])} "
        f"({100 * data['fraction_negative']:.1f}%)"
    )
    print("Perturbations per quartile:", data["n_per_bin"])
    print("Median knockdown by quartile:", np.round(data["knockdown_medians"], 1).tolist())


if __name__ == "__main__":
    make_figure1()
