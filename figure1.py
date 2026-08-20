import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# ============================================================
# Paths
# ============================================================
RAW_DOSE_CSV = "dose_validation_hct116_min.csv"
RESID_DOSE_CSV = "dose_validation_hct116_min_resid.csv"

OUT_PNG = "figure1_main.png"
OUT_PDF = "figure1_main.pdf"

# ============================================================
# Style
# ============================================================
BLUE = "#0072B2"
GRAY = "#A0A0A0"
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
    """One Spearman rho per target across the four dose bins."""
    rows = []

    for gene, sub in df.groupby("gene_target"):
        sub = sub.sort_values("dose_bin")

        # Spearman is undefined if either variable is constant.
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
    raw = pd.read_csv(RAW_DOSE_CSV)
    resid = pd.read_csv(RESID_DOSE_CSV)

    rho_raw = compute_target_spearman(raw)
    rho_resid = compute_target_spearman(resid)

    rho_pair = rho_raw.merge(
        rho_resid,
        on="gene_target",
        suffixes=("_raw", "_resid"),
    )

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

    return {
        "rho_pair": rho_pair,
        "median_raw": rho_raw["rho"].median(),
        "median_resid": rho_resid["rho"].median(),
        "fraction_negative": (rho_resid["rho"] < 0).mean(),
        "knockdown_by_bin": knockdown_by_bin,
        "knockdown_medians": medians,
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
# Panel B: paired correlations
# ============================================================
def plot_panel_b(ax, rho_pair, median_raw, median_resid):
    for _, row in rho_pair.iterrows():
        ax.plot(
            [0, 1],
            [row["rho_raw"], row["rho_resid"]],
            color="#D0D0D0",
            linewidth=0.8,
            alpha=0.75,
        )

    ax.scatter(
        np.zeros(len(rho_pair)),
        rho_pair["rho_raw"],
        s=22,
        color=GRAY,
        alpha=0.85,
        zorder=3,
    )
    ax.scatter(
        np.ones(len(rho_pair)),
        rho_pair["rho_resid"],
        s=22,
        color=BLUE,
        alpha=0.85,
        zorder=3,
    )

    # Larger points show group medians.
    ax.scatter(
        [0, 1],
        [median_raw, median_resid],
        s=78,
        color=[GRAY, BLUE],
        edgecolor="black",
        linewidth=0.7,
        zorder=4,
    )

    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xlim(-0.32, 1.32)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([
        f"Raw\nmedian = {median_raw:.3f}",
        f"Depth-residualized\nmedian = {median_resid:.3f}",
    ])
    ax.set_ylabel(r"Spearman $\rho$(dose, target expression)")
    ax.set_title("Depth correction strengthens dose-suppression coupling", pad=10)
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
            i, 104, f"{median:.0f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(["Q1\nlowest", "Q2", "Q3", "Q4\nhighest"])
    ax.set_ylim(-75, 112)
    ax.set_yticks([-75, -50, 0, 25, 50, 75, 100])
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
        gridspec_kw={"width_ratios": [1.05, 1.0, 1.05]},
    )

    plot_panel_a(axes[0])
    plot_panel_b(
        axes[1],
        data["rho_pair"],
        data["median_raw"],
        data["median_resid"],
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
        f"Defined target-level rho: n={len(data['rho_pair'])}; "
        f"raw median={data['median_raw']:.3f}; "
        f"residualized median={data['median_resid']:.3f}; "
        f"negative after residualization={100 * data['fraction_negative']:.1f}%"
    )
    print("Median knockdown by quartile:", np.round(data["knockdown_medians"], 1).tolist())


if __name__ == "__main__":
    make_figure1()
