import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ============================================================
# Paths
# ============================================================
ENV_REAL_CSV = "envelope_dose_finegrid.csv"
ENV_NULL_CSV = "envelope_2bin_null_finegrid.csv"
SHAM_CSV = "envelope_eval_sham.csv"
PCA_REAL_JSON = "eval_v2_dosed.json"
PCA_SHUF_JSON = "eval_v2_shuffled.json"
EVAL_SET_CSV = "eval_set_v2.csv"

OUT_PNG = "figure2_main.png"
OUT_PDF = "figure2_main.pdf"

# ============================================================
# Style
# ============================================================
BLUE = "#0072B2"
GRAY = "#7A7A7A"
ORANGE = "#D55E00"
GREEN = "#009E73"

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
})

ENV_ORDER = ["strong", "medium", "weak", "near_null"]
ENV_LABELS = {
    "strong": "Strong",
    "medium": "Medium",
    "weak": "Weak",
    "near_null": "Near-null",
}

EVAL_ORDER = ["strong_nsr", "mid_nsr", "rho_only", "near_null"]
EVAL_LABELS = {
    "strong_nsr": "Strong\nNSR",
    "mid_nsr": "Mid\nNSR",
    "rho_only": r"$\rho$-only",
    "near_null": "Near-null",
}


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
def envelope_gap(df):
    """Per-target high-dose minus low-dose gap in specific signal."""
    wide = df.pivot(
        index=["gene_target", "stratum"],
        columns="bin",
        values="specific_signal",
    ).reset_index()

    wide = wide.dropna(subset=[0, 1]).copy()
    wide["gap"] = wide[1] - wide[0]
    return wide


def load_pca_gap_table(json_path):
    """
    Average guide constructs within gene_target x dose_bin,
    then compute the high-dose minus low-dose score gap.
    """
    with open(json_path, "r") as f:
        obj = json.load(f)

    rows = [
        {
            "gene_target": rec["gene_target"],
            "dose_bin": rec["dose_bin"],
            "score": rec["CosineLogFCSpecific"],
        }
        for rec in obj["per_tuple"].values()
    ]

    df = pd.DataFrame(rows)
    gene_bin = (
        df.groupby(["gene_target", "dose_bin"], as_index=False)["score"]
          .mean()
    )

    wide = gene_bin.pivot(
        index="gene_target",
        columns="dose_bin",
        values="score",
    ).reset_index()

    wide = wide.dropna(subset=["bin0", "bin1"]).copy()
    wide["gap"] = wide["bin1"] - wide["bin0"]

    meta = pd.read_csv(EVAL_SET_CSV)
    meta["stratum"] = meta["stratum"].fillna("near_null")

    return wide.merge(
        meta[["gene_target", "stratum"]],
        on="gene_target",
        how="left",
    )


def load_figure2_data():
    env_real = pd.read_csv(ENV_REAL_CSV)
    env_null = pd.read_csv(ENV_NULL_CSV)
    sham = pd.read_csv(SHAM_CSV)

    pca_real = load_pca_gap_table(PCA_REAL_JSON)
    pca_shuffled = load_pca_gap_table(PCA_SHUF_JSON)

    return {
        "env_real": env_real,
        "env_null": env_null,
        "env_real_gap": envelope_gap(env_real),
        "env_null_gap": envelope_gap(env_null),
        "sham": sham,
        "pca_real": pca_real,
        "pca_shuffled": pca_shuffled,
    }


# ============================================================
# Shared grouped boxplot helper
# ============================================================
def grouped_boxplot(
    ax,
    observed,
    shuffled,
    order,
    label_map,
    ylabel,
    title,
    ylim,
):
    base = np.arange(len(order)) * 2.2
    pos_observed = base - 0.28
    pos_shuffled = base + 0.28

    observed_data = [
        observed.loc[observed["stratum"] == s, "gap"].to_numpy()
        for s in order
    ]
    shuffled_data = [
        shuffled.loc[shuffled["stratum"] == s, "gap"].to_numpy()
        for s in order
    ]

    observed_bp = ax.boxplot(
        observed_data,
        positions=pos_observed,
        widths=0.45,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.3},
    )
    for box in observed_bp["boxes"]:
        box.set_facecolor(BLUE)
        box.set_edgecolor(BLUE)
        box.set_alpha(0.45)
    for key in ["whiskers", "caps"]:
        for item in observed_bp[key]:
            item.set_color(BLUE)

    shuffled_bp = ax.boxplot(
        shuffled_data,
        positions=pos_shuffled,
        widths=0.45,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.3},
    )
    for box in shuffled_bp["boxes"]:
        box.set_facecolor(GRAY)
        box.set_edgecolor(GRAY)
        box.set_alpha(0.30)
    for key in ["whiskers", "caps"]:
        for item in shuffled_bp[key]:
            item.set_color(GRAY)

    # Deterministic jitter so repeated runs are identical.
    rng = np.random.default_rng(0)
    for i, values in enumerate(observed_data):
        x = rng.normal(pos_observed[i], 0.045, len(values))
        ax.scatter(x, values, s=8, alpha=0.18, color=BLUE, linewidths=0)

    for i, values in enumerate(shuffled_data):
        x = rng.normal(pos_shuffled[i], 0.045, len(values))
        ax.scatter(x, values, s=8, alpha=0.14, color=GRAY, linewidths=0)

    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(base)
    ax.set_xticklabels([label_map[s] for s in order])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(*ylim)

    ax.scatter([], [], s=45, color=BLUE, alpha=0.6, label="Observed dose")
    ax.scatter([], [], s=45, color=GRAY, alpha=0.6, label="Dose-shuffled control")
    ax.legend(frameon=False, loc="best")

    return observed_data, shuffled_data, base


# ============================================================
# Panel A: sham-axis artifact
# ============================================================
def plot_panel_a(ax, sham):
    rng = np.random.default_rng(1)

    for dose_bin in [0, 1, 2]:
        values = sham.loc[
            sham["bin"] == dose_bin,
            "ceiling_corrected",
        ].to_numpy()

        violin = ax.violinplot(
            [values],
            positions=[dose_bin],
            widths=0.75,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )

        for body in violin["bodies"]:
            body.set_facecolor(ORANGE)
            body.set_edgecolor(ORANGE)
            body.set_alpha(0.18)

        jitter_x = rng.normal(dose_bin, 0.06, len(values))
        ax.scatter(
            jitter_x,
            values,
            s=8,
            alpha=0.18,
            color=ORANGE,
            linewidths=0,
        )

        median = np.median(values)
        ax.plot(
            [dose_bin - 0.20, dose_bin + 0.20],
            [median, median],
            color=ORANGE,
            linewidth=2.0,
        )

    # Keep the headline number outside the data cloud.
    mean_by_bin = sham.groupby("bin")["ceiling_corrected"].mean()
    max_bin = int(mean_by_bin.idxmax())
    max_mean = float(mean_by_bin.max())
    ax.text(
        0.98, 0.96,
        f"largest sham-bin mean = {max_mean:.3f} (bin {max_bin})",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )

    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(-0.10, 0.86)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Bin 0", "Bin 1", "Bin 2"])
    ax.set_xlabel(r"Sham strata defined by $pct\_counts\_mt$")
    ax.set_ylabel("Apparent perturbation signal\n(ceiling-corrected)")
    ax.set_title("Naïve sham stratification produces apparent signal")
    add_panel_label(ax, "A")


# ============================================================
# Panel B: matched ground-truth medians
# ============================================================
def plot_panel_b(ax, env_real):
    x = np.array([0, 1])

    for stratum, color, label in [
        ("strong", BLUE, "Strong"),
        ("near_null", GREEN, "Near-null"),
    ]:
        medians = (
            env_real.loc[env_real["stratum"] == stratum]
            .groupby("bin")["specific_signal"]
            .median()
            .reindex([0, 1])
        )

        ax.plot(
            x,
            medians.to_numpy(),
            marker="o",
            markersize=6,
            linewidth=2.2,
            color=color,
            label=label,
        )

        for xi, yi in zip(x, medians.to_numpy()):
            ax.text(
                xi, yi + 0.010, f"{yi:.3f}",
                ha="center",
                va="bottom",
                color=color,
                fontsize=9,
            )

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(0, 0.27)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Low dose", "High dose"])
    ax.set_xlabel("Realized dose bin")
    ax.set_ylabel("Perturbation-specific\nreproducible signal")
    ax.set_title("Matched ground truth changes across realized dose")
    ax.legend(frameon=False, loc="upper left")
    add_panel_label(ax, "B")


# ============================================================
# Panel C: ground-truth dose gaps
# ============================================================
def plot_panel_c(ax, env_real_gap, env_null_gap):
    grouped_boxplot(
        ax,
        env_real_gap,
        env_null_gap,
        order=ENV_ORDER,
        label_map=ENV_LABELS,
        ylabel="Ground-truth dose gap\n(high − low specific signal)",
        title="Per-perturbation ground-truth dose gaps vs shuffled control",
        ylim=(-0.35, 0.43),
    )
    add_panel_label(ax, "C")


# ============================================================
# Panel D: PCA dose-score gaps
# ============================================================
def plot_panel_d(ax, pca_real, pca_shuffled):
    observed_data, shuffled_data, base = grouped_boxplot(
        ax,
        pca_real,
        pca_shuffled,
        order=EVAL_ORDER,
        label_map=EVAL_LABELS,
        ylabel="PCA score dose gap\n(high − low specific cosine)",
        title="PCA dose-score gaps vs shuffled control",
        ylim=(-1.70, 1.70),
    )

    # Small inset: medians on a readable vertical scale.
    inset = inset_axes(ax, width="45%", height="38%", loc="upper right", borderpad=1.2)

    observed_medians = [np.median(values) for values in observed_data]
    shuffled_medians = [np.median(values) for values in shuffled_data]

    inset.plot(base, observed_medians, "o-", color=BLUE, linewidth=1.2, markersize=4)
    inset.plot(base, shuffled_medians, "o--", color=GRAY, linewidth=1.1, markersize=4)
    inset.axhline(0, color="black", linewidth=0.7)
    inset.set_xticks([])
    inset.set_ylabel("Median gap", fontsize=8)
    inset.tick_params(axis="y", labelsize=7)
    inset.set_title("Medians", fontsize=8)
    inset.spines["top"].set_visible(False)
    inset.spines["right"].set_visible(False)

    add_panel_label(ax, "D")


# ============================================================
# Assemble figure
# ============================================================
def make_figure2():
    data = load_figure2_data()

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.8))

    plot_panel_a(axes[0, 0], data["sham"])
    plot_panel_b(axes[0, 1], data["env_real"])
    plot_panel_c(
        axes[1, 0],
        data["env_real_gap"],
        data["env_null_gap"],
    )
    plot_panel_d(
        axes[1, 1],
        data["pca_real"],
        data["pca_shuffled"],
    )

    fig.tight_layout(h_pad=2.0, w_pad=2.0)
    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    sham_means = data["sham"].groupby("bin")["ceiling_corrected"].mean()
    print("Sham mean by bin:", sham_means.round(4).to_dict())

    for stratum in ["strong", "near_null"]:
        medians = (
            data["env_real"]
            .loc[data["env_real"]["stratum"] == stratum]
            .groupby("bin")["specific_signal"]
            .median()
            .reindex([0, 1])
        )
        print(
            f"{stratum}: low={medians.iloc[0]:.4f}, "
            f"high={medians.iloc[1]:.4f}, "
            f"gap={medians.iloc[1] - medians.iloc[0]:.4f}"
        )

    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    make_figure2()
