import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODEL_RANKS = "../identity_retrieval_ranks.csv"
OBS_RANKS = "../observed_identity_positive_control_ranks.csv"

model_df = pd.read_csv(MODEL_RANKS)
obs_df = pd.read_csv(OBS_RANKS)

obs = obs_df["true_rank"].to_numpy(float)
c2s = model_df["C2S_real_token_rank"].to_numpy(float)
state = model_df["STATE_zero_shot_rank"].to_numpy(float)
N = len(obs)

obs_mrr = np.mean(1 / obs)
c2s_mrr = np.mean(1 / c2s)
state_mrr = np.mean(1 / state)
null_lo, null_hi = 0.0069427, 0.01377167

plt.rcParams.update({
    "font.size": 8.2,
    "axes.labelsize": 8.2,
    "xtick.labelsize": 7.7,
    "ytick.labelsize": 7.7,
    "legend.fontsize": 7.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig = plt.figure(figsize=(4.8, 4.25))
gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.15], hspace=0.52)

ax = fig.add_subplot(gs[0, 0])
ax.text(-0.08, 1.03, "A", transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
ks = np.arange(1, 101)
def curve(r):
    return np.array([(r <= k).mean() for k in ks])

co = curve(obs)
cc = curve(c2s)
cs = curve(state)
chance = ks / N

ax.plot(ks, co, lw=2.0, label="Observed split-half")
ax.plot(ks, cc, lw=1.5, label="C2S zero-shot")
ax.plot(ks, cs, lw=1.5, label="STATE zero-shot")
ax.plot(ks, chance, lw=1.1, ls="--", label="Chance")
ax.set_xlim(1, 100)
ax.set_ylim(0, 0.46)
ax.set_xlabel("Retrieval rank k")
ax.set_ylabel("Fraction retrieved by rank k")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.18, linewidth=0.5)
ax.legend(frameon=False, loc="upper left", ncol=2, columnspacing=0.8, handlelength=2.2)

ax.text(
    68, 0.205,
    f"At k=10\nObserved: {100*co[9]:.1f}%\nChance: {100*chance[9]:.2f}%",
    ha="center", va="center", fontsize=7.2,
    bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="0.75", linewidth=0.5)
)

ax = fig.add_subplot(gs[1, 0])
ax.text(-0.08, 1.04, "B", transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
ax.axvspan(null_lo, null_hi, alpha=0.18, linewidth=0)
ax.axvline((null_lo + null_hi) / 2, ls="--", lw=0.8)

labels = ["Observed split-half", "C2S zero-shot", "STATE zero-shot"]
yy = [2, 1, 0]
vals = [obs_mrr, c2s_mrr, state_mrr]
markers = ["o", "s", "^"]

for y, v, mk in zip(yy, vals, markers):
    ax.scatter([v], [y], s=32, marker=mk, zorder=3)
    ax.text(v + 0.004, y, f"{v:.3f}", va="center", fontsize=7.5)

ax.set_yticks(yy)
ax.set_yticklabels(labels)
ax.set_xlim(0, 0.18)
ax.set_ylim(-0.72, 2.55)
ax.set_xlabel("Mean reciprocal rank")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", alpha=0.18, linewidth=0.5)
ax.text(0.111, 2.14, r"$p_{perm}<0.001$", fontsize=7.5)

y_bracket = -0.28
ax.plot([null_lo, null_hi], [y_bracket, y_bracket], lw=0.8)
ax.plot([null_lo, null_lo], [y_bracket - 0.07, y_bracket + 0.07], lw=0.8)
ax.plot([null_hi, null_hi], [y_bracket - 0.07, y_bracket + 0.07], lw=0.8)


fig.subplots_adjust(left=0.20, right=0.985, top=0.98, bottom=0.12)
fig.savefig("figure3_identity_retrieval_final_v3.png", dpi=300, bbox_inches="tight")
fig.savefig("figure3_identity_retrieval_final_v3.pdf", bbox_inches="tight")
plt.close(fig)
