import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
INK, MUTE, GRID = "#0b0b0b", "#52514e", "#e4e3df"; C = {"IW": "#2a78d6", "M3": "#eb6834"}
NAMES = {20: "Kansas", 31: "Nebraska", 27: "Minnesota", 53: "Washington"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTE, "axes.linewidth": 0.6,
                     "xtick.color": MUTE, "ytick.color": INK, "axes.labelcolor": INK, "text.color": INK})
fr = []
for f in ["../results/results_v17_brfss_full.csv", "../results/results_v18_brfss_full.csv"]:
    R = pd.read_csv(f); L = R.pivot_table(index=["donor", "receiver", "rep"], columns="arm", values="logloss") * 1000
    fr.append(L[["IW", "M3"]].rsub(L.M1, axis=0).groupby(level=["donor", "receiver"]).mean())
G = pd.concat(fr)
fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.7), sharex=True, sharey=True)
rng = np.random.default_rng(1)
for ax, d in zip(axes, [20, 31, 27, 53]):
    g = G.xs(d, level="donor")
    for k, (arm, mk) in enumerate([("IW", "o"), ("M3", "^")]):
        yj = k + rng.uniform(-0.22, 0.22, len(g))
        ax.scatter(g[arm], yj, s=16, marker=mk, color=C[arm], alpha=0.75, edgecolor="white", linewidth=0.4, zorder=3)
        ax.plot([g[arm].median()] * 2, [k - 0.34, k + 0.34], color=INK, lw=1.2, zorder=4)
        if 72 in g.index:
            j = list(g.index).index(72)
            ax.scatter([g.loc[72, arm]], [yj[j]], s=70, facecolor="none", edgecolor=INK, linewidth=0.9, zorder=5)
    ax.axvline(0, color=MUTE, lw=0.7, zorder=1); ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    ax.set_title(f"Donor: {NAMES[d]}", fontsize=8, loc="left"); ax.set_ylim(-0.7, 1.7)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Reweighting", "Targeted\nsynthetic data"])
    for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
axes[0].scatter([], [], s=70, facecolor="none", edgecolor=INK, linewidth=0.9, label="Puerto Rico"); axes[0].plot([], [], color=INK, lw=1.2, label="median of receivers")
fig.legend(loc="lower right", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.995, 0.105), handletextpad=0.4, columnspacing=1.2)
fig.supxlabel("Gain in test log-loss over updated sharing, ×10$^{-3}$ nats; right of zero is better", fontsize=8, y=0.015, x=0.38)
plt.tight_layout(rect=(0, 0.12, 1, 1)); plt.savefig("fig7_states.pdf"); plt.savefig("fig7_states.png", dpi=300)
print(G.groupby(level="donor").agg(["median", "min", "max"]).round(1)); print((G > 0).groupby(level="donor").mean().round(2))
