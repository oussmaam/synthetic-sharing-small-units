import pandas as pd, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
INK, MUTE, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTE, "axes.linewidth": 0.6,
                     "xtick.color": MUTE, "ytick.color": MUTE, "axes.labelcolor": INK, "text.color": INK})
R = pd.read_csv("../results/results_v16a_natality_full.csv"); idx = ["gamma", "share", "supp", "rep"]
P = R.pivot_table(index=idx, columns="arm", values="ex_kl") * 1000
I = R[R.arm == "decision_inputs"].set_index(idx)
G = pd.DataFrame({"M3": P.share_LR_ct - P.C_target, "IW": P.share_LR_ct - P.IW_donor, "M0": P.share_LR_ct - P.local})
pred = (P.oracle_D_ct_n - P.oracle_h_star).xs(1.0, level="supp").groupby(level=["gamma", "share"]).mean()
gain = np.array([pred.loc[(g, s)] >= 2 for g, s, q, r in G.index])
Gg = G[gain]; m = Gg.groupby(level="supp").mean(); se = Gg.groupby(level=["supp", "rep"]).mean().groupby(level="supp").sem()
print(m.round(1)); print(I.groupby(level="supp").frac_out_range.describe().round(3))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[1.35, 1]))
X = [0, 1, 2]; QS = [1.0, 0.9, 0.8]
for arm, lab, col, mk in [("M3", "Targeted synthetic data", "#eb6834", "^"), ("IW", "Reweighting", "#2a78d6", "o"), ("M0", "Local score only", "#1baf7a", "s")]:
    y = m.loc[QS, arm].values; e = 1.96 * se.loc[QS, arm].values
    a1.fill_between(X, y - e, y + e, color=col, alpha=0.15, lw=0)
    a1.plot(X, y, color=col, lw=1.6, marker=mk, ms=5, mec="white", mew=0.5, label=lab, zorder=3)
a1.axhline(0, color=MUTE, lw=0.7, zorder=1); a1.grid(axis="y", color=GRID, lw=0.5, zorder=0)
a1.set_xticks(X); a1.set_xticklabels(["Full support", "No donor patient\nabove 90th BMI centile", "No donor patient\nabove 80th BMI centile"])
a1.set_ylabel("Gain over updated sharing, ×10$^{-3}$ nats"); a1.set_title("a  Transfer fails when the donor lacks support", fontsize=8, loc="left")
a1.legend(frameon=False, fontsize=7.5, loc="lower left")
rng = np.random.default_rng(2)
for k, q in enumerate(QS):
    v = I.xs(q, level="supp").frac_out_range.values * 100
    a2.scatter(k + rng.uniform(-0.25, 0.25, len(v)), v, s=7, color="#52514e", alpha=0.35, lw=0, zorder=3)
    a2.plot([k - 0.33, k + 0.33], [np.median(v)] * 2, color=INK, lw=1.4, zorder=4)
a2.grid(axis="y", color=GRID, lw=0.5, zorder=0); a2.set_xticks(X); a2.set_xticklabels(["Full", "90th", "80th"])
a2.set_xlabel("Donor support (BMI centile cut-off)"); a2.set_ylabel("Local patients outside the range\nof the received synthetic pool, %")
a2.set_title("b  Local patients outside the donor's range", fontsize=8, loc="left")
for ax in (a1, a2):
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.tick_params(length=2)
plt.tight_layout(); plt.savefig("fig6_support.pdf"); plt.savefig("fig6_support.png", dpi=300)
