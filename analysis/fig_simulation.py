import pandas as pd, numpy as np, os, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
INK, MUTE, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ARMS = [("C_target", "Targeted synthetic data", "#eb6834", "^"), ("IW_donor", "Reweighting", "#2a78d6", "o"),
        ("BR_cond", "Augmentation + recalibration (Brown et al.)", "#1baf7a", "s")]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTE, "axes.linewidth": 0.6,
                     "xtick.color": MUTE, "ytick.color": MUTE, "axes.labelcolor": INK, "text.color": INK})
RES = "../results/"
SETS = [("US Natality 2023 (gestational hypertension)", RES + "results_v15a_natality_full.csv"),
        ("CDC Diabetes Health Indicators", RES + "results_v15a_cdc_diabetes_full.csv")]
SETS = [s for s in SETS if os.path.exists(s[1])]
fig, axes = plt.subplots(len(SETS), 3, figsize=(7.2, 2.45 * len(SETS)), sharex=True, sharey="row", squeeze=False)
for r, (name, f) in enumerate(SETS):
    R = pd.read_csv(f); P = R.pivot_table(index=["gamma", "share", "rep"], columns="arm", values="ex_kl") * 1000
    G = P[[a[0] for a in ARMS]].rsub(P["share_LR_ct"], axis=0)
    m = G.groupby(level=["gamma", "share"]).mean(); se = G.groupby(level=["gamma", "share"]).sem()
    print(name, R.rep.nunique(), "reps"); print(m.round(1).to_string())
    for c, g in enumerate([0.0, 0.5, 1.0]):
        ax = axes[r, c]; mm, ss = m.xs(g, level="gamma"), se.xs(g, level="gamma")
        ax.axhline(0, color=MUTE, lw=0.7, zorder=1); ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
        for a, lab, col, mk in ARMS:
            ax.fill_between(mm.index, mm[a] - 1.96 * ss[a], mm[a] + 1.96 * ss[a], color=col, alpha=0.15, lw=0)
            ax.plot(mm.index, mm[a], color=col, lw=1.6, marker=mk, ms=4.5, mec="white", mew=0.5, label=lab if (r, c) == (0, 0) else None, zorder=3)
        for s in ["top", "right"]: ax.spines[s].set_visible(False)
        ax.tick_params(length=2); ax.set_xticks([0.5, 0.7, 0.9])
        if r == 0: ax.set_title(["Logistic truth (γ = 0)", "Moderate non-linearity (γ = 0.5)", "Strong non-linearity (γ = 1)"][c], fontsize=8, loc="left")
        if c == 0: ax.set_ylabel(name.split(" (")[0] + "\ngain, ×10$^{-3}$ nats", fontsize=7.5)
        if r == len(SETS) - 1: ax.set_xlabel("Receiver share above the BMI median\n(0.5 = no shift)", fontsize=7.5)
fig.legend(loc="upper center", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 1.0), handlelength=2.2, columnspacing=1.6)
plt.tight_layout(rect=(0, 0, 1, 0.94)); plt.savefig("fig4_simulation.pdf"); plt.savefig("fig4_simulation.png", dpi=300)
