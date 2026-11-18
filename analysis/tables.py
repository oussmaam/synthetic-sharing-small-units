"""Reproduce the numbers in the tables of the main text and the Supplement from the result files.

Usage:  python tables.py            (reads ../results/)
Gains are differences in log-loss from updated sharing (closed testing), x 1000; positive favours the strategy.
Intervals are 95% t intervals over the repetition means.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
rd = lambda f: pd.read_csv(os.path.join(RES, f))


def ci(x):
    x = np.asarray(x, float)
    h = stats.t.ppf(0.975, len(x) - 1) * stats.sem(x)
    return f"{x.mean():6.1f} ({x.mean() - h:6.1f}, {x.mean() + h:6.1f})"


def by_rep(s):
    """Average over scenarios within each repetition, then summarise across repetitions."""
    return ci(s.groupby(level="rep").mean())


def gain_mask(P, cell):
    """Scenarios with room for gain: updated sharing at n = 200 is at least 2 units worse than the true risk."""
    pred = (P["oracle_D_ct_n"] - P["oracle_h_star"]).groupby(level=cell).transform("mean")
    return pred >= 2


def header(t):
    print("\n" + "=" * 100 + "\n" + t + "\n" + "=" * 100)


# ---------------------------------------------------------------- Table 3: semi-synthetic runs
header("Table 3. Experiment 1, semi-synthetic runs (v15a; COMMUTE from v11a)")
SIM = {"local": "Local score", "IW_donor": "Reweighting", "B_untargeted": "Untargeted synthetic",
       "C_target": "Targeted synthetic", "BR_OS": "Brown: oversampling", "BR_SDG": "Brown: local generator",
       "BR_cond": "Brown: conditional", "BR_cond_big": "Brown: conditional, 10,000", "R1": "Decision rule R1"}
for ds in ["natality", "cdc_diabetes"]:
    d = rd(f"results_v15a_{ds}_full.csv"); cell = ["gamma", "share"]
    P = d.pivot_table(index=cell + ["rep"], columns="arm", values="ex_kl") * 1000
    I = d[d.arm == "decision_inputs"].set_index(cell + ["rep"])[["d_mean", "ess_don"]].reindex(P.index)
    P["R1"] = np.where(I.d_mean < 0, np.where(I.ess_don < 0.40, P.C_target, P.IW_donor), P.share_LR_ct)
    g = gain_mask(P, cell)
    print(f"\n{ds}: {g.groupby(level=cell).first().sum()} of {P.groupby(level=cell).ngroups} scenarios with room for gain")
    for a, lab in SIM.items():
        G = P.share_LR_ct - P[a]
        print(f"  {lab:28s} gain {by_rep(G[g])}   other {G[~g].mean():6.1f}")
    t = rd(f"results_v11a_{ds}_full.csv")
    Q = t.pivot_table(index=cell + ["rep"], columns="arm", values="ex_kl") * 1000; gq = gain_mask(Q, cell)
    print(f"  {'COMMUTE-style transfer (v11a)':28s} gain {by_rep((Q.share_LR_ct - Q.transfer_LR)[gq])}"
          f"   targeted in same scenarios {by_rep((Q.share_LR_ct - Q.C_target)[gq])}")

# ---------------------------------------------------------------- Registry runs
header("Supplement Table S4. Registry runs (v15b)")
REG = {"local_score": "Local score", "IW_donor": "Reweighting", "B_untargeted": "Untargeted synthetic",
       "C_target": "Targeted synthetic", "Platt_local": "Brown: Platt on local", "BR_cond_big": "Brown: conditional, 10,000"}
for ds in ["natality", "cdc_diabetes"]:
    d = rd(f"results_v15b_{ds}_full.csv"); cell = ["shift", "share"]
    P = d.pivot_table(index=cell + ["rep"], columns="arm", values="logloss_y") * 1000
    print(f"\n{ds}")
    for a, lab in REG.items():
        print(f"  {lab:28s} {by_rep(P.share_score_ct - P[a])}")
    print("  AUC", d.groupby("arm").auc.mean().round(3)[["share_score_ct", "IW_donor", "C_target"]].to_dict())
    print("  median calibration slope", d.groupby("arm").cal_slope.median().round(2)[["share_score_ct", "IW_donor", "C_target"]].to_dict())

# ---------------------------------------------------------------- Soft labels etc.
header("Supplement Table S5. Soft labels, real records, donor-side distillation (v16a, full support)")
for ds in ["natality", "cdc_diabetes"]:
    d = rd(f"results_v16a_{ds}_full.csv"); d = d[d.supp == 1.0]; cell = ["gamma", "share"]
    P = d.pivot_table(index=cell + ["rep"], columns="arm", values="ex_kl") * 1000; g = gain_mask(P, cell)
    print(f"\n{ds}")
    for a in ["C_target", "M3_hard", "M3_realX", "IW_KD", "IW_donor"]:
        print(f"  {a:10s} {by_rep((P.share_LR_ct - P[a])[g])}")
    print(f"  soft - hard {by_rep((P.M3_hard - P.C_target)[g])}")

# ---------------------------------------------------------------- Weight, simulation
header("Supplement Table S6. Weight of the synthetic records, birth data (v12a/b)")
d = rd("results_v12a_natality_full.csv"); cell = ["gamma", "share"]
P = d.pivot_table(index=cell + ["rep"], columns="arm", values="ex_kl") * 1000; g = gain_mask(P, cell)
e = rd("results_v12b_natality_full.csv")
Q = e.pivot_table(index=["shift", "share", "rep"], columns="arm", values="logloss_y") * 1000
for a in ["C_w0.25", "C_w1", "C_w5", "C_w10", "C_w50", "C_cv", "IW_donor"]:
    G = P.share_LR_ct - P[a]
    print(f"  {a:9s} semi gain {by_rep(G[g])}  semi other {by_rep(G[~g])}  registry {by_rep(Q.share_score_ct - Q[a])}")

# ---------------------------------------------------------------- Real sites
header("Supplement Table S3. Real sites (v17: Kansas, Nebraska; v18: Minnesota, Washington)")
for f in ["results_v17_brfss_full.csv", "results_v18_brfss_full.csv"]:
    d = rd(f); L = d.pivot_table(index=["donor", "receiver", "rep"], columns="arm", values="logloss") * 1000
    for don, Ld in L.groupby(level="donor"):
        arms = ["M0", "IW", "M3", "R1"] + (["R2"] if "R2" in Ld else [])
        print(f"  donor {don}: " + " | ".join(f"{a} {by_rep(Ld.M1 - Ld[a])}" for a in arms))
        pr = Ld[Ld.index.get_level_values("receiver") == 72]
        print(f"     Puerto Rico: " + ", ".join(f"{a} {(pr.M1 - pr[a]).mean():.1f}" for a in arms))
    if "R2" in L:
        print(f"  R1 - R2 {by_rep(L.R1 - L.R2)}   always IW - R2 {by_rep(L.R2 - L.IW)}")
    out = L[L.index.get_level_values("receiver") != 72]
    print("  share of receivers where IW beats M1:", round(((out.M1 - out.IW).groupby(level=["donor", "receiver"]).mean() > 0).mean(), 2))

# ---------------------------------------------------------------- Weight, real sites
header("Supplement Table S7. Weight of the synthetic records on real sites")
d = rd("results_wsens_brfss_full.csv")
L = d.pivot_table(index=["donor", "receiver", "rep"], columns="arm", values="logloss") * 1000
arms = ["IW", "M3_w1", "M3_w5", "M3_w10", "M3_w20", "M3"]
out = L[L.index.get_level_values("receiver") != 72]
for don, q in out.groupby(level="donor"):
    print(f"  donor {don}: " + ", ".join(f"{a} {(q.M1 - q[a]).mean():.1f}" for a in arms))
print("  median calibration slope:", d[d.receiver != 72].groupby("arm").cal_slope.median().round(2)[["M1"] + arms].to_dict())
pr = L[L.index.get_level_values("receiver") == 72]
print("  Puerto Rico draws, M3_w1:", (pr.M1 - pr.M3_w1).round(1).tolist(), " IW:", (pr.M1 - pr.IW).round(1).tolist())

# ---------------------------------------------------------------- Variants
header("Supplement Tables S8-S9. Two further kinds of difference")
for v in ["age_nl", "concept"]:
    d = rd(f"results_var_{v}_cdc_diabetes.csv"); cell = ["gamma", "share"]
    P = d.pivot_table(index=cell + ["rep"], columns="arm", values="ex_kl") * 1000
    room = (P.share_LR_ct - P.oracle_h_star).groupby(level=cell).mean()
    print(f"\n{v}: room over updated sharing {room.min():.1f} to {room.max():.1f}")
    arms = ["local", "IW_donor", "B_untargeted", "C_target", "transfer_LR"]
    for a in arms:
        print(f"  {a:13s} {by_rep(P.share_LR_ct - P[a])}")
    print(P[arms].rsub(P.share_LR_ct, axis=0).groupby(level=cell).mean().round(1).to_string())
