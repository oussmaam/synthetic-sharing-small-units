display = lambda x: print(x.to_string() if hasattr(x, 'to_string') else x)

DATA_PATH = "brfss2015_subset.csv.gz"
QUICK = True                      # True: 2 repetitions, 6 receivers. False: the full design.
N_LOCAL, N_DONOR, N_SYN, POOL_MULT = 200, 8000, 10000, 3
MIN_RECEIVER = 1500
N_REPS = 2 if QUICK else 10
SEED0 = 2026
W_CLIP_Q, CART_MIN_LEAF, ALPHA_CT = 0.99, 5, 0.05
ESS_T, SUPPORT_T = 0.40, 0.10     # frozen rule threshold; support step threshold (fixed before running)
SCORE = ["Age", "BMI", "HighBP"]
OUT_CSV = f"results_v18_brfss_{'quick' if QUICK else 'full'}.csv"

import sys
if len(sys.argv) > 1 and sys.argv[1] == 'full': QUICK = False; N_REPS = 10; OUT_CSV = 'results_v18_brfss_full.csv'

import os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

STATE_NAMES = {1:"Alabama",2:"Alaska",4:"Arizona",5:"Arkansas",6:"California",8:"Colorado",9:"Connecticut",10:"Delaware",
 11:"District of Columbia",12:"Florida",13:"Georgia",15:"Hawaii",16:"Idaho",17:"Illinois",18:"Indiana",19:"Iowa",20:"Kansas",
 21:"Kentucky",22:"Louisiana",23:"Maine",24:"Maryland",25:"Massachusetts",26:"Michigan",27:"Minnesota",28:"Mississippi",
 29:"Missouri",30:"Montana",31:"Nebraska",32:"Nevada",33:"New Hampshire",34:"New Jersey",35:"New Mexico",36:"New York",
 37:"North Carolina",38:"North Dakota",39:"Ohio",40:"Oklahoma",41:"Oregon",42:"Pennsylvania",44:"Rhode Island",
 45:"South Carolina",46:"South Dakota",47:"Tennessee",48:"Texas",49:"Utah",50:"Vermont",51:"Virginia",53:"Washington",
 54:"West Virginia",55:"Wisconsin",56:"Wyoming",66:"Guam",72:"Puerto Rico"}

def load_brfss(path):
    # same derivation as the CDC Diabetes Health Indicators dataset (BRFSS 2015 codebook)
    d = pd.read_csv(path)
    yn = lambda s: s.map({1: 1, 2: 0})                                   # 1 = yes, 2 = no; 7/9 -> missing
    out = pd.DataFrame({
        "state": d["_STATE"],
        "y": d["DIABETE3"].map({1: 1, 4: 1, 2: 0, 3: 0}),               # diabetes or prediabetes vs neither (gestational only = no)
        "HighBP": d["_RFHYPE5"].map({1: 0, 2: 1}),
        "HighChol": yn(d["TOLDHI2"]),
        "CholCheck": d["_CHOLCHK"].map({1: 1, 2: 0, 3: 0}),
        "BMI": d["_BMI5"] / 100,
        "Smoker": yn(d["SMOKE100"]),
        "Stroke": yn(d["CVDSTRK3"]),
        "HeartDiseaseorAttack": yn(d["_MICHD"]),
        "PhysActivity": yn(d["_TOTINDA"]),
        "Fruits": yn(d["_FRTLT1"]),
        "Veggies": yn(d["_VEGLT1"]),
        "HvyAlcoholConsump": d["_RFDRHV5"].map({1: 0, 2: 1}),
        "AnyHealthcare": yn(d["HLTHPLN1"]),
        "NoDocbcCost": yn(d["MEDCOST"]),
        "GenHlth": d["GENHLTH"].where(d["GENHLTH"].between(1, 5)),
        "MentHlth": d["MENTHLTH"].replace({88: 0}).where(lambda s: s.between(0, 30)),
        "PhysHlth": d["PHYSHLTH"].replace({88: 0}).where(lambda s: s.between(0, 30)),
        "DiffWalk": yn(d["DIFFWALK"]),
        "Sex": d["SEX"].map({1: 1, 2: 0}),
        "Age": d["_AGEG5YR"].where(d["_AGEG5YR"].between(1, 13)),
        "Education": d["EDUCA"].where(d["EDUCA"].between(1, 6)),
        "Income": d["INCOME2"].where(d["INCOME2"].between(1, 8))})
    n0 = len(out); out = out.dropna(); out = out[out.BMI < 70].reset_index(drop=True)   # sensitivity: implausible BMI removed
    out = out.astype(float); out["y"] = out["y"].astype(int); out["state"] = out["state"].astype(int)
    print(f"BRFSS 2015: {n0} respondents, {len(out)} complete records, prevalence {out.y.mean():.3f}")
    return out

DATA = load_brfss(DATA_PATH)
FEATURES = [c for c in DATA.columns if c not in ("state", "y")]
X_ALL = DATA[FEATURES].to_numpy(float); Y_ALL = DATA.y.to_numpy(); ST = DATA.state.to_numpy()
S_IDX = [FEATURES.index(c) for c in SCORE]
counts = DATA.state.value_counts()
DONORS = list(counts.index[2:4])                    # v18: new donors (v17 used the two largest)
RECEIVERS = [s for s in counts.index if s not in DONORS and counts[s] >= MIN_RECEIVER]
if QUICK: RECEIVERS = RECEIVERS[::max(1, len(RECEIVERS) // 6)][:6]
print("donors:", [STATE_NAMES.get(s, s) for s in DONORS], "| receivers:", len(RECEIVERS))
print(pd.DataFrame({"n": counts, "prevalence": DATA.groupby("state").y.mean()}).rename(index=STATE_NAMES).head(60).round(3).to_string())

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, log_loss
from xgboost import XGBClassifier
from scipy import stats
import statsmodels.api as sm

expit = lambda u: 1 / (1 + np.exp(-u))
COLS = FEATURES + ["y"]
_NB = [j for j in range(X_ALL.shape[1]) if len(np.unique(X_ALL[:5000, j])) > 2]

def cart_synth(df, n, seed):
    rng = np.random.default_rng(seed)
    out = pd.DataFrame({COLS[0]: rng.choice(df[COLS[0]].to_numpy(), n, replace=True)})
    for j, c in enumerate(COLS[1:], 1):
        prev = COLS[:j]
        tree = DecisionTreeRegressor(min_samples_leaf=CART_MIN_LEAF, random_state=seed + j).fit(df[prev], df[c])
        lr_, ls_ = tree.apply(df[prev]), tree.apply(out[prev])
        vals = df[c].to_numpy(); new = np.empty(n)
        groups = pd.Series(np.arange(len(df))).groupby(lr_).apply(np.array)
        for leaf in np.unique(ls_):
            pos = np.where(ls_ == leaf)[0]
            new[pos] = vals[rng.choice(groups[leaf], len(pos), replace=True)]
        out[c] = new
    return out

def _tilt(X): return np.c_[X, X[:, _NB] ** 2]

def ratio_classifier(pool_X, local_X):
    clf = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced"))])
    return clf.fit(np.vstack([_tilt(local_X), _tilt(pool_X)]), np.r_[np.ones(len(local_X)), np.zeros(len(pool_X))])

def est_weights(clf, X):
    p = np.clip(clf.predict_proba(_tilt(X))[:, 1], 1e-6, 1 - 1e-6)
    w = p / (1 - p); w = np.minimum(w, np.quantile(w, W_CLIP_Q)); return w / w.mean()

def resample_w(w, n, rng):
    p = w / w.sum(); return rng.choice(len(w), size=n, replace=True, p=p), 1 / np.sum(p ** 2) / len(p)

def fit_lr(X_hard=None, y_hard=None, X_soft=None, p_soft=None, C=1.0):
    Xs, ys, ws = [], [], []
    if X_hard is not None: Xs.append(X_hard); ys.append(y_hard); ws.append(np.ones(len(y_hard)))
    if X_soft is not None: Xs += [X_soft, X_soft]; ys += [np.ones(len(p_soft)), np.zeros(len(p_soft))]; ws += [p_soft, 1 - p_soft]
    m = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=C, max_iter=5000))])
    return m.fit(np.vstack(Xs), np.concatenate(ys), lr__sample_weight=np.concatenate(ws))

def fit_lr_w(X, y, w):
    m = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=1.0, max_iter=5000))])
    return m.fit(X, y, lr__sample_weight=w)

def logit(p): p = np.clip(p, 1e-9, 1 - 1e-9); return np.log(p / (1 - p))

def _glm(y, X, offset=None):
    try:
        r = sm.GLM(y, X, family=sm.families.Binomial(), offset=offset).fit()
        return r if r.converged and np.all(np.isfinite(r.params)) else None
    except Exception:
        return None

def closed_test(lp_tr, y_tr, lp_te):
    # Vergouwe et al. (2017): original / recalibration in the large / logistic recalibration
    ll0 = np.sum(y_tr * np.log(expit(lp_tr)) + (1 - y_tr) * np.log(1 - expit(lp_tr)))
    m1 = _glm(y_tr, np.ones((len(y_tr), 1)), offset=lp_tr); m2 = _glm(y_tr, sm.add_constant(lp_tr, has_constant="add"))
    ll = lambda m: -np.inf if m is None else m.llf
    sig = lambda a, b, df: 2 * (ll(a) - b) > stats.chi2.ppf(1 - ALPHA_CT, df)
    if m2 is None or not sig(m2, ll0, 2): return expit(lp_te), "original"
    if sig(m2, ll(m1), 1): return expit(m2.params[0] + m2.params[1] * lp_te), "recalibration"
    return (expit(m1.params[0] + lp_te), "intercept") if m1 is not None else (expit(lp_te), "original")

def teacher_fit(X, y, seed):
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9, eval_metric="logloss",
                        random_state=seed, n_jobs=-1, verbosity=0)
    return CalibratedClassifierCV(xgb, method="sigmoid", cv=5).fit(X, y)

def cal_stats(y, p):
    lp = logit(p)
    s = _glm(y, sm.add_constant(lp, has_constant="add")); c = _glm(y, np.ones((len(y), 1)), offset=lp)
    return (np.nan if s is None else s.params[1]), (np.nan if c is None else c.params[0])

def net_benefit(y, p, t):
    pos = p >= t; return float(np.mean(pos & (y == 1)) - np.mean(pos & (y == 0)) * t / (1 - t))

def done_keys():
    if not os.path.exists(OUT_CSV): return set()
    r = pd.read_csv(OUT_CSV); return set(zip(r.donor, r.receiver, r.rep))

def run_cell(donor, recv, rep):
    t0 = time.time()
    seed = SEED0 + 100000 * DONORS.index(donor) + 100 * int(recv) + rep
    rng = np.random.default_rng(seed)
    X, y, S = X_ALL, Y_ALL, S_IDX
    r_idx = rng.permutation(np.flatnonzero(ST == recv)); loc, te = r_idx[:N_LOCAL], r_idx[N_LOCAL:]
    don = rng.choice(np.flatnonzero(ST == donor), N_DONOR, replace=False)
    res = {}
    res["M0"] = fit_lr(X[loc][:, S], y[loc]).predict_proba(X[te][:, S])[:, 1]
    hD = fit_lr(X[don][:, S], y[don])
    lp_loc, lp_te = hD.decision_function(X[loc][:, S]), hD.decision_function(X[te][:, S])
    res["shared_raw"] = expit(lp_te)
    res["M1"], ct = closed_test(lp_loc, y[loc], lp_te)
    t = teacher_fit(X[don], y[don], seed)
    res["teacher"] = t.predict_proba(X[te])[:, 1]
    ddf = pd.DataFrame(X[don], columns=FEATURES).assign(y=y[don])
    pool = cart_synth(ddf, POOL_MULT * N_DONOR, seed)[FEATURES].to_numpy(float)
    p_pool = t.predict_proba(pool)[:, 1]
    clf = ratio_classifier(pool, X[loc])
    idxC, ess_pool = resample_w(est_weights(clf, pool), N_SYN, rng)
    w_d = est_weights(clf, X[don]); ess_don = float(w_d.sum() ** 2 / np.sum(w_d ** 2) / len(w_d))
    hIW = fit_lr_w(X[don][:, S], y[don], w_d)
    res["IW"] = hIW.predict_proba(X[te][:, S])[:, 1]
    res["IW_ct"], _ = closed_test(hIW.decision_function(X[loc][:, S]), y[loc], hIW.decision_function(X[te][:, S]))
    res["M2"] = fit_lr(X[loc][:, S], y[loc], pool[:N_SYN][:, S], p_pool[:N_SYN]).predict_proba(X[te][:, S])[:, 1]
    hM3 = fit_lr(X[loc][:, S], y[loc], pool[idxC][:, S], p_pool[idxC])
    res["M3"] = hM3.predict_proba(X[te][:, S])[:, 1]
    res["M3_ct"], _ = closed_test(hM3.decision_function(X[loc][:, S]), y[loc], hM3.decision_function(X[te][:, S]))
    # what the receiver computes before choosing
    lt = np.clip(t.predict_proba(X[loc])[:, 1], 1e-9, 1 - 1e-9); ls = np.clip(expit(lp_loc), 1e-9, 1 - 1e-9); yy = y[loc]
    d_mean = float(np.mean(-(yy * np.log(lt) + (1 - yy) * np.log(1 - lt)) + (yy * np.log(ls) + (1 - yy) * np.log(1 - ls))))
    lo, hi = pool.min(0), pool.max(0)
    frac_out = float(np.mean(np.any((X[loc] < lo) | (X[loc] > hi), axis=1)))
    choice = "M1" if d_mean >= 0 else ("M3" if ess_don < ESS_T else "IW")
    res["R1"] = res[choice]
    # corrected rule R2: support -> M1; else (ESS low and teacher better -> M3, otherwise IW), then local closed testing
    choice_s = "M1" if frac_out > SUPPORT_T else ("M3_ct" if (ess_don < ESS_T and d_mean < 0) else "IW_ct")
    res["R2"] = res[choice_s]
    rows = []
    for arm, p in res.items():
        pc = np.clip(p, 1e-9, 1 - 1e-9); sl, citl = cal_stats(y[te], pc)
        rows.append(dict(donor=donor, receiver=recv, rep=rep, arm=arm, logloss=log_loss(y[te], pc, labels=[0, 1]),
                         brier=float(np.mean((pc - y[te]) ** 2)), auc=roc_auc_score(y[te], pc), cal_slope=sl, citl=citl,
                         nb10=net_benefit(y[te], pc, 0.10), nb20=net_benefit(y[te], pc, 0.20),
                         n_test=len(te), prev_test=float(y[te].mean()), prev_loc=float(y[loc].mean()),
                         d_mean=d_mean, ess_don=ess_don, frac_out=frac_out, choice=choice, choice_s=choice_s, ct=ct))
    pd.DataFrame(rows).to_csv(OUT_CSV, mode="a", header=not os.path.exists(OUT_CSV), index=False)
    print(f"  {STATE_NAMES.get(donor, donor):>12s} -> {STATE_NAMES.get(recv, recv):<22s} rep {rep}  ESS {ess_don:.2f}  out {frac_out:.2f}  {choice:>2s}  {time.time() - t0:4.1f}s")

t_start = time.time()
for rep in range(N_REPS):
    _done = done_keys()
    for donor in DONORS:
        for recv in RECEIVERS:
            if (donor, recv, rep) not in _done: run_cell(donor, recv, rep)
print(f"Finished in {(time.time() - t_start) / 60:.1f} min -> {OUT_CSV}")

from scipy.stats import spearmanr
R = pd.read_csv(OUT_CSV); pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
L = R.pivot_table(index=["donor", "receiver", "rep"], columns="arm", values="logloss") * 1000
META = R[R.arm == "M1"].set_index(["donor", "receiver", "rep"])[["ess_don", "frac_out", "choice", "choice_s", "prev_test", "n_test"]]
BEST = L[["M1", "IW", "M3"]].min(axis=1)
STRATS = ["R2", "R1", "M1", "IW", "IW_ct", "M3", "M3_ct", "M2", "M0"]
reg = L[STRATS].sub(BEST, axis=0)
per_rep = reg.groupby(level="rep").mean()                                          # average over donors and receivers within a repetition
summ = pd.DataFrame({"mean regret": reg.mean(), "MC 95% half-width": 1.96 * per_rep.std() / np.sqrt(len(per_rep))})
print("Mean regret against the best of {M1, IW, M3} (x1000), all donors and receivers"); display(summ.round(2))
H1 = bool(reg["R2"].mean() < reg["R1"].mean()); H2b = bool(reg["R2"].mean() < min(reg[k].mean() for k in ["M1", "IW", "M3"]))
dd = (L.R1 - L.R2).groupby(level="rep").mean(); print(f"R1 - R2 (positive: R2 better) = {dd.mean():.2f} +/- {1.96 * dd.std() / np.sqrt(len(dd)):.2f}")
dd = (L.IW - L.R2).groupby(level="rep").mean(); print(f"IW - R2 (positive: R2 better) = {dd.mean():.2f} +/- {1.96 * dd.std() / np.sqrt(len(dd)):.2f}")

G = L[["M0", "IW", "IW_ct", "M3", "M3_ct", "R1", "R2"]].rsub(L.M1, axis=0)                  # gain over M1 (positive = better)
st = G.groupby(level=["donor", "receiver"]).mean().join(META.groupby(level=["donor", "receiver"])[["ess_don", "frac_out", "prev_test", "n_test"]].mean())
st["share_M3"] = META.choice_s.eq("M3_ct").groupby(level=["donor", "receiver"]).mean()
st["share_IW"] = META.choice_s.eq("IW_ct").groupby(level=["donor", "receiver"]).mean()
st = st.rename(index=STATE_NAMES)
print("\nPer donor and receiver: mean gain over M1 (x1000) and decision inputs"); display(st.round(2).sort_values(["donor", "M3"]))
H3 = bool((st["R2"] > -2).all())
pr = st[st.index.get_level_values("receiver") == "Puerto Rico"]["R2"]; H4 = bool((pr > -2).all())
print(f"\nH18.1 R2 regret < R1 regret: {H1}\nH18.2 R2 regret < every fixed strategy: {H2b}"
      f"\nH18.3 R2 never > 2 units worse than M1 in any state: {H3} (worst state {st['R2'].min():.2f})"
      f"\nH18.4 Puerto Rico, R2 vs M1: {H4} ({pr.round(2).to_dict()})")

# whole-pipeline uncertainty of the deployed model, per receiver (spread across repetitions)
C = R[R.arm.isin(["R2", "M1"])].pivot_table(index=["donor", "receiver", "rep"], columns="arm", values=["logloss", "cal_slope", "citl", "auc"])
def q(s): return f"{s.mean():.3f} [{s.quantile(0.025):.3f}, {s.quantile(0.975):.3f}]"
U = C.groupby(level=["donor", "receiver"]).agg(q)
U.index = [(STATE_NAMES.get(a, a), STATE_NAMES.get(b, b)) for a, b in U.index]
print("\nDeployed model across repetitions: mean [2.5%, 97.5%] (log-loss in nats, calibration slope, CITL, AUC)"); display(U)
print("\nChoices made by R2:"); display(META.choice_s.value_counts(normalize=True).round(3))
