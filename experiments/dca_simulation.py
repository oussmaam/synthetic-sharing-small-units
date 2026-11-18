import sys
_ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]   # positional arguments
display = lambda x: print(x.to_string() if hasattr(x,'to_string') else x)
import matplotlib; matplotlib.use('Agg')


DATASET = _ARGS[0] if _ARGS else "cdc_diabetes"   # "cdc_diabetes" or "natality"
NAT_URL, NAT_READ, NAT_N = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/DVS/natality/Nat2023us.zip", 0.02, 10**9
CSV_PATH, TARGET_COL, POSITIVE, DROP_COLS = "x.csv", None, None, []

SHIFT_VAR, AGE_VAR = "BMI", "Age"
GAMMAS   = [0.0, 0.25, 0.5, 0.75, 1.0]   # misspecification of the logistic model
SKEWS    = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]   # receiver share above the BMI median (donor: 1 - share)
DESIGN   = [(g, s) for g in [0.0, 0.5, 1.0] for s in [0.5, 0.7, 0.9]]                     # the v9 grid: used to choose the rule
HELDOUT  = [(g, s) for g in [0.25, 0.75] for s in [0.6, 0.8, 0.95]] + [(g, s) for g in [0.0, 1.0] for s in [0.8, 0.95]]
SCEN     = DESIGN + HELDOUT
N_LOCAL  = 200                     # sensitivity: 100 / 400
N_DONOR  = 10000
TEST_N   = 9500
N_SYN    = 10000                  # synthetic patients sent by the donor
POOL_MULT = 3                     # synthetic pool for targeting = POOL_MULT x N_DONOR
POP_N    = 30000 if DATASET == "cdc_diabetes" else 8000   # population samples for the oracles
W_CLIP_Q, CART_MIN_LEAF, ALPHA_CT = 0.99, 5, 0.05

QUICK  = "--quick" in sys.argv          # --quick: 2 repetitions. Default: the full design.
N_REPS = 2 if QUICK else 20
SEED0  = 2026
TEACHER_N = None                  # sensitivity: 1000 = weaker teacher (trained on 1,000 donor women)
OUT_CSV = f"results_v15a_{DATASET}_{'quick' if QUICK else 'full'}{'' if N_LOCAL == 200 else f'_n{N_LOCAL}'}{'' if TEACHER_N is None else f'_t{TEACHER_N}'}.csv"
OUT_COEF = OUT_CSV.replace('.csv', '_coef.csv')
print(f"{DATASET} QUICK={QUICK} reps={N_REPS} -> {OUT_CSV}")

N_REPS = 2 if QUICK else 10; OUT_CSV = f"results_dca_sim_{DATASET}{'_quick' if QUICK else ''}.csv"; OUT_COEF = OUT_CSV.replace('.csv','_coef.csv')


import warnings, os, time, itertools
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

NAT_FIELDS = {   # 1-based positions from the NCHS 2023 natality user guide -> python slices
    "MAGER": (74, 76), "BMI": (282, 286), "RF_PDIAB": (312, 313), "RF_PHYPE": (314, 315),
    "RF_GHYPE": (315, 316), "RF_EHYPE": (316, 317), "RF_PPTERM": (317, 318), "RF_INFTR": (324, 325),
    "CIG_0": (252, 254), "PRIORLIVE": (170, 172), "PRIORDEAD": (172, 174), "MEDUC": (123, 124),
    "DPLURAL": (453, 454), "PAY_REC": (435, 436)}

def _zip_lines(zpath):
    # The CDC archive uses Deflate64, which Python's zipfile cannot read; stream it through the system unzip tool instead.
    import zipfile, subprocess, shutil
    try:
        with zipfile.ZipFile(zpath) as z:
            name = [n for n in z.namelist() if n.lower().endswith(".txt")][0]
            with z.open(name) as f:
                yield from f
        return
    except NotImplementedError:
        pass
    if shutil.which("unzip"):
        cmd = ["unzip", "-p", zpath]
    elif shutil.which("7z"):
        cmd = ["7z", "e", "-so", zpath]
    else:
        raise RuntimeError("Install unzip first:  !apt-get -qq install unzip")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1 << 20)
    yield from p.stdout
    if p.wait() != 0:
        raise RuntimeError(f"{cmd[0]} failed while reading {zpath}")

def load_natality():
    import urllib.request, zipfile, random
    cache = "natality2023_sample.csv"
    if not os.path.exists(cache):
        zpath = "Nat2023us.zip"
        if not os.path.exists(zpath):
            print("downloading the 2023 natality file (about 230 MB) ..."); urllib.request.urlretrieve(NAT_URL, zpath)
        rnd = random.Random(SEED0); rows = []
        for line in _zip_lines(zpath):
            if rnd.random() >= NAT_READ: continue
            s = line.decode("latin-1")
            rows.append([s[a:b] for a, b in NAT_FIELDS.values()])
        pd.DataFrame(rows, columns=list(NAT_FIELDS)).to_csv(cache, index=False)
        print(f"read {len(rows)} birth records -> {cache}")
    d = pd.read_csv(cache, dtype=str).apply(lambda c: c.str.strip())
    n0 = len(d)
    rf = ["RF_PDIAB", "RF_PHYPE", "RF_GHYPE", "RF_EHYPE", "RF_PPTERM", "RF_INFTR"]
    d = d[(d.DPLURAL == "1") & d[rf].isin(["Y", "N"]).all(axis=1)]
    num = {c: pd.to_numeric(d[c], errors="coerce") for c in ["MAGER", "BMI", "CIG_0", "PRIORLIVE", "PRIORDEAD", "MEDUC", "PAY_REC"]}
    ok = (num["BMI"] < 99) & (num["CIG_0"] < 99) & (num["PRIORLIVE"] < 99) & (num["PRIORDEAD"] < 99) \
         & num["MEDUC"].between(1, 8) & (d.RF_PHYPE == "N")      # women with chronic hypertension excluded
    d = d[ok]; num = {k: v[ok] for k, v in num.items()}
    parity = num["PRIORLIVE"] + num["PRIORDEAD"]
    X = pd.DataFrame({
        "Age": num["MAGER"], "BMI": num["BMI"],
        "nulliparous": (parity == 0).astype(int), "prior_births": parity.clip(upper=6),
        "pregestational_diabetes": (d.RF_PDIAB == "Y").astype(int),
        "previous_preterm": (d.RF_PPTERM == "Y").astype(int), "infertility_treatment": (d.RF_INFTR == "Y").astype(int),
        "cigarettes_before": num["CIG_0"].clip(upper=40), "education": num["MEDUC"]})
    y = ((d.RF_GHYPE == "Y") | (d.RF_EHYPE == "Y")).astype(int)
    out = X.assign(y=y.values).sample(min(NAT_N, len(X)), random_state=SEED0).reset_index(drop=True)
    print(f"natality: {n0} records read | {len(X)} singleton births, no chronic hypertension, complete data | {len(out)} kept")
    return out

def load_raw():
    if DATASET == "natality":
        return load_natality()
    if DATASET == "uci":
        from ucimlrepo import fetch_ucirepo
        d = fetch_ucirepo(id=863)
        X = d.data.features.copy()
        y_raw = d.data.targets.iloc[:, 0].astype(str).str.strip().str.lower()
        y = (y_raw == "high risk").astype(int)          # high risk vs low/mid risk
    else:
        df = pd.read_csv(CSV_PATH).drop(columns=DROP_COLS, errors="ignore")
        tcol = TARGET_COL or df.columns[-1]
        pos = POSITIVE if POSITIVE is not None else df[tcol].value_counts().idxmin()
        y = (df[tcol] == pos).astype(int); X = df.drop(columns=[tcol])
        print(f"Outcome column {tcol!r}, positive value {pos!r}")
    X = X.select_dtypes(include="number")
    X = X[X.columns[X.isna().mean() <= 0.30]]
    return X.assign(y=y.values)

def audit(df):
    if DATASET == "natality":   # registry data: identical rows are different women, so no de-duplication
        feats = [c for c in df.columns if c != "y"]
        print(f"features: {feats}\nprevalence of gestational hypertension/eclampsia: {df.y.mean():.3f}")
        return df.reset_index(drop=True)
    n0 = len(df); df = df.dropna(); n1 = len(df)
    df = df.drop_duplicates(); n2 = len(df)
    feats = [c for c in df.columns if c != "y"]
    conflict = df.groupby(feats)["y"].transform("nunique") > 1
    df = df[~conflict].reset_index(drop=True)
    print(f"rows: raw {n0} | complete {n1} | after removing exact duplicates {n2} "
          f"| after removing {int(conflict.sum())} conflicting rows {len(df)}")
    print(f"features: {feats}\nprevalence: {df.y.mean():.3f}")
    assert SHIFT_VAR in feats, f"SHIFT_VAR {SHIFT_VAR!r} is not a column"
    return df



def load_cdc():
    cache = "cdc_diabetes_binary.csv"
    if os.path.exists(cache):
        df = pd.read_csv(cache)
    else:
        from ucimlrepo import fetch_ucirepo
        d = fetch_ucirepo(id=891)
        df = d.data.features.copy(); df["y"] = d.data.targets["Diabetes_binary"].values
        df.to_csv(cache, index=False)
    df = df[df.BMI < 70]                      # implausible BMI values (recording errors), see Supplement S2
    df = df.astype(float); df["y"] = df["y"].astype(int)
    print(f"CDC Diabetes Health Indicators: {len(df)} adults, prevalence {df.y.mean():.3f}")
    return df.reset_index(drop=True)

DATA = load_cdc() if DATASET == "cdc_diabetes" else audit(load_raw())
FEATURES = [c for c in DATA.columns if c != "y"]
assert SHIFT_VAR in FEATURES and AGE_VAR in FEATURES

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.optimize import brentq

X_ALL = DATA[FEATURES].to_numpy(float)
_Z = (X_ALL - X_ALL.mean(0)) / X_ALL.std(0)
_Z = np.nan_to_num(_Z)
_BETA = LogisticRegression(C=1e4, max_iter=5000).fit(_Z, DATA.y).coef_[0]
_zb = np.clip(_Z[:, FEATURES.index(SHIFT_VAR)], -2.5, 2.5)
_G = (_zb ** 2 - np.mean(_zb ** 2)) + _zb * _Z[:, FEATURES.index(AGE_VAR)]
_LIN = _Z @ _BETA
PREV = DATA.y.mean()
expit = lambda u: 1 / (1 + np.exp(-u))
ETA = {}
for g in GAMMAS:
    b0 = brentq(lambda c: expit(c + _LIN + g * _G).mean() - PREV, -30, 30)
    ETA[g] = expit(b0 + _LIN + g * _G)
_HIGH = (DATA[SHIFT_VAR] > DATA[SHIFT_VAR].median()).to_numpy()
RULE = (FEATURES.index(SHIFT_VAR), DATA[SHIFT_VAR].median(), False)

# Where does the real outcome sit on the misspecification axis? (held-out log-loss of logistic vs XGBoost on the real outcome)
from xgboost import XGBClassifier
from sklearn.metrics import log_loss
_i = np.random.default_rng(0).permutation(len(DATA)); _tr, _te = _i[:40000], _i[40000:80000]
_lr = LogisticRegression(C=1.0, max_iter=5000).fit(_Z[_tr], DATA.y.iloc[_tr])
_xg = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9, verbosity=0, n_jobs=-1).fit(_Z[_tr], DATA.y.iloc[_tr])
REAL_GAP = log_loss(DATA.y.iloc[_te], _lr.predict_proba(_Z[_te])[:, 1]) - log_loss(DATA.y.iloc[_te], _xg.predict_proba(_Z[_te])[:, 1])
print(f"real outcome: log-loss(logistic) - log-loss(XGBoost) = {REAL_GAP:+.4f} nats (rough misspecification index)")

import statsmodels.api as sm
from scipy import stats
from sklearn.tree import DecisionTreeRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, log_loss

COLS = FEATURES + ["y"]

def draw(avail, n, share_high, rng):
    nh = int(round(share_high * n))
    idx = np.r_[rng.choice(np.flatnonzero(avail & _HIGH), nh, replace=False),
                rng.choice(np.flatnonzero(avail & ~_HIGH), n - nh, replace=False)]
    avail[idx] = False
    return rng.permutation(idx)                     # shuffle: draws are built high-then-low, and calibration CV folds must not follow that order

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

def _tilt_features(X):
    # features for the density-ratio classifier: all covariates plus squares of the non-binary ones
    nb = [j for j in range(X.shape[1]) if len(np.unique(X_ALL[:2000, j])) > 2]
    return np.c_[X, X[:, nb] ** 2]

def target_resample(pool_X, local_X, n, rng):
    # Targeting done by the RECEIVER on the synthetic pool it received (its own covariates never leave the site):
    # a probabilistic classifier separates local covariates from the pool, and its odds estimate the density ratio p_R / q
    F_pool, F_loc = _tilt_features(pool_X), _tilt_features(local_X)
    clf = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced"))])
    clf.fit(np.vstack([F_loc, F_pool]), np.r_[np.ones(len(F_loc)), np.zeros(len(F_pool))])
    p = np.clip(clf.predict_proba(F_pool)[:, 1], 1e-6, 1 - 1e-6)
    w = p / (1 - p); w = np.minimum(w, np.quantile(w, W_CLIP_Q)); w /= w.sum()
    return rng.choice(len(pool_X), size=n, replace=True, p=w), 1 / np.sum(w ** 2) / len(w)



# ---------------- v8 additions ----------------
T_NB = [round(x,2) for x in np.arange(0.02, 0.405, 0.01)]          # risk thresholds for net benefit
B_CT = 30                          # simulations of the closed test with n local patients (finite-sample prediction)

def ratio_classifier(pool_X, local_X):
    # receiver-side density-ratio model: local covariates (1) vs the donor's synthetic covariates (0)
    F_pool, F_loc = _tilt_features(pool_X), _tilt_features(local_X)
    clf = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced"))])
    return clf.fit(np.vstack([F_loc, F_pool]), np.r_[np.ones(len(F_loc)), np.zeros(len(F_pool))])

def est_weights(clf, X):
    p = np.clip(clf.predict_proba(_tilt_features(X))[:, 1], 1e-6, 1 - 1e-6)
    w = p / (1 - p); w = np.minimum(w, np.quantile(w, W_CLIP_Q)); return w / w.mean()

def exact_weights(X, sk, rule):
    # the true density ratio p_R / p_donor of the simulated sites (known by design): sk/(1-sk) above the median, (1-sk)/sk below
    j, thr, ge = rule
    h = X[:, j] >= thr if ge else X[:, j] > thr
    return np.where(h, sk / (1 - sk), (1 - sk) / sk)

def resample_w(w, n, rng):
    p = w / w.sum(); return rng.choice(len(w), size=n, replace=True, p=p), 1 / np.sum(p ** 2) / len(p)

def fit_lr_w(X, y, w):
    m = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=1.0, max_iter=5000))])
    return m.fit(X, y, lr__sample_weight=w)

def net_benefit(y, p, t):
    pos = p >= t
    return float(np.mean(pos & (y == 1)) - np.mean(pos & (y == 0)) * t / (1 - t))

def cal_stats(y, p):
    lp = logit(p)
    s = _glm(y, sm.add_constant(lp, has_constant="add")); c = _glm(y, np.ones((len(y), 1)), offset=lp)
    return (np.nan if s is None else s.params[1]), (np.nan if c is None else c.params[0])

def ct_finite(lp_fn, X_pop, eta_pop, lp_te, e_te, n, B, rng):
    # expected excess risk of "population donor model + closed test on n local patients": the finite-sample
    # counterpart of Gamma_upd (the asymptotic version assumes perfect local updating)
    out = []
    for _ in range(B):
        i = rng.choice(len(X_pop), n, replace=False)
        yb = (rng.random(n) < eta_pop[i]).astype(int)
        p, _ = closed_test(lp_fn(X_pop[i]), yb, lp_te)
        out.append(kl_excess(e_te, p))
    return float(np.mean(out))

def fit_lr(X_hard=None, y_hard=None, X_soft=None, p_soft=None, C=1.0):
    # logistic regression on hard local labels and/or soft (teacher-probability) labels: each soft row is
    # duplicated with y=1 (weight p) and y=0 (weight 1-p), which is the infinite-relabelling limit of method B
    Xs, ys, ws = [], [], []
    if X_hard is not None:
        Xs.append(X_hard); ys.append(y_hard); ws.append(np.ones(len(y_hard)))
    if X_soft is not None:
        Xs += [X_soft, X_soft]; ys += [np.ones(len(p_soft)), np.zeros(len(p_soft))]; ws += [p_soft, 1 - p_soft]
    m = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=C, max_iter=5000))])
    return m.fit(np.vstack(Xs), np.concatenate(ys), lr__sample_weight=np.concatenate(ws))

def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9); return np.log(p / (1 - p))

def _glm(y, X, offset=None, w=None):
    try:
        r = sm.GLM(y, X, family=sm.families.Binomial(), offset=offset, freq_weights=w).fit()
        if not r.converged or not np.all(np.isfinite(r.params)): return None
        return r
    except Exception:
        return None

def recal(lp_tr, y_tr, lp_te):
    r = _glm(y_tr, sm.add_constant(lp_tr, has_constant="add"))
    return expit(lp_te) if r is None else expit(r.params[0] + r.params[1] * lp_te)

def closed_test(lp_tr, y_tr, lp_te):
    # Closed testing (Vergouwe et al. 2017) among original / recalibration-in-the-large / logistic recalibration.
    # Model revision (re-estimating all coefficients) is not offered: with a few hundred patients it overfits badly.
    ll0 = np.sum(y_tr * np.log(expit(lp_tr)) + (1 - y_tr) * np.log(1 - expit(lp_tr)))
    m1 = _glm(y_tr, np.ones((len(y_tr), 1)), offset=lp_tr)
    m2 = _glm(y_tr, sm.add_constant(lp_tr, has_constant="add"))
    ll = lambda m: -np.inf if m is None else m.llf
    sig = lambda a, b, df: 2 * (ll(a) - b) > stats.chi2.ppf(1 - ALPHA_CT, df)
    if m2 is None or not sig(m2, ll0, 2):
        return expit(lp_te), "original"
    if sig(m2, ll(m1), 1):
        return expit(m2.params[0] + m2.params[1] * lp_te), "recalibration"
    return (expit(m1.params[0] + lp_te), "intercept") if m1 is not None else (expit(lp_te), "original")

def kl_excess(eta, p):
    p = np.clip(p, 1e-9, 1 - 1e-9); e = np.clip(eta, 1e-12, 1 - 1e-12)
    return float(np.mean(e * np.log(e / p) + (1 - e) * np.log((1 - e) / (1 - p))))

def teacher_fit(X, y, seed):
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.9,
                        eval_metric="logloss", random_state=seed, n_jobs=-1, verbosity=0)
    return CalibratedClassifierCV(xgb, method="sigmoid", cv=5).fit(X, y)
def _coef_rows(models, Xte, cols, meta):
    sd = Xte.std(0); sd[sd == 0] = 1; mad = np.abs(Xte - Xte.mean(0)).mean(0); out = []
    for name, m in models.items():
        sc, lr = m.named_steps["sc"], m.named_steps["lr"]
        b = lr.coef_[0] / np.where(sc.scale_ == 0, 1, sc.scale_)
        for j, c in enumerate(cols):
            out.append(dict(meta, model=name, feature=c, beta_std=b[j] * sd[j], imp=abs(b[j]) * mad[j]))
    return out


# ---------------- v15: Brown et al. (2026) comparator ----------------
BR_BIG = N_SYN                     # volume-matched variant: as many synthetic rows as our targeted pool

def riley_n(P, phi, S=0.9, delta=0.05, r2_frac=0.15):
    # Riley et al. (BMJ 2020) minimum sample size for a binary-outcome model with P parameters
    phi = float(np.clip(phi, 0.01, 0.5))
    ln0 = phi * np.log(phi) + (1 - phi) * np.log(1 - phi)
    max_r2 = 1 - np.exp(2 * ln0); r2 = r2_frac * max_r2
    n1 = P / ((S - 1) * np.log(1 - r2 / S))                               # expected shrinkage >= S
    s_vh = r2 / (r2 + delta * max_r2)
    n2 = P / ((s_vh - 1) * np.log(1 - r2 / s_vh))                         # optimism in Nagelkerke R2 <= delta
    n3 = (1.96 / 0.05) ** 2 * phi * (1 - phi)                             # overall risk within +/- 0.05
    return int(np.ceil(max(n1, n2, n3)))

def cart_synth_cond(df_d, df_l, n, seed):
    # Brown et al. "SDG, Conditional": one generator on the union of the sites' data with a site indicator, sampled
    # conditionally on the receiver's site (this needs the receiver's raw data at the donor)
    rng = np.random.default_rng(seed)
    df = pd.concat([df_d.assign(_site=0.0), df_l.assign(_site=1.0)], ignore_index=True)
    cols = ["_site"] + COLS
    out = pd.DataFrame({"_site": np.ones(n)})
    for j, c in enumerate(cols[1:], 1):
        prev = cols[:j]
        tree = DecisionTreeRegressor(min_samples_leaf=CART_MIN_LEAF, random_state=seed + j).fit(df[prev], df[c])
        lr_, ls_ = tree.apply(df[prev]), tree.apply(out[prev])
        vals = df[c].to_numpy(); new = np.empty(n)
        groups = pd.Series(np.arange(len(df))).groupby(lr_).apply(np.array)
        for leaf in np.unique(ls_):
            pos = np.where(ls_ == leaf)[0]
            new[pos] = vals[rng.choice(groups[leaf], len(pos), replace=True)]
        out[c] = new
    return out[COLS]


COLUMNS = ["dataset", "gamma", "share", "rep", "arm", "ex_kl", "ex_brier", "brier_y", "auc", "ct_choice", "ess_C", "cal_slope", "citl", *[f"nb{int(round(t*100))}" for t in T_NB], "est_G", "est_Gupd_n", "ll_teacher_loc", "ll_score_loc", "ess_don", "p_better", "d_mean", "est_null", "n_riley"]

def done_keys():
    if not os.path.exists(OUT_CSV): return set()
    r = pd.read_csv(OUT_CSV); return set(zip(r.gamma, r["share"], r.rep))

def run_cell(gam, sk, rep):
    t0 = time.time()
    seed = SEED0 + 1000 * rep + 100 * GAMMAS.index(gam) + 10 * SKEWS.index(sk)
    rng = np.random.default_rng(seed)
    eta = ETA[gam]
    avail = np.ones(len(DATA), bool)
    te, loc, rpop = draw(avail, TEST_N, sk, rng), draw(avail, N_LOCAL, sk, rng), draw(avail, POP_N, sk, rng)
    don, dpop = draw(avail, N_DONOR, 1 - sk, rng), draw(avail, POP_N, 1 - sk, rng)
    y = (rng.random(len(DATA)) < eta).astype(int)                 # one outcome draw per repetition
    X = X_ALL; SIDX = np.arange(X.shape[1])
    e_te = eta[te]
    res = {}

    # receiver-only and oracle projections
    res["local"] = fit_lr(X[loc], y[loc]).predict_proba(X[te])[:, 1]
    h_star = fit_lr(X_soft=X[rpop], p_soft=eta[rpop], C=1e4)
    h_Dpop = fit_lr(X_soft=X[dpop], p_soft=eta[dpop], C=1e4)
    res["oracle_h_star"] = h_star.predict_proba(X[te])[:, 1]
    res["oracle_h_D"] = h_Dpop.predict_proba(X[te])[:, 1]
    lpD_pop = h_Dpop.decision_function(X[rpop])
    rc = fit_lr(X_soft=lpD_pop[:, None], p_soft=eta[rpop], C=1e4)
    res["oracle_h_D_recal"] = rc.predict_proba(h_Dpop.decision_function(X[te])[:, None])[:, 1]

    # model sharing: the donor publishes its logistic model fitted on its real data
    hD = fit_lr(X[don], y[don])
    lp_loc, lp_te = hD.decision_function(X[loc]), hD.decision_function(X[te])
    res["share_LR"] = expit(lp_te)
    res["share_LR_recal"] = recal(lp_loc, y[loc], lp_te)
    res["share_LR_ct"], ct = closed_test(lp_loc, y[loc], lp_te)
    extra = {"oracle_D_ct_n": ct_finite(h_Dpop.decision_function, X[rpop], eta[rpop], h_Dpop.decision_function(X[te]), e_te, N_LOCAL, B_CT, rng)}

    # donor's teacher (calibrated XGBoost on its real data) and synthetic data
    _tsub = don if TEACHER_N is None else np.random.default_rng(seed + 5).choice(don, TEACHER_N, replace=False)
    t = teacher_fit(X[_tsub], y[_tsub], seed)
    res["teacher"] = t.predict_proba(X[te])[:, 1]
    ddf = pd.DataFrame(X[don], columns=FEATURES).assign(y=y[don])
    pool = cart_synth(ddf, POOL_MULT * N_DONOR, seed)[FEATURES].to_numpy(float)
    p_pool = t.predict_proba(pool)[:, 1]
    synB = np.arange(N_SYN)
    clf = ratio_classifier(pool, X[loc])
    idxC, ess = resample_w(est_weights(clf, pool), N_SYN, rng)
    idxE, _ = resample_w(exact_weights(pool, sk, RULE), N_SYN, rng)
    hIW = fit_lr_w(X[don], y[don], est_weights(clf, X[don]))
    res["IW_donor"] = hIW.predict_proba(X[te])[:, 1]
    res["IW_donor_ct"], _ = closed_test(hIW.decision_function(X[loc]), y[loc], hIW.decision_function(X[te]))
    res["IW_donor_exact"] = fit_lr_w(X[don], y[don], exact_weights(X[don], sk, RULE)).predict_proba(X[te])[:, 1]
    res["C_exact_ratio"] = fit_lr(X[loc], y[loc], pool[idxE], p_pool[idxE]).predict_proba(X[te])[:, 1]
    # ---- v9: what the receiver can estimate BEFORE choosing (no access to the truth) ----
    Xt, pt = pool[idxC], p_pool[idxC]                                   # targeted pool with teacher soft labels
    h_hat = fit_lr(X_soft=Xt, p_soft=pt, C=1e4)                    # receiver's estimate of its best simple score
    ph = h_hat.predict_proba(Xt)[:, 1]
    lpD_t = hD.decision_function(Xt)
    est_G = kl_excess(ph, expit(lpD_t))                                 # estimated gain over the shared score
    upd = []
    for _ in range(B_CT):                                               # closed test on n simulated local patients
        i = rng.choice(len(Xt), N_LOCAL, replace=False)
        yb = (rng.random(N_LOCAL) < pt[i]).astype(int)
        p_b, _ = closed_test(lpD_t[i], yb, lpD_t)
        upd.append(kl_excess(ph, p_b))
    est_Gupd_n = float(np.mean(upd))                                    # estimated gain over the updated shared score
    ll_teacher_loc = log_loss(y[loc], np.clip(t.predict_proba(X[loc])[:, 1], 1e-9, 1 - 1e-9), labels=[0, 1])
    ll_score_loc = log_loss(y[loc], np.clip(expit(lp_loc), 1e-9, 1 - 1e-9), labels=[0, 1])
    w_d = est_weights(clf, X[don]); ess_don = float(w_d.sum() ** 2 / np.sum(w_d ** 2) / len(w_d))

    # ---- v13: bootstrap of Q1, bias-corrected gain estimate, absolute ESS ----
    rng2 = np.random.default_rng(seed + 999)
    _lt = np.clip(t.predict_proba(X[loc])[:, 1], 1e-9, 1 - 1e-9); _ls = np.clip(expit(lp_loc), 1e-9, 1 - 1e-9); _yy = y[loc]
    _d = -(_yy * np.log(_lt) + (1 - _yy) * np.log(1 - _lt)) + (_yy * np.log(_ls) + (1 - _yy) * np.log(1 - _ls))
    _bm = _d[rng2.integers(0, len(_d), (2000, len(_d)))].mean(1)
    p_better, d_mean = float((_bm < 0).mean()), float(_d.mean())
    Xu, pu = pool[np.arange(N_SYN)], p_pool[np.arange(N_SYN)]           # untargeted, donor-like pool: true gain there is ~0
    h_u = fit_lr(X_soft=Xu[:, SIDX], p_soft=pu, C=1e4); ph_u = h_u.predict_proba(Xu[:, SIDX])[:, 1]
    lpD_u = hD.decision_function(Xu[:, SIDX]); upd_u = []
    for _ in range(B_CT):
        i = rng2.choice(len(Xu), N_LOCAL, replace=False)
        yb = (rng2.random(N_LOCAL) < pu[i]).astype(int)
        p_b, _ = closed_test(lpD_u[i], yb, lpD_u)
        upd_u.append(kl_excess(ph_u, p_b))
    est_null = float(np.mean(upd_u))

    mB = fit_lr(X[loc], y[loc], pool[synB], p_pool[synB]); res["B_untargeted"] = mB.predict_proba(X[te])[:, 1]
    mC = fit_lr(X[loc], y[loc], pool[idxC], p_pool[idxC]); res["C_target"] = mC.predict_proba(X[te])[:, 1]
    xo = rpop[:N_SYN]
    res["C_oracle"] = fit_lr(X[loc], y[loc], X[xo], t.predict_proba(X[xo])[:, 1]).predict_proba(X[te])[:, 1]
    res["local_distill"] = fit_lr(X_soft=X[loc], p_soft=t.predict_proba(X[loc])[:, 1]).predict_proba(X[te])[:, 1]


    # ---- v15: Brown et al. (2026) style augmentation + Platt recalibration of the shared model ----
    ZF = lambda Z: Z
    n_req = riley_n(X.shape[1], float(np.mean(y[loc]))); t_add = max(n_req - N_LOCAL, 0)
    rng3 = np.random.default_rng(seed + 1515)
    ldf = pd.DataFrame(X[loc], columns=FEATURES).assign(y=y[loc])
    def _platt(Xa, ya): return recal(hD.decision_function(ZF(Xa)), ya, lp_te)
    def _os(k): i = rng3.integers(0, N_LOCAL, k); return X[loc][i], y[loc][i]
    nb_ = max(t_add, BR_BIG)
    _sl = cart_synth(ldf, nb_, seed + 15); slX, sly = _sl[FEATURES].to_numpy(float), np.rint(_sl["y"].to_numpy()).astype(int)
    _sc = cart_synth_cond(ddf, ldf, nb_, seed + 16); scX, scy = _sc[FEATURES].to_numpy(float), np.rint(_sc["y"].to_numpy()).astype(int)
    oX, oy = _os(t_add); h2 = t_add // 2; oX2, oy2 = _os(t_add - h2)
    res["Platt_local"] = _platt(X[loc], y[loc])
    res["BR_OS"] = _platt(np.r_[X[loc], oX], np.r_[y[loc], oy])
    res["BR_SDG"] = _platt(np.r_[X[loc], slX[:t_add]], np.r_[y[loc], sly[:t_add]])
    res["BR_SDG_OS"] = _platt(np.r_[X[loc], slX[:h2], oX2], np.r_[y[loc], sly[:h2], oy2])
    res["BR_cond"] = _platt(np.r_[X[loc], scX[:t_add]], np.r_[y[loc], scy[:t_add]])
    res["BR_SDG_big"] = _platt(np.r_[X[loc], slX[:BR_BIG]], np.r_[y[loc], sly[:BR_BIG]])
    res["BR_cond_big"] = _platt(np.r_[X[loc], scX[:BR_BIG]], np.r_[y[loc], scy[:BR_BIG]])
    res["LOC_SDG_refit"] = fit_lr(np.r_[ZF(X[loc]), ZF(slX[:BR_BIG])], np.r_[y[loc], sly[:BR_BIG]]).predict_proba(ZF(X[te]))[:, 1]

    rows = []
    for arm, p in res.items():
        rows.append(dict(dataset=DATASET, gamma=gam, share=sk, rep=rep, arm=arm, ex_kl=kl_excess(e_te, p),
                         ex_brier=float(np.mean((p - e_te) ** 2)), brier_y=float(np.mean((p - y[te]) ** 2)),
                         auc=roc_auc_score(y[te], p), ct_choice=ct if arm == "share_LR_ct" else "",
                         ess_C=ess if arm == "C_target" else np.nan,
                         **dict(zip(["cal_slope", "citl"], cal_stats(y[te], np.clip(p, 1e-9, 1 - 1e-9)))),
                         **{f"nb{int(round(t*100))}": net_benefit(y[te], p, t) for t in T_NB}))
    rows.append(dict(dataset=DATASET, gamma=gam, share=sk, rep=rep, arm="treat_all", **{f"nb{int(round(t*100))}": net_benefit(y[te], np.ones(len(te)), t) for t in T_NB}))
    for arm, v in extra.items():
        rows.append(dict(dataset=DATASET, gamma=gam, share=sk, rep=rep, arm=arm, ex_kl=v))
    rows.append(dict(dataset=DATASET, gamma=gam, share=sk, rep=rep, arm="decision_inputs", est_G=est_G, est_Gupd_n=est_Gupd_n, ll_teacher_loc=ll_teacher_loc, ll_score_loc=ll_score_loc, ess_don=ess_don, p_better=p_better, d_mean=d_mean, est_null=est_null, n_riley=n_req))
    _cr = _coef_rows({"shared": hD, "M2_untargeted": mB, "M3_targeted": mC, "IW": hIW, "oracle": h_star}, X[te], FEATURES, dict(dataset=DATASET, gamma=gam, share=sk, rep=rep))
    pd.DataFrame(_cr).to_csv(OUT_COEF, mode="a", header=not os.path.exists(OUT_COEF), index=False)
    pd.DataFrame(rows).reindex(columns=COLUMNS).to_csv(OUT_CSV, mode="a", header=not os.path.exists(OUT_CSV), index=False)
    print(f"  gamma {gam:.1f} share {sk:.1f} rep {rep:2d}  {time.time() - t0:5.1f}s")

t_start = time.time()
for rep in range(N_REPS):                                               # repetitions outermost: partial runs cover all scenarios
    _done = done_keys()
    for gam, sk in SCEN:
        if (gam, sk, rep) not in _done: run_cell(gam, sk, rep)
print(f"Finished in {(time.time() - t_start) / 60:.1f} min -> {OUT_CSV}")

