import sys
_ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]   # positional arguments
import matplotlib; matplotlib.use('Agg')
def display(x): print(x.to_string() if hasattr(x,'to_string') else x)
DATASET = _ARGS[0] if _ARGS else "cdc_diabetes"   # "cdc_diabetes" or "natality"
NAT_URL, NAT_READ, NAT_N = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/DVS/natality/Nat2023us.zip", 0.02, 10**9
CSV_PATH, TARGET_COL, POSITIVE, DROP_COLS = "x.csv", None, None, []

if DATASET == "natality":
    SCORE = ["Age", "BMI", "nulliparous"]
    SHIFTS = {"BMI (in score)": "BMI", "education (omitted)": "education", "age (in score)": "Age", "prior births (omitted)": "prior_births"}
    POP_N = 5000
else:
    SCORE = ["Age", "BMI", "HighBP"]
    SHIFTS = {"BMI (in score)": "BMI", "general health (omitted)": "GenHlth", "age (in score)": "Age", "income (omitted)": "Income"}
    POP_N = 20000
SKEWS    = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
_OLD     = list(SHIFTS)[:2]; _NEW = list(SHIFTS)[2:]
DESIGN   = [(v, s) for v in _OLD for s in [0.5, 0.7, 0.9]]                                    # the v9 grid: used to choose the rule
HELDOUT  = [(v, s) for v in _NEW for s in [0.6, 0.8, 0.95]] + [(v, s) for v in _OLD for s in [0.8, 0.95]]
SCEN     = DESIGN + HELDOUT
SCEN     = [(v, s_, q) for v in list(SHIFTS)[:2] for s_ in [0.5, 0.9] for q in [1.0, 0.9, 0.8]]   # v16
N_LOCAL, N_DONOR, TEST_N, N_SYN, POOL_MULT = 200, 10000, 9500, 10000, 3
REF_SHARE = 0.20                  # random share of the data reserved for the reference model eta_hat
W_CLIP_Q, CART_MIN_LEAF, ALPHA_CT = 0.99, 5, 0.05

QUICK  = "--quick" in sys.argv          # --quick: 2 repetitions. Default: the full design.
N_REPS = 2 if QUICK else 20
SEED0  = 2026
TEACHER_N = None                  # sensitivity: 1000 = weaker teacher (trained on 1,000 donor women)
OUT_CSV = f"results_v16b_{DATASET}_{'quick' if QUICK else 'full'}{'' if N_LOCAL == 200 else f'_n{N_LOCAL}'}{'' if TEACHER_N is None else f'_t{TEACHER_N}'}.csv"
OUT_COEF = OUT_CSV.replace('.csv', '_coef.csv')
print(f"{DATASET} QUICK={QUICK} reps={N_REPS} score={SCORE} -> {OUT_CSV}")

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


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

X_ALL = DATA[FEATURES].to_numpy(float)
Y_ALL = DATA.y.to_numpy()
S_IDX = [FEATURES.index(c) for c in SCORE]
expit = lambda u: 1 / (1 + np.exp(-u))

_perm = np.random.default_rng(SEED0).permutation(len(DATA))
REF = _perm[: int(REF_SHARE * len(DATA))]
IN_REF = np.zeros(len(DATA), bool); IN_REF[REF] = True
ETA_HAT = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9, verbosity=0, n_jobs=-1) \
    .fit(X_ALL[REF], Y_ALL[REF]).predict_proba(X_ALL)[:, 1]

HIGH, RULES = {}, {}
for name, v in SHIFTS.items():
    col = DATA[v].to_numpy(float); med = np.median(col[~IN_REF])
    h = col > med
    if h[~IN_REF].mean() < 0.3: h = col >= med
    HIGH[name] = h; RULES[name] = (FEATURES.index(v), med, bool(h[~IN_REF].mean() != (col[~IN_REF] > med).mean()))
    print(f"shift '{name}': share above the median = {h[~IN_REF].mean():.2f}")
_rest = ~IN_REF
_ri = np.flatnonzero(_rest); _h = min(20000, len(_ri) // 2); _tr, _te = _ri[:_h], _ri[_h:2 * _h]
_lrf = LogisticRegression(C=1.0, max_iter=5000).fit(X_ALL[_tr], Y_ALL[_tr])
_lrs = LogisticRegression(C=1.0, max_iter=5000).fit(X_ALL[_tr][:, S_IDX], Y_ALL[_tr])
print(f"held-out log-loss: full logistic {log_loss(Y_ALL[_te], _lrf.predict_proba(X_ALL[_te])[:, 1]):.4f} | "
      f"short score {log_loss(Y_ALL[_te], _lrs.predict_proba(X_ALL[_te][:, S_IDX])[:, 1]):.4f} | "
      f"reference XGBoost {log_loss(Y_ALL[_te], ETA_HAT[_te]):.4f}")

import statsmodels.api as sm
from scipy import stats
from sklearn.tree import DecisionTreeRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, log_loss

COLS = FEATURES + ["y"]

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



# ---------------- v8 additions ----------------
T_NB = [0.10, 0.15, 0.20]          # risk thresholds for net benefit
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

def draw(avail, n, share_high, high, rng):
    nh = int(round(share_high * n))
    idx = np.r_[rng.choice(np.flatnonzero(avail & high), nh, replace=False),
                rng.choice(np.flatnonzero(avail & ~high), n - nh, replace=False)]
    avail[idx] = False
    return rng.permutation(idx)                     # shuffle: draws are built high-then-low, and calibration CV folds must not follow that order

def proj_score(X, p):
    # population I-projection of probabilities p onto the short logistic score
    return fit_lr(X_soft=X[:, S_IDX], p_soft=p, C=1e4)
def _coef_rows(models, Xte, cols, meta):
    sd = Xte.std(0); sd[sd == 0] = 1; mad = np.abs(Xte - Xte.mean(0)).mean(0); out = []
    for name, m in models.items():
        sc, lr = m.named_steps["sc"], m.named_steps["lr"]
        b = lr.coef_[0] / np.where(sc.scale_ == 0, 1, sc.scale_)
        for j, c in enumerate(cols):
            out.append(dict(meta, model=name, feature=c, beta_std=b[j] * sd[j], imp=abs(b[j]) * mad[j]))
    return out


# ---------------- v16: soft vs hard labels, real covariates, lack of support ----------------
SUPPS = [1.0, 0.9, 0.8]                                   # donor support: no donor patient above this quantile of BMI
_jb = FEATURES.index("BMI")
SUPP_OK = {q: (np.ones(len(X_ALL), bool) if q >= 1.0 else X_ALL[:, _jb] <= np.quantile(X_ALL[:, _jb], q)) for q in SUPPS}

def fit_lr_ws(X, p, w, C=1.0):
    # weighted logistic regression on soft labels: each row enters twice, with weights w*p (y=1) and w*(1-p) (y=0)
    m = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(C=C, max_iter=5000))])
    return m.fit(np.vstack([X, X]), np.r_[np.ones(len(p)), np.zeros(len(p))], lr__sample_weight=np.r_[w * p, w * (1 - p)])

def teacher_oof(X, y, seed, k=3):
    # out-of-fold teacher probabilities on the donor's own patients (in-sample probabilities would leak the outcome)
    out = np.empty(len(y)); fold = np.random.default_rng(seed + 77).integers(0, k, len(y))
    for f in range(k):
        m = teacher_fit(X[fold != f], y[fold != f], seed + 31 * f); out[fold == f] = m.predict_proba(X[fold == f])[:, 1]
    return out

def support_diag(pool, Xloc, clf):
    lo, hi = pool.min(0), pool.max(0)
    out_rng = float(np.mean(np.any((Xloc < lo) | (Xloc > hi), axis=1)))
    p_loc = clf.predict_proba(_tilt_features(Xloc))[:, 1]
    return out_rng, float(np.mean(p_loc > 0.99))


COLUMNS = ["dataset", "shift", "share", "supp", "rep", "arm", "logloss_y", "brier_y", "auc", "ex_kl_ref", "ct_choice", "ess_C", "cal_slope", "citl", "nb10", "nb15", "nb20", "est_G", "est_Gupd_n", "ll_teacher_loc", "ll_score_loc", "ess_don", "p_better", "d_mean", "est_null"]

def done_keys():
    if not os.path.exists(OUT_CSV): return set()
    r = pd.read_csv(OUT_CSV); return set(zip(r["shift"], r["share"], r["supp"], r["rep"]))

def run_cell(shift, sk, rep, q=1.0):
    t0 = time.time()
    seed = SEED0 + 1000 * rep + 100 * list(SHIFTS).index(shift) + 10 * SKEWS.index(sk)
    rng = np.random.default_rng(seed)
    high = HIGH[shift]
    avail = ~IN_REF.copy()
    te, loc, rpop = draw(avail, TEST_N, sk, high, rng), draw(avail, N_LOCAL, sk, high, rng), draw(avail, POP_N, sk, high, rng)
    _av = avail & SUPP_OK[q]                                         # v16: donor support truncated at the q-quantile of BMI
    don, dpop = draw(_av, N_DONOR, 1 - sk, high, rng), draw(_av, POP_N, 1 - sk, high, rng); avail[don] = False; avail[dpop] = False
    X, y, S = X_ALL, Y_ALL, S_IDX; SIDX = S
    res = {}

    res["local_score"] = fit_lr(X[loc][:, S], y[loc]).predict_proba(X[te][:, S])[:, 1]
    star, dstar = proj_score(X[rpop], ETA_HAT[rpop]), proj_score(X[dpop], ETA_HAT[dpop])
    res["oracle_star"] = star.predict_proba(X[te][:, S])[:, 1]
    res["oracle_D"] = dstar.predict_proba(X[te][:, S])[:, 1]
    rc = fit_lr(X_soft=dstar.decision_function(X[rpop][:, S])[:, None], p_soft=ETA_HAT[rpop], C=1e4)
    res["oracle_D_recal"] = rc.predict_proba(dstar.decision_function(X[te][:, S])[:, None])[:, 1]

    hD = fit_lr(X[don][:, S], y[don])                                  # the donor publishes its short score
    lp_loc, lp_te = hD.decision_function(X[loc][:, S]), hD.decision_function(X[te][:, S])
    res["share_score"] = expit(lp_te)
    res["share_score_recal"] = recal(lp_loc, y[loc], lp_te)
    res["share_score_ct"], ct = closed_test(lp_loc, y[loc], lp_te)
    lpf = lambda Z: dstar.decision_function(Z[:, S])
    extra = {"oracle_D_ct_n": ct_finite(lpf, X[rpop], ETA_HAT[rpop], lpf(X[te]), ETA_HAT[te], N_LOCAL, B_CT, rng)}

    _tsub = don if TEACHER_N is None else np.random.default_rng(seed + 5).choice(don, TEACHER_N, replace=False)
    t = teacher_fit(X[_tsub], y[_tsub], seed)                               # teacher sees all predictors
    res["teacher_full"] = t.predict_proba(X[te])[:, 1]
    ddf = pd.DataFrame(X[don], columns=FEATURES).assign(y=y[don])
    pool = cart_synth(ddf, POOL_MULT * N_DONOR, seed)[FEATURES].to_numpy(float)
    p_pool = t.predict_proba(pool)[:, 1]
    clf = ratio_classifier(pool, X[loc])                              # targeting uses all predictors held locally
    idxC, ess = resample_w(est_weights(clf, pool), N_SYN, rng)
    idxE, _ = resample_w(exact_weights(pool, sk, RULES[shift]), N_SYN, rng)
    hIW = fit_lr_w(X[don][:, S], y[don], est_weights(clf, X[don]))
    res["IW_donor"] = hIW.predict_proba(X[te][:, S])[:, 1]
    res["IW_donor_ct"], _ = closed_test(hIW.decision_function(X[loc][:, S]), y[loc], hIW.decision_function(X[te][:, S]))
    res["IW_donor_exact"] = fit_lr_w(X[don][:, S], y[don], exact_weights(X[don], sk, RULES[shift])).predict_proba(X[te][:, S])[:, 1]
    res["C_exact_ratio"] = fit_lr(X[loc][:, S], y[loc], pool[idxE][:, S], p_pool[idxE]).predict_proba(X[te][:, S])[:, 1]
    # ---- v9: what the receiver can estimate BEFORE choosing (no access to the truth) ----
    Xt, pt = pool[idxC], p_pool[idxC]                                   # targeted pool with teacher soft labels
    h_hat = fit_lr(X_soft=Xt[:, S], p_soft=pt, C=1e4)                    # receiver's estimate of its best simple score
    ph = h_hat.predict_proba(Xt[:, S])[:, 1]
    lpD_t = hD.decision_function(Xt[:, S])
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

    B = np.arange(N_SYN)
    mB = fit_lr(X[loc][:, S], y[loc], pool[B][:, S], p_pool[B]); res["B_untargeted"] = mB.predict_proba(X[te][:, S])[:, 1]
    mC = fit_lr(X[loc][:, S], y[loc], pool[idxC][:, S], p_pool[idxC]); res["C_target"] = mC.predict_proba(X[te][:, S])[:, 1]
    res["local_distill"] = fit_lr(X_soft=X[loc][:, S], p_soft=t.predict_proba(X[loc])[:, 1]).predict_proba(X[te][:, S])[:, 1]


    # ---- v16: soft vs hard labels, generation vs real covariates, support diagnostics ----
    ZF = lambda Z: Z[:, S_IDX]
    rng3 = np.random.default_rng(seed + 1616)
    yh = (rng3.random(N_SYN) < p_pool[idxC]).astype(int)
    res["M3_hard"] = fit_lr(np.vstack([ZF(X[loc]), ZF(pool[idxC])]), np.r_[y[loc], yh]).predict_proba(ZF(X[te]))[:, 1]
    p_oof = teacher_oof(X[don], y[don], seed)
    idxR, _ = resample_w(w_d, N_SYN, rng3)
    res["M3_realX"] = fit_lr(ZF(X[loc]), y[loc], ZF(X[don][idxR]), p_oof[idxR]).predict_proba(ZF(X[te]))[:, 1]
    res["IW_KD"] = fit_lr_ws(ZF(X[don]), p_oof, w_d).predict_proba(ZF(X[te]))[:, 1]
    frac_out_range, frac_out_p99 = support_diag(pool, X[loc], clf)

    rows = []
    for arm, p in res.items():
        pc = np.clip(p, 1e-9, 1 - 1e-9)
        rows.append(dict(dataset=DATASET, shift=shift, share=sk, rep=rep, arm=arm,
                         logloss_y=log_loss(y[te], pc, labels=[0, 1]), brier_y=float(np.mean((pc - y[te]) ** 2)),
                         auc=roc_auc_score(y[te], pc), ex_kl_ref=kl_excess(ETA_HAT[te], pc),
                         ct_choice=ct if arm == "share_score_ct" else "", ess_C=ess if arm == "C_target" else np.nan,
                         **dict(zip(["cal_slope", "citl"], cal_stats(y[te], pc))),
                         **{f"nb{int(t*100)}": net_benefit(y[te], pc, t) for t in T_NB}))
    for r_ in rows: r_['supp'] = q
    for arm, v in extra.items():
        rows.append(dict(dataset=DATASET, shift=shift, share=sk, rep=rep, arm=arm, ex_kl_ref=v))
    rows.append(dict(dataset=DATASET, shift=shift, share=sk, rep=rep, arm="decision_inputs", est_G=est_G, est_Gupd_n=est_Gupd_n, ll_teacher_loc=ll_teacher_loc, ll_score_loc=ll_score_loc, ess_don=ess_don, p_better=p_better, d_mean=d_mean, est_null=est_null, frac_out_range=frac_out_range, frac_out_p99=frac_out_p99))
    _cr = _coef_rows({"shared": hD, "M2_untargeted": mB, "M3_targeted": mC, "IW": hIW, "oracle": star}, X[te][:, S], SCORE, dict(dataset=DATASET, shift=shift, share=sk, supp=q, rep=rep))
    pd.DataFrame(_cr).to_csv(OUT_COEF, mode="a", header=not os.path.exists(OUT_COEF), index=False)
    for r_ in rows: r_['supp'] = q
    pd.DataFrame(rows).reindex(columns=COLUMNS + ['frac_out_range', 'frac_out_p99']).to_csv(OUT_CSV, mode="a", header=not os.path.exists(OUT_CSV), index=False)
    print(f"  {shift:26s} share {sk:.1f} supp {q:.1f} rep {rep:2d}  {time.time() - t0:5.1f}s")

t_start = time.time()
for rep in range(N_REPS):
    _done = done_keys()
    for shift, sk, q in SCEN:
        if (shift, sk, q, rep) not in _done: run_cell(shift, sk, rep, q)
print(f"Finished in {(time.time() - t_start) / 60:.1f} min -> {OUT_CSV}")


import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score as _auc
R = pd.read_csv(OUT_CSV)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
CELL = ["shift", "share"] + ["supp"]; M, MR = "logloss_y", "ex_kl_ref"; CT, STAR, LOC = "share_score_ct", "oracle_star", "local_score"
P = R.pivot_table(index=CELL + ["rep"], columns="arm", values=M) * 1000
PR = R.pivot_table(index=CELL + ["rep"], columns="arm", values=MR) * 1000
I = R[R.arm == "decision_inputs"].set_index(CELL + ["rep"])
J = pd.DataFrame({"M0": P[LOC], "M1": P[CT], "M2": P["B_untargeted"], "IW": P["IW_donor"], "IW_KD": P["IW_KD"],
                  "M3": P["C_target"], "M3_hard": P["M3_hard"], "M3_realX": P["M3_realX"],
                  "pred_n": PR["oracle_D_ct_n"] - PR[STAR]}).join(I[["d_mean", "ess_don", "frac_out_range", "frac_out_p99"]])
J["M4"] = np.where(J.d_mean < 0, np.where(J.ess_don < 0.40, J.M3, J.IW), J.M1)            # frozen rule R1
key = J.index.droplevel(["supp", "rep"])
full = J.xs(1.0, level="supp").groupby(level=["shift", "share"]).pred_n.mean()
_fd = full.to_dict(); J["gain_cell"] = np.array([_fd.get(tuple(k), -np.inf) >= 2 for k in key])   # defined at full support
ARMS = ["M0", "M2", "IW", "IW_KD", "M3_hard", "M3_realX", "M3", "M4"]
G = J[ARMS].rsub(J.M1, axis=0)
print("Gain over M1 (x1000; positive is better), by donor support q and by whether a gain exists at full support")
T = G.groupby([J.index.get_level_values("supp"), J.gain_cell.values]).mean().T
T.columns = [f"q={a:.1f} {'gain' if b else 'no-gain'}" for a, b in T.columns]; display(T.round(2))

print("\nPer scenario (gain over M1)"); display(G.join(J.gain_cell).groupby(level=CELL).mean().round(1))

gc = J.gain_cell.values; q1 = (J.index.get_level_values("supp") == 1.0)
def mg(a, mask): return G.loc[mask, a].mean()
S1 = bool(mg("M3", gc) > mg("M3_hard", gc))
S2 = bool(abs((J.M3 - J.M3_realX)[q1].mean()) < 2)
S3 = bool(mg("M3", gc) > mg("IW_KD", gc))
g10, g08 = mg("M3", gc & q1), mg("M3", gc & (J.index.get_level_values("supp") == 0.8))
S4 = bool(g08 < 0.5 * g10)
lab = (~q1).astype(int)
auc_rng, auc_p99 = _auc(lab, J.frac_out_range), _auc(lab, J.frac_out_p99)
S5 = bool(auc_rng > 0.8)
print(f"\nM3 gain in gain scenarios: q=1.0 {g10:.1f}, q=0.8 {g08:.1f}")
print(f"support diagnostics: AUC(frac_out_range) {auc_rng:.2f}, AUC(frac_out_p99) {auc_p99:.2f}")
display(J.groupby(level="supp")[["frac_out_range", "frac_out_p99", "ess_don"]].describe().T.round(3))
print(f"\nS1 soft > hard labels: {S1}\nS2 generation costs < 2 units (full support): {S2}\nS3 M3 > IW_KD: {S3}\nS4 lack of support halves M3's gain: {S4}\nS5 frac_out_range detects lack of support (AUC > 0.8): {S5}")

# regret and the exploratory support check
BEST = J[["M1", "IW", "M3"]].min(axis=1)
rows = {"R1 frozen (M4)": J.M4, "always M1": J.M1, "always IW": J.IW, "always M3": J.M3}
for tau in [0.01, 0.02, 0.05]:
    for fb in ["M1", "IW"]:
        rows[f"R1 + support check tau={tau} -> {fb}"] = np.where(J.frac_out_range > tau, np.where(J.M4 == J.M3, J[fb], J.M4), J.M4)
TR = pd.DataFrame({k: (pd.Series(v, index=J.index) - BEST).groupby(level="supp").mean() for k, v in rows.items()}).T
print("\nMean regret against best of {M1, IW, M3} (x1000), by donor support (support-check rows are EXPLORATORY)"); display(TR.round(2))

fig, ax = plt.subplots(figsize=(7, 4))
for a, mk in [("M3", "o"), ("M3_hard", "s"), ("M3_realX", "^"), ("IW_KD", "v"), ("IW", "D"), ("M4", "*")]:
    ax.plot(SUPPS, [mg(a, gc & (J.index.get_level_values("supp") == q)) for q in SUPPS], marker=mk, label=a)
ax.axhline(0, color="#999", lw=0.8); ax.invert_xaxis(); ax.set_xlabel("donor support q (no donor patient above this BMI quantile)")
ax.set_ylabel("mean gain over M1 in gain scenarios (x1000)"); ax.legend(fontsize=8); plt.tight_layout()
plt.savefig(OUT_CSV.replace(".csv", "_support.png"), dpi=150); plt.show()
