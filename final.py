import ast
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent
TRAIN_PATH  = ROOT / "train.csv"
TEST_PATH   = ROOT / "test.csv"
SAMPLE_PATH = ROOT / "sample_submission.csv"
# ───────────────────────────────────────────────────────────────────────────

CRITICAL_CODES = {
    "18", "24", "20", "26", "19", "2016", "2014", "2017",
    "29", "404", "2024", "2025", "90", "59", "2021", "2015",
    "13", "306", "2022", "2023",
}
MINOR_CODES = {
    "55", "205", "304", "40", "201", "904", "58", "43",
    "102", "35", "33", "31", "307", "32", "303", "903",
    "34", "42", "901", "367",
}

CAT_COLS = ["Facility Type", "Risk", "State", "Inspection Type"]

NUM_COLS = [
    "Zip", "Latitude", "Longitude", "year", "month", "weekday",
    "viol_count", "unique_viol_count", "has_any_violation",
    "critical_viol_count", "minor_viol_count", "has_critical_viol",
    "critical_ratio", "minor_ratio",
    "month_sin", "month_cos", "weekday_sin", "weekday_cos",
    "is_reinspection", "risk_ordinal", "years_since_2010",
    "viol_passrate_mean", "viol_passrate_min",
    "viol_count_sq", "viol_density",
    "te_City", "te_Zip", "te_FacilityType",
    "te_InspectionType", "te_geo_cluster",
]

# ── LightGBM native configs (compatible with all versions ≥3.x) ───────────
LGB_BASE = dict(
    objective          = "binary",
    metric             = "binary_logloss",
    boosting_type      = "gbdt",
    num_leaves         = 127,
    min_data_in_leaf   = 20,
    feature_fraction   = 0.8,
    bagging_fraction   = 0.8,
    bagging_freq       = 1,
    lambda_l1          = 0.1,
    lambda_l2          = 1.0,
    learning_rate      = 0.03,
    is_unbalance       = True,
    n_jobs             = -1,
    verbose            = -1,
    seed               = 42,
)

LGB_CONFIGS = [
    # lgb1 — deep, regularised
    dict(**LGB_BASE, num_leaves=127, learning_rate=0.03,
         feature_fraction=0.8, bagging_fraction=0.8,
         lambda_l1=0.1, lambda_l2=1.0, seed=42),
    # lgb2 — shallower, faster convergence
    dict(**LGB_BASE, num_leaves=63, learning_rate=0.05,
         feature_fraction=0.7, bagging_fraction=0.7,
         lambda_l1=0.5, lambda_l2=2.0, seed=7),
    # lgb3 — very deep, strong regularisation
    dict(**LGB_BASE, num_leaves=255, learning_rate=0.02,
         feature_fraction=0.9, bagging_fraction=0.9,
         lambda_l1=0.05, lambda_l2=0.5, seed=99),
]

# Early stopping: train up to 3000 rounds, stop if no improvement for 80
MAX_ROUNDS    = 3000
EARLY_STOP    = 80
VALID_FRAC    = 0.1   # internal val split for early stopping

# ExtraTrees weight in blend
ET_WEIGHT    = 0.08
LGB_WEIGHTS  = [0.42, 0.30, 0.20]   # must sum to 1 - ET_WEIGHT
MODEL_WEIGHTS = LGB_WEIGHTS + [ET_WEIGHT]


# ══════════════════════════════════════════════════════════════════
# PARSING
# ══════════════════════════════════════════════════════════════════

def parse_violation_list(value):
    if pd.isna(value):
        return []
    text = str(value).strip()
    if text in {"", "[]", "nan", "None"}:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(v).strip() for v in parsed if str(v).strip()]
    except Exception:
        pass
    cleaned = text.strip("[]")
    if not cleaned:
        return []
    parts = [p.strip().strip("'").strip('"') for p in cleaned.split(",")]
    return [p for p in parts if p]


def canonical_code(code):
    s = str(code).strip()
    if s == "":
        return s
    return str(int(s)) if s.isdigit() else s


# ══════════════════════════════════════════════════════════════════
# BASE FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════

def add_base_features(df):
    out = df.copy()

    out["viol_codes"] = out["Violation_List"].apply(parse_violation_list)
    out["viol_codes"] = out["viol_codes"].apply(
        lambda arr: [canonical_code(x) for x in arr if canonical_code(x) != ""]
    )

    out["viol_count"]        = out["viol_codes"].apply(len)
    out["unique_viol_count"] = out["viol_codes"].apply(lambda x: len(set(x)))
    out["has_any_violation"] = (out["viol_count"] > 0).astype(int)

    # Non-linear violation count features
    out["viol_count_sq"]  = out["viol_count"] ** 2
    out["viol_density"]   = out["unique_viol_count"] / (out["viol_count"] + 1)

    out["critical_viol_count"] = out["viol_codes"].apply(
        lambda x: sum(1 for c in x if c in CRITICAL_CODES)
    )
    out["minor_viol_count"] = out["viol_codes"].apply(
        lambda x: sum(1 for c in x if c in MINOR_CODES)
    )
    out["has_critical_viol"] = (out["critical_viol_count"] > 0).astype(int)
    out["critical_ratio"]    = out["critical_viol_count"] / (out["viol_count"] + 1)
    out["minor_ratio"]       = out["minor_viol_count"]    / (out["viol_count"] + 1)

    out["City"]            = out["City"].fillna("UNKNOWN").astype(str).str.upper().str.strip()
    out["Facility Type"]   = out["Facility Type"].fillna("UNKNOWN").astype(str).str.strip()
    out["Risk"]            = out["Risk"].fillna("UNKNOWN").astype(str).str.strip()
    out["Inspection Type"] = out["Inspection Type"].fillna("UNKNOWN").astype(str).str.strip()
    out["State"]           = out["State"].fillna("UNKNOWN").astype(str).str.strip()

    for col in ["Zip", "Latitude", "Longitude", "year", "month", "weekday"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["month_sin"]   = np.sin(2 * np.pi * out["month"]   / 12)
    out["month_cos"]   = np.cos(2 * np.pi * out["month"]   / 12)
    out["weekday_sin"] = np.sin(2 * np.pi * out["weekday"] / 7)
    out["weekday_cos"] = np.cos(2 * np.pi * out["weekday"] / 7)

    reinspect_kws = ["re-inspection", "reinspection", "re inspection"]
    out["is_reinspection"] = out["Inspection Type"].str.lower().apply(
        lambda x: int(any(kw in x for kw in reinspect_kws))
    )

    risk_map = {
        "Risk 1 (High)": 3, "Risk 2 (Medium)": 2,
        "Risk 3 (Low)": 1, "All": 0, "UNKNOWN": 2,
    }
    out["risk_ordinal"]     = out["Risk"].map(risk_map).fillna(2).astype(int)
    out["years_since_2010"] = (out["year"] - 2010).clip(0, 20)

    return out


# ══════════════════════════════════════════════════════════════════
# GEO CLUSTERING
# ══════════════════════════════════════════════════════════════════

def fit_geo_clusters(train_df, n_clusters=80):
    coords  = train_df[["Latitude", "Longitude"]].copy()
    med_lat = coords["Latitude"].median()
    med_lon = coords["Longitude"].median()
    coords  = coords.fillna({"Latitude": med_lat, "Longitude": med_lon})
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=5)
    km.fit(coords)
    return km, med_lat, med_lon


def apply_geo_clusters(df, km, med_lat, med_lon):
    coords = df[["Latitude", "Longitude"]].fillna(
        {"Latitude": med_lat, "Longitude": med_lon}
    )
    return km.predict(coords)


# ══════════════════════════════════════════════════════════════════
# TARGET ENCODING (smoothed, fold-safe)
# ══════════════════════════════════════════════════════════════════

def target_encode_col(train_df, test_df, col, target_col, smoothing=30):
    global_mean = train_df[target_col].mean()
    stats = train_df.groupby(col)[target_col].agg(["mean", "count"])
    stats["smoothed"] = (
        stats["mean"] * stats["count"] + global_mean * smoothing
    ) / (stats["count"] + smoothing)
    encode_map = stats["smoothed"].to_dict()
    return (
        train_df[col].map(encode_map).fillna(global_mean).values,
        test_df[col].map(encode_map).fillna(global_mean).values,
    )


# ══════════════════════════════════════════════════════════════════
# VIOLATION CODE PASS-RATE SCORES
# ══════════════════════════════════════════════════════════════════

def build_code_passrate_map(train_df, min_count=30, smoothing=30):
    global_mean = train_df["target"].mean()
    exploded = (
        train_df[["target", "viol_codes"]]
        .explode("viol_codes")
        .dropna(subset=["viol_codes"])
    )
    exploded = exploded[exploded["viol_codes"] != ""]
    stats = exploded.groupby("viol_codes")["target"].agg(["mean", "count"])
    stats = stats[stats["count"] >= min_count]
    stats["smoothed"] = (
        stats["mean"] * stats["count"] + global_mean * smoothing
    ) / (stats["count"] + smoothing)
    return stats["smoothed"].to_dict(), global_mean


def viol_score_mean(codes, code_map, gm):
    return float(np.mean([code_map.get(c, gm) for c in codes])) if codes else gm


def viol_score_min(codes, code_map, gm):
    return float(np.min([code_map.get(c, gm) for c in codes])) if codes else gm


# ══════════════════════════════════════════════════════════════════
# VIOLATION BAG-OF-WORDS
# ══════════════════════════════════════════════════════════════════

def build_violation_dicts(series, keep_codes):
    keep = set(keep_codes)
    dicts = []
    for codes in series:
        d = {}
        for c in codes:
            if c in keep:
                d[f"v_{c}"] = d.get(f"v_{c}", 0) + 1
        dicts.append(d)
    return dicts


# ══════════════════════════════════════════════════════════════════
# FEATURE MATRIX BUILDER (leak-free)
# ══════════════════════════════════════════════════════════════════

def build_feature_matrix(X_tr_df, X_va_df, y_tr, top_codes, km, med_lat, med_lon):
    X_tr = X_tr_df.copy()
    X_va = X_va_df.copy()

    X_tr["geo_cluster"] = apply_geo_clusters(X_tr, km, med_lat, med_lon)
    X_va["geo_cluster"] = apply_geo_clusters(X_va, km, med_lat, med_lon)

    tr_rates           = X_tr.copy()
    tr_rates["target"] = y_tr.values
    code_map, gm       = build_code_passrate_map(tr_rates)

    for df in [X_tr, X_va]:
        df["viol_passrate_mean"] = df["viol_codes"].apply(lambda c: viol_score_mean(c, code_map, gm))
        df["viol_passrate_min"]  = df["viol_codes"].apply(lambda c: viol_score_min(c, code_map, gm))

    tr_te           = X_tr.copy()
    tr_te["target"] = y_tr.values
    for col, te_name in [
        ("City",           "te_City"),
        ("Zip",            "te_Zip"),
        ("Facility Type",  "te_FacilityType"),
        ("Inspection Type","te_InspectionType"),
        ("geo_cluster",    "te_geo_cluster"),
    ]:
        tr_enc, va_enc = target_encode_col(tr_te, X_va, col, "target")
        X_tr[te_name]  = tr_enc
        X_va[te_name]  = va_enc

    d_tr  = build_violation_dicts(X_tr["viol_codes"], top_codes)
    d_va  = build_violation_dicts(X_va["viol_codes"], top_codes)
    vec   = DictVectorizer(sparse=True)
    Xv_tr = vec.fit_transform(d_tr)
    Xv_va = vec.transform(d_va)

    num_cols_present = [c for c in NUM_COLS if c in X_tr.columns]
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), CAT_COLS),
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler(with_mean=False)),
            ]), num_cols_present),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )
    Xp_tr = pre.fit_transform(X_tr)
    Xp_va = pre.transform(X_va)

    return (
        hstack([Xp_tr, Xv_tr]).tocsr(),
        hstack([Xp_va, Xv_va]).tocsr(),
        pre, vec, code_map, gm,
    )


# ══════════════════════════════════════════════════════════════════
# LGBM TRAINING WITH EARLY STOPPING
# ══════════════════════════════════════════════════════════════════

def train_lgbm_with_early_stopping(params, X_tr, y_tr, X_va, y_va):
    """
    Trains LightGBM using native API with early stopping.
    Returns (booster, best_n_rounds).
    """
    # Hold out a small internal val set from training data for early stopping
    n_val = int(len(y_tr) * VALID_FRAC)
    idx   = np.random.RandomState(params["seed"]).permutation(len(y_tr))
    val_idx, trn_idx = idx[:n_val], idx[n_val:]

    dtrain = lgb.Dataset(X_tr[trn_idx], label=y_tr.iloc[trn_idx] if hasattr(y_tr, "iloc") else y_tr[trn_idx])
    dval   = lgb.Dataset(X_tr[val_idx], label=y_tr.iloc[val_idx] if hasattr(y_tr, "iloc") else y_tr[val_idx],
                         reference=dtrain)

    callbacks = [
        lgb.early_stopping(EARLY_STOP, verbose=False),
        lgb.log_evaluation(period=-1),
    ]

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=MAX_ROUNDS,
        valid_sets=[dval],
        callbacks=callbacks,
    )
    return booster, booster.best_iteration


def predict_lgbm(booster, X):
    return booster.predict(X, num_iteration=booster.best_iteration)


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  Improved Food Inspection Model — LightGBM Ensemble v3")
    print("=" * 65)

    print("\n[1/6] Loading data...")
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    print(f"  Train: {train.shape}  |  Test: {test.shape}")

    print("\n[2/6] Engineering features...")
    train = add_base_features(train)
    test  = add_base_features(test)
    y     = train["target"].astype(int)
    print(f"  Target: pass={y.sum()} ({y.mean()*100:.1f}%)  "
          f"fail={(~y.astype(bool)).sum()} ({(1-y.mean())*100:.1f}%)")

    top_codes = (
        train["viol_codes"].explode().dropna().astype(str)
        .value_counts().head(300).index.tolist()
    )
    print(f"  Violation code features: {len(top_codes)}")

    print("\n[3/6] Fitting geo clusters (80 clusters)...")
    km, med_lat, med_lon = fit_geo_clusters(train, n_clusters=80)

    keep_cols = [
        "Facility Type", "Risk", "City", "State", "Zip",
        "Inspection Type", "Latitude", "Longitude",
        "year", "month", "weekday", "viol_codes",
        "viol_count", "unique_viol_count", "has_any_violation",
        "critical_viol_count", "minor_viol_count", "has_critical_viol",
        "critical_ratio", "minor_ratio",
        "month_sin", "month_cos", "weekday_sin", "weekday_cos",
        "is_reinspection", "risk_ordinal", "years_since_2010",
        "viol_count_sq", "viol_density",
    ]
    X      = train[keep_cols].copy()
    X_test = test[keep_cols].copy()

    # ── 5-Fold CV ────────────────────────────────────────────────
    print("\n[4/6] Cross-validating (5-fold, LightGBM with early stopping)...")
    skf        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    n_lgb      = len(LGB_CONFIGS)
    n_models   = n_lgb + 1          # +1 for ET
    oof_probs  = np.zeros((len(X), n_models))
    fold_f1s   = []
    best_iters = [[] for _ in range(n_lgb)]  # track early stopping rounds

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_va = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        Xall_tr, Xall_va, _, _, _, _ = build_feature_matrix(
            X_tr, X_va, y_tr, top_codes, km, med_lat, med_lon
        )

        fold_probs_va = []

        # LightGBM models
        for i, params in enumerate(LGB_CONFIGS):
            booster, best_it = train_lgbm_with_early_stopping(
                params, Xall_tr, y_tr, Xall_va, y_va
            )
            best_iters[i].append(best_it)
            prob = predict_lgbm(booster, Xall_va)
            oof_probs[va_idx, i] = prob
            fold_probs_va.append(prob)

        # ExtraTrees
        et = ExtraTreesClassifier(
            n_estimators=400, max_depth=None, min_samples_leaf=5,
            max_features=0.3, class_weight="balanced",
            n_jobs=-1, random_state=42,
        )
        et.fit(Xall_tr.toarray(), y_tr)
        et_prob = et.predict_proba(Xall_va.toarray())[:, 1]
        oof_probs[va_idx, n_lgb] = et_prob
        fold_probs_va.append(et_prob)

        fold_blend = sum(w * p for w, p in zip(MODEL_WEIGHTS, fold_probs_va))
        fold_pred  = (fold_blend >= 0.5).astype(int)
        fold_f1    = f1_score(y_va, fold_pred)
        fold_f1s.append(fold_f1)

        best_it_str = " | ".join(
            f"lgb{i+1}:{best_iters[i][-1]}" for i in range(n_lgb)
        )
        print(f"  Fold {fold} | F1: {fold_f1:.5f} | "
              f"Features: {Xall_tr.shape[1]} | {best_it_str}")

    cv_mean = np.mean(fold_f1s)
    cv_std  = np.std(fold_f1s)
    print(f"\n  CV F1: {cv_mean:.5f} ± {cv_std:.5f}")

    # Best n_estimators per model (mean of early stopping rounds + buffer)
    final_rounds = [
        min(int(np.mean(best_iters[i]) * 1.1) + 50, MAX_ROUNDS)
        for i in range(n_lgb)
    ]
    print(f"  Final training rounds: { {f'lgb{i+1}': r for i, r in enumerate(final_rounds)} }")

    # ── Threshold search ─────────────────────────────────────────
    print("\n[5/6] Optimizing decision threshold (800 steps)...")
    oof_blend = sum(w * oof_probs[:, i] for i, w in enumerate(MODEL_WEIGHTS))
    best_thr, best_oof_f1 = 0.5, 0.0
    for thr in np.linspace(0.1, 0.9, 801):
        f = f1_score(y, (oof_blend >= thr).astype(int))
        if f > best_oof_f1:
            best_oof_f1 = f
            best_thr    = thr
    print(f"  Best threshold: {best_thr:.4f}  |  OOF F1: {best_oof_f1:.5f}")

    # ── Stacking meta-learner (optional 2nd layer) ───────────────
    print("  Training stacking meta-learner on OOF probs...")
    meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta.fit(oof_probs, y)
    meta_oof = meta.predict_proba(oof_probs)[:, 1]
    best_meta_thr, best_meta_f1 = 0.5, 0.0
    for thr in np.linspace(0.1, 0.9, 801):
        f = f1_score(y, (meta_oof >= thr).astype(int))
        if f > best_meta_f1:
            best_meta_f1 = f
            best_meta_thr = thr
    print(f"  Meta-learner OOF F1: {best_meta_f1:.5f} (thr={best_meta_thr:.4f})")
    use_meta = best_meta_f1 > best_oof_f1
    print(f"  Using {'meta-learner' if use_meta else 'weighted blend'} for final predictions")

    # ── Final models on full data ─────────────────────────────────
    print("\n[6/6] Training final models on full data...")
    train_full           = X.copy()
    train_full["target"] = y.values
    train_full["geo_cluster"] = apply_geo_clusters(train_full, km, med_lat, med_lon)

    code_map_full, gm_full = build_code_passrate_map(train_full)
    X_test_full = X_test.copy()
    X_test_full["geo_cluster"] = apply_geo_clusters(X_test_full, km, med_lat, med_lon)

    for df in [train_full, X_test_full]:
        df["viol_passrate_mean"] = df["viol_codes"].apply(lambda c: viol_score_mean(c, code_map_full, gm_full))
        df["viol_passrate_min"]  = df["viol_codes"].apply(lambda c: viol_score_min(c, code_map_full, gm_full))

    for col, te_name in [
        ("City",           "te_City"),
        ("Zip",            "te_Zip"),
        ("Facility Type",  "te_FacilityType"),
        ("Inspection Type","te_InspectionType"),
        ("geo_cluster",    "te_geo_cluster"),
    ]:
        tr_enc, te_enc       = target_encode_col(train_full, X_test_full, col, "target")
        train_full[te_name]  = tr_enc
        X_test_full[te_name] = te_enc

    d_full    = build_violation_dicts(train_full["viol_codes"], top_codes)
    d_test    = build_violation_dicts(X_test_full["viol_codes"], top_codes)
    vec_final = DictVectorizer(sparse=True)
    Xv_full   = vec_final.fit_transform(d_full)
    Xv_test   = vec_final.transform(d_test)

    num_cols_present = [c for c in NUM_COLS if c in train_full.columns]
    pre_final = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), CAT_COLS),
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler(with_mean=False)),
            ]), num_cols_present),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )
    Xp_full   = pre_final.fit_transform(train_full)
    Xp_test   = pre_final.transform(X_test_full)
    Xfull     = hstack([Xp_full, Xv_full]).tocsr()
    Xtest_mat = hstack([Xp_test, Xv_test]).tocsr()
    print(f"  Final feature matrix: {Xfull.shape}")

    test_probs = np.zeros((len(X_test), n_models))

    # LightGBM final
    for i, (params, n_rounds) in enumerate(zip(LGB_CONFIGS, final_rounds)):
        print(f"  Fitting final lgb{i+1} ({n_rounds} rounds)...")
        dtrain  = lgb.Dataset(Xfull, label=y.values)
        booster = lgb.train(
            params, dtrain,
            num_boost_round=n_rounds,
            callbacks=[lgb.log_evaluation(period=-1)],
        )
        test_probs[:, i] = booster.predict(Xtest_mat)

    # ExtraTrees final
    print("  Fitting final ExtraTrees...")
    et_final = ExtraTreesClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=5,
        max_features=0.3, class_weight="balanced",
        n_jobs=-1, random_state=42,
    )
    et_final.fit(Xfull.toarray(), y)
    test_probs[:, n_lgb] = et_final.predict_proba(Xtest_mat.toarray())[:, 1]

    # Predict using best method
    if use_meta:
        test_blend = meta.predict_proba(test_probs)[:, 1]
        test_pred  = (test_blend >= best_meta_thr).astype(int)
        used_thr   = best_meta_thr
    else:
        test_blend = sum(w * test_probs[:, i] for i, w in enumerate(MODEL_WEIGHTS))
        test_pred  = (test_blend >= best_thr).astype(int)
        used_thr   = best_thr

    # ── Save outputs ─────────────────────────────────────────────
    submission = pd.DataFrame({"id": test["id"].astype(int), "target": test_pred})
    if SAMPLE_PATH.exists():
        sample     = pd.read_csv(SAMPLE_PATH)
        submission = sample[["id"]].merge(submission, on="id", how="left")

    out_path = ROOT / "submission_improved.csv"
    submission.to_csv(out_path, index=False)

    report = {
        "cv_mean_f1":          float(cv_mean),
        "cv_std_f1":           float(cv_std),
        "fold_f1s":            [float(f) for f in fold_f1s],
        "best_threshold_oof":  float(best_thr),
        "best_oof_f1":         float(best_oof_f1),
        "meta_oof_f1":         float(best_meta_f1),
        "used_meta":           bool(use_meta),
        "final_threshold":     float(used_thr),
        "n_features":          int(Xfull.shape[1]),
        "final_rounds":        final_rounds,
        "model_weights":       MODEL_WEIGHTS,
    }
    with open(ROOT / "model_report_improved.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 65)
    print(f"  CV F1:          {cv_mean:.5f} ± {cv_std:.5f}")
    print(f"  OOF F1 (blend): {best_oof_f1:.5f}  (threshold={best_thr:.4f})")
    print(f"  OOF F1 (meta):  {best_meta_f1:.5f}  (threshold={best_meta_thr:.4f})")
    print(f"  Features used:  {Xfull.shape[1]}")
    print(f"  Test pass rate: {test_pred.mean()*100:.1f}%")
    print(f"  Output:         submission_improved.csv")
    print("=" * 65)


if __name__ == "__main__":
    main()