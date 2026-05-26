import ast
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from scipy.stats import entropy as scipy_entropy
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

ROOT        = Path(__file__).resolve().parent
TRAIN_PATH  = ROOT / "train.csv"
TEST_PATH   = ROOT / "test.csv"
SAMPLE_PATH = ROOT / "sample_submission.csv"

# Empirically derived critical codes (lowest pass rate in training data)
CRITICAL_CODES = {
    "18", "24", "20", "26", "19", "2016", "2014", "2017",
    "29", "404", "2024", "2025", "90", "59", "2021", "2015",
    "13", "306", "2022", "2023", "14", "15", "16", "17",
    "21", "22", "23", "25", "27", "28",
}
MINOR_CODES = {
    "55", "205", "304", "40", "201", "904", "58", "43",
    "102", "35", "33", "31", "307", "32", "303", "903",
    "34", "42", "901", "367", "36", "37", "38", "39",
    "41", "47", "48", "49", "51", "53", "56", "57",
}

OUT_OF_BUSINESS_KWS = ["out of business", "not located", "illegal operation", "business not located"]
NO_ENTRY_KWS        = ["no entry", "not ready", "non-inspection", "no-entry"]
FOOD_POISON_KWS     = ["food poison", "sfp"]
TASK_FORCE_KWS      = ["task force", "liquor 147", "package liquor"]

CAT_COLS = ["Facility Type", "Risk", "State", "Inspection Type"]

NUM_COLS = [
    "Zip", "Latitude", "Longitude", "year", "month", "weekday",
    "viol_count", "unique_viol_count", "has_any_violation",
    "critical_viol_count", "minor_viol_count", "has_critical_viol",
    "critical_ratio", "minor_ratio",
    "month_sin", "month_cos", "weekday_sin", "weekday_cos",
    "is_reinspection", "is_canvass_reinspection",
    "is_complaint_reinspection", "is_license_reinspection",
    "is_out_of_business", "is_no_entry", "is_food_poison", "is_task_force",
    "risk_ordinal", "years_since_2010", "is_recent_year", "season",
    "viol_count_sq", "viol_density", "viol_entropy", "max_code_freq",
    "viol_severity_score", "viol_weighted_sum",
    "zero_viol_x_reinspection",
    "viol_passrate_mean", "viol_passrate_min", "viol_passrate_std",
    # Facility history features (most important)
    "facility_hist_pass_rate", "facility_hist_count",
    "facility_last_result", "facility_consecutive_passes",
    "facility_recent_fail_rate", "facility_inspection_number",
    "facility_months_since_last",
    # Target encoded
    "te_City", "te_Zip", "te_FacilityType", "te_InspectionType",
    "te_geo_cluster", "te_FacilityRisk", "te_InspectionRisk",
    "te_location",
]

# LightGBM configs — 5 diverse models
LGB_CONFIGS = [
    dict(objective="binary", metric="binary_logloss", boosting_type="gbdt",
         num_leaves=127, min_data_in_leaf=20, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1,
         lambda_l1=0.1, lambda_l2=1.0, learning_rate=0.03,
         is_unbalance=True, n_jobs=-1, verbose=-1, seed=42),
    dict(objective="binary", metric="binary_logloss", boosting_type="gbdt",
         num_leaves=63, min_data_in_leaf=30, feature_fraction=0.7,
         bagging_fraction=0.7, bagging_freq=1,
         lambda_l1=0.5, lambda_l2=2.0, learning_rate=0.05,
         is_unbalance=True, n_jobs=-1, verbose=-1, seed=7),
    dict(objective="binary", metric="binary_logloss", boosting_type="gbdt",
         num_leaves=255, min_data_in_leaf=15, feature_fraction=0.9,
         bagging_fraction=0.9, bagging_freq=1,
         lambda_l1=0.05, lambda_l2=0.5, learning_rate=0.02,
         is_unbalance=True, n_jobs=-1, verbose=-1, seed=99),
    dict(objective="binary", metric="binary_logloss", boosting_type="goss",
         num_leaves=127, min_data_in_leaf=20, feature_fraction=0.8,
         lambda_l1=0.2, lambda_l2=1.5, learning_rate=0.03,
         top_rate=0.2, other_rate=0.1,
         is_unbalance=True, n_jobs=-1, verbose=-1, seed=13),
    dict(objective="binary", metric="binary_logloss", boosting_type="gbdt",
         num_leaves=200, min_data_in_leaf=25, feature_fraction=0.75,
         bagging_fraction=0.85, bagging_freq=1,
         lambda_l1=0.3, lambda_l2=0.8, learning_rate=0.025,
         min_gain_to_split=0.01,
         is_unbalance=True, n_jobs=-1, verbose=-1, seed=2024),
]
MODEL_WEIGHTS = [0.30, 0.20, 0.18, 0.15, 0.12, 0.05]  # 5 lgb + ET
MAX_ROUNDS    = 3000
EARLY_STOP    = 100
VALID_FRAC    = 0.1


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
# VIOLATION SEVERITY SCORE
# ══════════════════════════════════════════════════════════════════

def code_severity(code):
    """Return severity weight: 3=critical, 2=serious, 1=minor, 0.5=unknown."""
    try:
        n = int(code)
        if n <= 14:   return 3.0   # Critical
        if n <= 29:   return 2.0   # Serious
        if n <= 100:  return 1.0   # Minor
        return 0.5                 # Special codes
    except:
        # Non-numeric codes (like '2016', '404') — check critical list
        if code in CRITICAL_CODES:
            return 3.0
        if code in MINOR_CODES:
            return 1.0
        return 1.5


# ══════════════════════════════════════════════════════════════════
# BASE FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════

def _flag(text_lower, kws):
    return int(any(kw in text_lower for kw in kws))


def add_base_features(df):
    out = df.copy()

    out["viol_codes"] = out["Violation_List"].apply(parse_violation_list)
    out["viol_codes"] = out["viol_codes"].apply(
        lambda arr: [canonical_code(x) for x in arr if canonical_code(x) != ""]
    )

    # Count features
    out["viol_count"]        = out["viol_codes"].apply(len)
    out["unique_viol_count"] = out["viol_codes"].apply(lambda x: len(set(x)))
    out["has_any_violation"] = (out["viol_count"] > 0).astype(int)
    out["viol_count_sq"]     = out["viol_count"] ** 2
    out["viol_density"]      = out["unique_viol_count"] / (out["viol_count"] + 1)

    def code_entropy(codes):
        if not codes: return 0.0
        counts = np.array(list(pd.Series(codes).value_counts()))
        probs  = counts / counts.sum()
        return float(scipy_entropy(probs))

    out["viol_entropy"]  = out["viol_codes"].apply(code_entropy)
    out["max_code_freq"] = out["viol_codes"].apply(
        lambda c: int(pd.Series(c).value_counts().iloc[0]) if c else 0
    )

    # Severity tiers
    out["critical_viol_count"] = out["viol_codes"].apply(
        lambda x: sum(1 for c in x if c in CRITICAL_CODES)
    )
    out["minor_viol_count"] = out["viol_codes"].apply(
        lambda x: sum(1 for c in x if c in MINOR_CODES)
    )
    out["has_critical_viol"]  = (out["critical_viol_count"] > 0).astype(int)
    out["critical_ratio"]     = out["critical_viol_count"] / (out["viol_count"] + 1)
    out["minor_ratio"]        = out["minor_viol_count"]    / (out["viol_count"] + 1)

    # Severity score (weighted sum & mean)
    out["viol_severity_score"] = out["viol_codes"].apply(
        lambda c: float(np.mean([code_severity(x) for x in c])) if c else 0.0
    )
    out["viol_weighted_sum"] = out["viol_codes"].apply(
        lambda c: float(sum(code_severity(x) for x in c))
    )

    # Categorical cleaning
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
    out["season"]      = ((out["month"].fillna(1) - 1) // 3).astype(int)

    out["years_since_2010"] = (out["year"] - 2010).clip(0, 20)
    out["is_recent_year"]   = (out["year"] >= 2023).astype(int)

    # Inspection type flags
    insp_lower = out["Inspection Type"].str.lower()
    out["is_reinspection"]           = insp_lower.apply(lambda x: _flag(x, ["re-inspection", "reinspection", "re inspection"]))
    out["is_canvass_reinspection"]   = insp_lower.apply(lambda x: _flag(x, ["canvass re-inspection", "canvass re inspection"]))
    out["is_complaint_reinspection"] = insp_lower.apply(lambda x: _flag(x, ["complaint re-inspection", "complaint re inspection"]))
    out["is_license_reinspection"]   = insp_lower.apply(lambda x: _flag(x, ["license re-inspection", "license re inspection"]))
    out["is_out_of_business"]        = insp_lower.apply(lambda x: _flag(x, OUT_OF_BUSINESS_KWS))
    out["is_no_entry"]               = insp_lower.apply(lambda x: _flag(x, NO_ENTRY_KWS))
    out["is_food_poison"]            = insp_lower.apply(lambda x: _flag(x, FOOD_POISON_KWS))
    out["is_task_force"]             = insp_lower.apply(lambda x: _flag(x, TASK_FORCE_KWS))
    out["zero_viol_x_reinspection"]  = ((out["viol_count"] == 0) & (out["is_reinspection"] == 1)).astype(int)

    # Risk ordinal
    risk_map = {"Risk 1 (High)": 3, "Risk 2 (Medium)": 2, "Risk 3 (Low)": 1, "All": 0, "UNKNOWN": 2}
    out["risk_ordinal"] = out["Risk"].map(risk_map).fillna(2).astype(int)

    # Facility location key (lat/lon rounded to 4dp = ~11m precision)
    out["loc_key"] = (
        out["Latitude"].round(4).astype(str) + "_" +
        out["Longitude"].round(4).astype(str)
    )

    # Time index for ordering inspections (year * 12 + month)
    out["time_idx"] = out["year"].fillna(2015) * 12 + out["month"].fillna(6)

    # Interaction strings for target encoding
    out["FacilityRisk"]   = out["Facility Type"] + "_" + out["Risk"]
    out["InspectionRisk"] = out["Inspection Type"] + "_" + out["Risk"]

    return out


# ══════════════════════════════════════════════════════════════════
# FACILITY HISTORY FEATURES  ← THE BIG NEW FEATURE
# ══════════════════════════════════════════════════════════════════

def add_facility_history(df_with_target, df_target, smoothing=10):
    """
    For each row, compute the facility's historical pass rate using ONLY
    PAST inspections (chronologically before the current one).
    This avoids label leakage.

    df_with_target: DataFrame with 'target' column (training data)
    df_target: DataFrame to add features to (can be same or test)
    """
    # Sort by time
    df_t = df_with_target.copy()
    df_t = df_t.sort_values("time_idx").reset_index(drop=True)

    # Compute expanding historical stats per location (using train only)
    global_mean = df_t["target"].mean()

    # Group-level stats from full training data (used for test)
    loc_stats = df_t.groupby("loc_key")["target"].agg(["mean", "count", "sum"])
    loc_stats.columns = ["loc_pass_rate", "loc_count", "loc_pass_sum"]

    # Smoothed pass rate
    loc_stats["facility_hist_pass_rate"] = (
        loc_stats["loc_pass_sum"] + global_mean * smoothing
    ) / (loc_stats["loc_count"] + smoothing)
    loc_stats["facility_hist_count"] = loc_stats["loc_count"]

    # Last result per location
    last_result = df_t.groupby("loc_key")["target"].last().rename("facility_last_result")

    # Consecutive pass streak (count from end)
    def consec_passes(series):
        vals = series.values[::-1]  # reverse
        count = 0
        for v in vals:
            if v == 1:
                count += 1
            else:
                break
        return count

    consec = df_t.groupby("loc_key")["target"].apply(consec_passes).rename("facility_consecutive_passes")

    # Recent fail rate (last 2 years of training data)
    max_year = df_t["year"].max()
    recent   = df_t[df_t["year"] >= max_year - 2]
    recent_fail = recent.groupby("loc_key")["target"].agg(
        lambda x: 1 - x.mean()
    ).rename("facility_recent_fail_rate")

    # Last inspection time per location
    last_time = df_t.groupby("loc_key")["time_idx"].max().rename("last_time_idx")

    # Merge all history features
    out = df_target.copy()
    out = out.merge(
        loc_stats[["facility_hist_pass_rate", "facility_hist_count"]],
        on="loc_key", how="left"
    )
    out = out.merge(last_result, on="loc_key", how="left")
    out = out.merge(consec,      on="loc_key", how="left")
    out = out.merge(recent_fail, on="loc_key", how="left")
    out = out.merge(last_time,   on="loc_key", how="left")

    out["facility_hist_pass_rate"]     = out["facility_hist_pass_rate"].fillna(global_mean)
    out["facility_hist_count"]         = out["facility_hist_count"].fillna(0)
    out["facility_last_result"]        = out["facility_last_result"].fillna(global_mean)
    out["facility_consecutive_passes"] = out["facility_consecutive_passes"].fillna(0)
    out["facility_recent_fail_rate"]   = out["facility_recent_fail_rate"].fillna(1 - global_mean)

    # Months since last inspection
    out["facility_months_since_last"] = (
        out["time_idx"] - out["last_time_idx"].fillna(out["time_idx"])
    ).clip(0, 120)

    # Inspection number at this facility
    insp_num = df_t.groupby("loc_key").cumcount() + 1
    df_t["insp_num"] = insp_num
    last_insp_num = df_t.groupby("loc_key")["insp_num"].max().rename("facility_inspection_number")
    out = out.merge(last_insp_num, on="loc_key", how="left")
    out["facility_inspection_number"] = out["facility_inspection_number"].fillna(1)

    return out


def add_facility_history_cv(X_tr_df, X_va_df, y_tr):
    """
    Fold-safe version: fit history on training fold only, apply to val fold.
    """
    train_for_hist = X_tr_df.copy()
    train_for_hist["target"] = y_tr.values

    X_tr_out = add_facility_history(train_for_hist, X_tr_df)
    X_va_out = add_facility_history(train_for_hist, X_va_df)
    return X_tr_out, X_va_out


# ══════════════════════════════════════════════════════════════════
# GEO CLUSTERING
# ══════════════════════════════════════════════════════════════════

def fit_geo_clusters(train_df, n_clusters=100):
    coords  = train_df[["Latitude", "Longitude"]].copy()
    med_lat = coords["Latitude"].median()
    med_lon = coords["Longitude"].median()
    coords  = coords.fillna({"Latitude": med_lat, "Longitude": med_lon})
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(coords)
    return km, med_lat, med_lon


def apply_geo_clusters(df, km, med_lat, med_lon):
    coords = df[["Latitude", "Longitude"]].fillna({"Latitude": med_lat, "Longitude": med_lon})
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
# VIOLATION PASS-RATE SCORES
# ══════════════════════════════════════════════════════════════════

def build_code_passrate_map(train_df, min_count=20, smoothing=30):
    global_mean = train_df["target"].mean()
    exploded = train_df[["target","viol_codes"]].explode("viol_codes").dropna(subset=["viol_codes"])
    exploded = exploded[exploded["viol_codes"] != ""]
    stats = exploded.groupby("viol_codes")["target"].agg(["mean","count"])
    stats = stats[stats["count"] >= min_count]
    stats["smoothed"] = (stats["mean"]*stats["count"] + global_mean*smoothing) / (stats["count"]+smoothing)
    return stats["smoothed"].to_dict(), global_mean


def viol_score_mean(codes, code_map, gm):
    return float(np.mean([code_map.get(c, gm) for c in codes])) if codes else gm

def viol_score_min(codes, code_map, gm):
    return float(np.min([code_map.get(c, gm) for c in codes])) if codes else gm

def viol_score_std(codes, code_map, gm):
    return float(np.std([code_map.get(c, gm) for c in codes])) if len(codes) >= 2 else 0.0


# ══════════════════════════════════════════════════════════════════
# VIOLATION BAG-OF-WORDS
# ══════════════════════════════════════════════════════════════════

def build_violation_dicts(series, keep_codes):
    keep = set(keep_codes)
    return [
        {f"v_{c}": codes.count(c) for c in set(codes) if c in keep}
        for codes in series
    ]


# ══════════════════════════════════════════════════════════════════
# FULL FEATURE MATRIX BUILDER
# ══════════════════════════════════════════════════════════════════

def build_feature_matrix(X_tr_df, X_va_df, y_tr, top_codes, km, med_lat, med_lon):
    # Facility history (fold-safe)
    X_tr, X_va = add_facility_history_cv(X_tr_df, X_va_df, y_tr)

    # Geo clusters
    X_tr["geo_cluster"] = apply_geo_clusters(X_tr, km, med_lat, med_lon)
    X_va["geo_cluster"] = apply_geo_clusters(X_va, km, med_lat, med_lon)

    # Violation pass-rate scores
    tr_rates           = X_tr.copy()
    tr_rates["target"] = y_tr.values
    code_map, gm       = build_code_passrate_map(tr_rates)
    for df in [X_tr, X_va]:
        df["viol_passrate_mean"] = df["viol_codes"].apply(lambda c: viol_score_mean(c, code_map, gm))
        df["viol_passrate_min"]  = df["viol_codes"].apply(lambda c: viol_score_min(c, code_map, gm))
        df["viol_passrate_std"]  = df["viol_codes"].apply(lambda c: viol_score_std(c, code_map, gm))

    # Target encoding
    tr_te           = X_tr.copy()
    tr_te["target"] = y_tr.values
    for col, te_name in [
        ("City",           "te_City"),
        ("Zip",            "te_Zip"),
        ("Facility Type",  "te_FacilityType"),
        ("Inspection Type","te_InspectionType"),
        ("geo_cluster",    "te_geo_cluster"),
        ("FacilityRisk",   "te_FacilityRisk"),
        ("InspectionRisk", "te_InspectionRisk"),
        ("loc_key",        "te_location"),
    ]:
        tr_enc, va_enc = target_encode_col(tr_te, X_va, col, "target")
        X_tr[te_name]  = tr_enc
        X_va[te_name]  = va_enc

    # Violation BOW
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
        remainder="drop", sparse_threshold=1.0,
    )
    Xp_tr = pre.fit_transform(X_tr)
    Xp_va = pre.transform(X_va)

    return (
        hstack([Xp_tr, Xv_tr]).tocsr(),
        hstack([Xp_va, Xv_va]).tocsr(),
        pre, vec, code_map, gm,
    )


# ══════════════════════════════════════════════════════════════════
# LGBM TRAINING
# ══════════════════════════════════════════════════════════════════

def train_lgbm(params, X_tr, y_tr, n_rounds=None):
    y_arr = y_tr.values if hasattr(y_tr, "values") else y_tr
    if n_rounds is None:
        n_val  = int(len(y_arr) * VALID_FRAC)
        rng    = np.random.RandomState(params["seed"])
        idx    = rng.permutation(len(y_arr))
        vi, ti = idx[:n_val], idx[n_val:]
        dtrain = lgb.Dataset(X_tr[ti], label=y_arr[ti])
        dval   = lgb.Dataset(X_tr[vi], label=y_arr[vi], reference=dtrain)
        cbs    = [lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(period=-1)]
        bst    = lgb.train(params, dtrain, num_boost_round=MAX_ROUNDS, valid_sets=[dval], callbacks=cbs)
        return bst, bst.best_iteration
    else:
        dtrain = lgb.Dataset(X_tr, label=y_arr)
        bst    = lgb.train(params, dtrain, num_boost_round=n_rounds, callbacks=[lgb.log_evaluation(period=-1)])
        return bst, n_rounds


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  Food Inspection — WINNING VERSION (target: 0.965-0.970)")
    print("=" * 65)

    print("\n[1/7] Loading data...")
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    print(f"  Train: {train.shape}  |  Test: {test.shape}")

    print("\n[2/7] Engineering features...")
    train = add_base_features(train)
    test  = add_base_features(test)
    y     = train["target"].astype(int) 
    print(f"  Target: pass={y.sum()} ({y.mean()*100:.1f}%)  "
          f"fail={(~y.astype(bool)).sum()} ({(1-y.mean())*100:.1f}%)")

    # Unique locations
    print(f"  Unique facility locations (train): {train['loc_key'].nunique()}")
    print(f"  Unique facility locations (test):  {test['loc_key'].nunique()}")
    test_in_train = test['loc_key'].isin(train['loc_key']).mean()
    print(f"  Test locations in train history:   {test_in_train*100:.1f}%")

    top_codes = (
        train["viol_codes"].explode().dropna().astype(str)
        .value_counts().head(300).index.tolist()
    )
    print(f"  Violation BOW features: {len(top_codes)}")

    print("\n[3/7] Fitting geo clusters (100 clusters)...")
    km, med_lat, med_lon = fit_geo_clusters(train, n_clusters=100)

    keep_cols = [
        "Facility Type", "Risk", "City", "State", "Zip",
        "Inspection Type", "Latitude", "Longitude",
        "year", "month", "weekday", "viol_codes", "loc_key", "time_idx",
        "viol_count", "unique_viol_count", "has_any_violation",
        "critical_viol_count", "minor_viol_count", "has_critical_viol",
        "critical_ratio", "minor_ratio", "viol_severity_score", "viol_weighted_sum",
        "month_sin", "month_cos", "weekday_sin", "weekday_cos",
        "season", "is_recent_year",
        "is_reinspection", "is_canvass_reinspection", "is_complaint_reinspection",
        "is_license_reinspection", "is_out_of_business", "is_no_entry",
        "is_food_poison", "is_task_force", "zero_viol_x_reinspection",
        "risk_ordinal", "years_since_2010",
        "viol_count_sq", "viol_density", "viol_entropy", "max_code_freq",
        "FacilityRisk", "InspectionRisk",
    ]
    X      = train[keep_cols].copy()
    X_test = test[keep_cols].copy()

    # ── 5-Fold CV ────────────────────────────────────────────────
    print("\n[4/7] Cross-validating (5-fold, 5× LightGBM + ExtraTrees)...")
    skf        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    n_lgb      = len(LGB_CONFIGS)
    n_models   = n_lgb + 1
    oof_probs  = np.zeros((len(X), n_models))
    fold_f1s   = []
    best_iters = [[] for _ in range(n_lgb)]

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        X_tr = X.iloc[tr_idx].copy()
        X_va = X.iloc[va_idx].copy()
        y_tr = y.iloc[tr_idx]
        y_va = y.iloc[va_idx]

        Xall_tr, Xall_va, _, _, _, _ = build_feature_matrix(
            X_tr, X_va, y_tr, top_codes, km, med_lat, med_lon
        )

        fold_probs_va = []
        for i, params in enumerate(LGB_CONFIGS):
            bst, best_it = train_lgbm(params, Xall_tr, y_tr)
            best_iters[i].append(best_it)
            prob = bst.predict(Xall_va, num_iteration=bst.best_iteration)
            oof_probs[va_idx, i] = prob
            fold_probs_va.append(prob)

        et = ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=5, max_features=0.3,
            class_weight="balanced", n_jobs=-1, random_state=42,
        )
        et.fit(Xall_tr.toarray(), y_tr)
        et_prob = et.predict_proba(Xall_va.toarray())[:, 1]
        oof_probs[va_idx, n_lgb] = et_prob
        fold_probs_va.append(et_prob)

        fold_blend = sum(w * p for w, p in zip(MODEL_WEIGHTS, fold_probs_va))
        fold_pred  = (fold_blend >= 0.5).astype(int)
        fold_f1    = f1_score(y_va, fold_pred)
        fold_f1s.append(fold_f1)
        iters_str = " | ".join(f"lgb{i+1}:{best_iters[i][-1]}" for i in range(n_lgb))
        print(f"  Fold {fold} | F1: {fold_f1:.5f} | {iters_str}")

    cv_mean = np.mean(fold_f1s)
    cv_std  = np.std(fold_f1s)
    print(f"\n  CV F1: {cv_mean:.5f} ± {cv_std:.5f}")

    final_rounds = [
        min(int(np.mean(best_iters[i]) * 1.1) + 50, MAX_ROUNDS)
        for i in range(n_lgb)
    ]
    print(f"  Final rounds: { {f'lgb{i+1}': r for i, r in enumerate(final_rounds)} }")

    # ── Threshold search ─────────────────────────────────────────
    print("\n[5/7] Optimizing threshold (800 steps)...")
    oof_blend = sum(w * oof_probs[:, i] for i, w in enumerate(MODEL_WEIGHTS))
    best_thr, best_oof_f1 = 0.5, 0.0
    for thr in np.linspace(0.05, 0.95, 901):
        f = f1_score(y, (oof_blend >= thr).astype(int))
        if f > best_oof_f1:
            best_oof_f1 = f
            best_thr    = thr
    print(f"  Blend  → thr={best_thr:.4f}  OOF F1={best_oof_f1:.5f}")

    # ── Stacking ─────────────────────────────────────────────────
    print("\n[6/7] Stacking meta-learner...")
    meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta.fit(oof_probs, y)
    meta_oof = meta.predict_proba(oof_probs)[:, 1]
    best_meta_thr, best_meta_f1 = 0.5, 0.0
    for thr in np.linspace(0.05, 0.95, 901):
        f = f1_score(y, (meta_oof >= thr).astype(int))
        if f > best_meta_f1:
            best_meta_f1 = f
            best_meta_thr = thr
    print(f"  Meta   → thr={best_meta_thr:.4f}  OOF F1={best_meta_f1:.5f}")
    use_meta = best_meta_f1 > best_oof_f1
    print(f"  Using: {'meta-learner' if use_meta else 'weighted blend'}")

    # ── Final models ─────────────────────────────────────────────
    print("\n[7/7] Training final models on full data...")
    train_full           = X.copy()
    train_full["target"] = y.values
    train_full_hist      = add_facility_history(train_full, train_full)
    X_test_hist          = add_facility_history(train_full, X_test)

    train_full_hist["geo_cluster"] = apply_geo_clusters(train_full_hist, km, med_lat, med_lon)
    X_test_hist["geo_cluster"]     = apply_geo_clusters(X_test_hist,     km, med_lat, med_lon)

    code_map_full, gm_full = build_code_passrate_map(train_full)
    for df in [train_full_hist, X_test_hist]:
        df["viol_passrate_mean"] = df["viol_codes"].apply(lambda c: viol_score_mean(c, code_map_full, gm_full))
        df["viol_passrate_min"]  = df["viol_codes"].apply(lambda c: viol_score_min(c, code_map_full, gm_full))
        df["viol_passrate_std"]  = df["viol_codes"].apply(lambda c: viol_score_std(c, code_map_full, gm_full))

    for col, te_name in [
        ("City",           "te_City"),
        ("Zip",            "te_Zip"),
        ("Facility Type",  "te_FacilityType"),
        ("Inspection Type","te_InspectionType"),
        ("geo_cluster",    "te_geo_cluster"),
        ("FacilityRisk",   "te_FacilityRisk"),
        ("InspectionRisk", "te_InspectionRisk"),
        ("loc_key",        "te_location"),
    ]:
        tr_enc, te_enc            = target_encode_col(train_full_hist, X_test_hist, col, "target")
        train_full_hist[te_name]  = tr_enc
        X_test_hist[te_name]      = te_enc

    d_full    = build_violation_dicts(train_full_hist["viol_codes"], top_codes)
    d_test    = build_violation_dicts(X_test_hist["viol_codes"], top_codes)
    vec_final = DictVectorizer(sparse=True)
    Xv_full   = vec_final.fit_transform(d_full)
    Xv_test   = vec_final.transform(d_test)

    num_cols_present = [c for c in NUM_COLS if c in train_full_hist.columns]
    pre_final = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), CAT_COLS),
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler(with_mean=False)),
            ]), num_cols_present),
        ],
        remainder="drop", sparse_threshold=1.0,
    )
    Xp_full   = pre_final.fit_transform(train_full_hist)
    Xp_test   = pre_final.transform(X_test_hist)
    Xfull     = hstack([Xp_full, Xv_full]).tocsr()
    Xtest_mat = hstack([Xp_test, Xv_test]).tocsr()
    print(f"  Final feature matrix: {Xfull.shape}")

    test_probs = np.zeros((len(X_test), n_models))
    for i, (params, n_rounds) in enumerate(zip(LGB_CONFIGS, final_rounds)):
        print(f"  Fitting final lgb{i+1} ({n_rounds} rounds)...")
        bst, _ = train_lgbm(params, Xfull, y, n_rounds=n_rounds)
        test_probs[:, i] = bst.predict(Xtest_mat)

    print("  Fitting final ExtraTrees...")
    et_final = ExtraTreesClassifier(
        n_estimators=400, min_samples_leaf=5, max_features=0.3,
        class_weight="balanced", n_jobs=-1, random_state=42,
    )
    et_final.fit(Xfull.toarray(), y)
    test_probs[:, n_lgb] = et_final.predict_proba(Xtest_mat.toarray())[:, 1]

    if use_meta:
        test_blend = meta.predict_proba(test_probs)[:, 1]
        test_pred  = (test_blend >= best_meta_thr).astype(int)
        used_f1, used_thr = best_meta_f1, best_meta_thr
    else:
        test_blend = sum(w * test_probs[:, i] for i, w in enumerate(MODEL_WEIGHTS))
        test_pred  = (test_blend >= best_thr).astype(int)
        used_f1, used_thr = best_oof_f1, best_thr

    # Post-processing: override near-certain cases with rules
    insp_lower = test["Inspection Type"].str.lower().str.strip()
    # Definite FAIL
    definite_fail = insp_lower.apply(lambda x: _flag(x, OUT_OF_BUSINESS_KWS + NO_ENTRY_KWS))
    test_pred[definite_fail.values == 1] = 0
    overrides = definite_fail.sum()
    if overrides:
        print(f"\n  Post-processing: {overrides} near-certain FAIL overrides applied")

    # Save
    submission = pd.DataFrame({"id": test["id"].astype(int), "target": test_pred})
    if SAMPLE_PATH.exists():
        sample     = pd.read_csv(SAMPLE_PATH)
        submission = sample[["id"]].merge(submission, on="id", how="left")
    submission.to_csv(ROOT / "submission_improved.csv", index=False)

    report = {
        "cv_mean_f1":      float(cv_mean),
        "cv_std_f1":       float(cv_std),
        "fold_f1s":        [float(f) for f in fold_f1s],
        "blend_oof_f1":    float(best_oof_f1),
        "meta_oof_f1":     float(best_meta_f1),
        "used_meta":       bool(use_meta),
        "final_threshold": float(used_thr),
        "n_features":      int(Xfull.shape[1]),
        "final_rounds":    final_rounds,
    }
    with open(ROOT / "model_report_improved.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 65)
    print(f"  CV F1:          {cv_mean:.5f} ± {cv_std:.5f}")
    print(f"  OOF F1 (blend): {best_oof_f1:.5f}  (thr={best_thr:.4f})")
    print(f"  OOF F1 (meta):  {best_meta_f1:.5f}  (thr={best_meta_thr:.4f})")
    print(f"  Features used:  {Xfull.shape[1]}")
    print(f"  Test pass rate: {test_pred.mean()*100:.1f}%")
    print(f"  Output:         submission_improved.csv")
    print("=" * 65)


if __name__ == "__main__":
    main()