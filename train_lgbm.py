import ast
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import f1_score
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "train.csv"
TEST_PATH = ROOT / "test.csv"
SAMPLE_PATH = ROOT / "sample_submission.csv"


def parse_violation_list(value):
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
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
    if s.isdigit():
        return str(int(s))
    return s


def add_engineered_features(df):
    out = df.copy()
    out["viol_codes"] = out["Violation_List"].apply(parse_violation_list)
    out["viol_codes"] = out["viol_codes"].apply(lambda arr: [canonical_code(x) for x in arr if canonical_code(x) != ""])
    out["viol_count"] = out["viol_codes"].apply(len)
    out["unique_viol_count"] = out["viol_codes"].apply(lambda x: len(set(x)))
    out["has_any_violation"] = (out["viol_count"] > 0).astype(int)
    out["City"] = out["City"].fillna("UNKNOWN").astype(str).str.upper().str.strip()
    out["Facility Type"] = out["Facility Type"].fillna("UNKNOWN").astype(str).str.strip()
    out["Risk"] = out["Risk"].fillna("UNKNOWN").astype(str).str.strip()
    out["Inspection Type"] = out["Inspection Type"].fillna("UNKNOWN").astype(str).str.strip()
    out["State"] = out["State"].fillna("UNKNOWN").astype(str).str.strip()
    out["Zip"] = pd.to_numeric(out["Zip"], errors="coerce")
    out["Latitude"] = pd.to_numeric(out["Latitude"], errors="coerce")
    out["Longitude"] = pd.to_numeric(out["Longitude"], errors="coerce")
    out["year"] = pd.to_numeric(out["year"], errors="coerce")
    out["month"] = pd.to_numeric(out["month"], errors="coerce")
    out["weekday"] = pd.to_numeric(out["weekday"], errors="coerce")
    return out


def build_violation_dicts(series, keep_codes):
    keep = set(keep_codes)
    dicts = []
    for codes in series:
        d = {}
        for c in codes:
            if c in keep:
                d[f"viol_{c}"] = 1
        dicts.append(d)
    return dicts


def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    train = add_engineered_features(train)
    test = add_engineered_features(test)
    y = train["target"].astype(int)

    top_codes = train["viol_codes"].explode().dropna().astype(str).value_counts().head(100).index.tolist()
    train["viol_dict"] = build_violation_dicts(train["viol_codes"], top_codes)
    test["viol_dict"] = build_violation_dicts(test["viol_codes"], top_codes)

    feature_cols = [
        "Facility Type", "Risk", "City", "State", "Inspection Type",
        "Zip", "Latitude", "Longitude", "year", "month", "weekday",
        "viol_count", "unique_viol_count", "has_any_violation", "viol_dict",
    ]

    X = train[feature_cols].copy()
    X_test = test[feature_cols].copy()

    cat_cols = ["Facility Type", "Risk", "City", "State", "Inspection Type"]
    num_cols = ["Zip", "Latitude", "Longitude", "year", "month", "weekday", "viol_count", "unique_viol_count", "has_any_violation"]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    importances = {}

    vec = DictVectorizer(sparse=True)

    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        X_tr = X.iloc[tr_idx].copy()
        X_va = X.iloc[va_idx].copy()
        y_tr = y.iloc[tr_idx]
        y_va = y.iloc[va_idx]

        d_tr = X_tr.pop("viol_dict")
        d_va = X_va.pop("viol_dict")

        Xv_tr = vec.fit_transform(d_tr) if fold == 1 else vec.transform(d_tr)
        Xv_va = vec.transform(d_va)

        pre = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler(with_mean=False))]), num_cols),
            ],
            remainder="drop",
            sparse_threshold=1.0,
        )

        Xp_tr = pre.fit_transform(X_tr)
        Xp_va = pre.transform(X_va)

        from scipy.sparse import hstack
        Xall_tr = hstack([Xp_tr, Xv_tr]).tocsr()
        Xall_va = hstack([Xp_va, Xv_va]).tocsr()

        clf = lgb.LGBMClassifier(
            objective='binary',
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )

        clf.fit(Xall_tr, y_tr)

        pred = clf.predict(Xall_va)
        score = f1_score(y_va, pred)
        fold_scores.append(score)
        print(f"Fold {fold} F1: {score:.5f}")

        oof_preds[va_idx] = clf.predict_proba(Xall_va)[:, 1]

    mean_f1 = np.mean(fold_scores)
    std_f1 = np.std(fold_scores)
    print(f"CV mean F1: {mean_f1:.5f}, std: {std_f1:.5f}")

    # Train final model on full data
    d_all = X.pop("viol_dict")
    d_test = X_test.pop("viol_dict")
    Xv_all = vec.fit_transform(d_all)
    Xv_test = vec.transform(d_test)

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler(with_mean=False))]), num_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )

    Xp_all = pre.fit_transform(X)
    Xp_test = pre.transform(X_test)
    from scipy.sparse import hstack
    Xfull = hstack([Xp_all, Xv_all]).tocsr()
    Xtest_full = hstack([Xp_test, Xv_test]).tocsr()

    final = lgb.LGBMClassifier(objective='binary', n_estimators=2000, learning_rate=0.03, num_leaves=64, random_state=42, n_jobs=-1, class_weight='balanced')
    final.fit(Xfull, y)
    test_pred_prob = final.predict_proba(Xtest_full)[:, 1]

    # Choose threshold by maximizing F1 on OOF
    best_thr = 0.5
    best_f1 = 0
    for thr in np.linspace(0.3, 0.7, 41):
        f = f1_score((oof_preds >= thr).astype(int), y)
        if f > best_f1:
            best_f1 = f
            best_thr = thr
    print(f"Best threshold from OOF maximizing F1: {best_thr:.3f} (OOF F1 {best_f1:.5f})")

    test_pred = (test_pred_prob >= best_thr).astype(int)
    submission = pd.DataFrame({"id": test["id"].astype(int), "target": test_pred})
    sample = pd.read_csv(SAMPLE_PATH)
    submission = sample[["id"]].merge(submission, on="id", how="left")
    submission.to_csv(ROOT / "submission_lgbm.csv", index=False)

    report = {
        "cv_mean_f1": float(mean_f1),
        "cv_std_f1": float(std_f1),
        "best_threshold_oof": float(best_thr),
        "best_oof_f1": float(best_f1),
    }
    with open(ROOT / "model_report_lgbm.json", "w") as f:
        json.dump(report, f, indent=2)

    print("Created submission_lgbm.csv and model_report_lgbm.json")


if __name__ == '__main__':
    main()
