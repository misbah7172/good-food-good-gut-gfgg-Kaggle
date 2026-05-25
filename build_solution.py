import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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

    # Try strict literal parsing first.
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(v).strip() for v in parsed if str(v).strip()]
    except Exception:
        pass

    # Fallback: tolerate malformed brackets/quotes.
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


def cross_val_f1(X_model, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for tr_idx, va_idx in skf.split(X_model, y):
        X_tr = X_model.iloc[tr_idx].copy()
        X_va = X_model.iloc[va_idx].copy()
        y_tr = y.iloc[tr_idx]
        y_va = y.iloc[va_idx]

        # Fit vectorizer only on train fold to avoid leakage.
        vdict_tr = X_tr.pop("viol_dict")
        vdict_va = X_va.pop("viol_dict")

        viol_vec = DictVectorizer(sparse=True)
        Xv_tr = viol_vec.fit_transform(vdict_tr)
        Xv_va = viol_vec.transform(vdict_va)

        cat_cols = ["Facility Type", "Risk", "City", "State", "Inspection Type"]
        num_cols = ["Zip", "Latitude", "Longitude", "year", "month", "weekday", "viol_count", "unique_viol_count", "has_any_violation"]

        pre = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
                (
                    "num",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]),
                    num_cols,
                ),
            ],
            remainder="drop",
            sparse_threshold=1.0,
        )

        Xp_tr = pre.fit_transform(X_tr)
        Xp_va = pre.transform(X_va)

        from scipy.sparse import hstack

        Xall_tr = hstack([Xp_tr, Xv_tr]).tocsr()
        Xall_va = hstack([Xp_va, Xv_va]).tocsr()

        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            n_jobs=None,
            solver="liblinear",
            random_state=42,
        )
        clf.fit(Xall_tr, y_tr)

        pred = clf.predict(Xall_va)
        scores.append(f1_score(y_va, pred))

    return float(np.mean(scores)), float(np.std(scores))


def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    train = add_engineered_features(train)
    test = add_engineered_features(test)

    y = train["target"].astype(int)

    # Keep frequent violation codes to stabilize sparse feature space.
    all_codes = train["viol_codes"].explode().dropna().astype(str)
    top_codes = all_codes.value_counts().head(80).index.tolist()

    train["viol_dict"] = build_violation_dicts(train["viol_codes"], top_codes)
    test["viol_dict"] = build_violation_dicts(test["viol_codes"], top_codes)

    model_cols = [
        "Facility Type",
        "Risk",
        "City",
        "State",
        "Inspection Type",
        "Zip",
        "Latitude",
        "Longitude",
        "year",
        "month",
        "weekday",
        "viol_count",
        "unique_viol_count",
        "has_any_violation",
        "viol_dict",
    ]

    X_model = train[model_cols].copy()
    cv_mean, cv_std = cross_val_f1(X_model, y)

    # Train final model on full training data.
    X_tr = train[model_cols].copy()
    X_te = test[model_cols].copy()

    vdict_tr = X_tr.pop("viol_dict")
    vdict_te = X_te.pop("viol_dict")

    viol_vec = DictVectorizer(sparse=True)
    Xv_tr = viol_vec.fit_transform(vdict_tr)
    Xv_te = viol_vec.transform(vdict_te)

    cat_cols = ["Facility Type", "Risk", "City", "State", "Inspection Type"]
    num_cols = ["Zip", "Latitude", "Longitude", "year", "month", "weekday", "viol_count", "unique_viol_count", "has_any_violation"]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False)),
                ]),
                num_cols,
            ),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )

    Xp_tr = pre.fit_transform(X_tr)
    Xp_te = pre.transform(X_te)

    from scipy.sparse import hstack

    Xall_tr = hstack([Xp_tr, Xv_tr]).tocsr()
    Xall_te = hstack([Xp_te, Xv_te]).tocsr()

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )
    clf.fit(Xall_tr, y)
    test_pred = clf.predict(Xall_te)

    sample = pd.read_csv(SAMPLE_PATH)
    submission = pd.DataFrame({"id": test["id"].astype(int), "target": test_pred.astype(int)})

    # Preserve sample ordering if needed.
    if "id" in sample.columns:
        submission = sample[["id"]].merge(submission, on="id", how="left")
    submission.to_csv(ROOT / "submission.csv", index=False)

    # EDA answers for required questions.
    failed = train[train["target"] == 0].copy()

    fail_codes = failed["viol_codes"].explode().dropna().astype(str)
    top10_fail_codes = fail_codes.value_counts().head(10)

    fac_code_counts = (
        train.explode("viol_codes")
        .dropna(subset=["viol_codes"])
        .groupby("Facility Type")["viol_codes"]
        .nunique()
        .sort_values(ascending=False)
    )

    risk_fail_rate = train.groupby("Risk")["target"].apply(lambda s: (s == 0).mean()).sort_values(ascending=False)

    month_fail = train.groupby("month")["target"].apply(lambda s: (s == 0).mean()).sort_values(ascending=False)
    weekday_fail = train.groupby("weekday")["target"].apply(lambda s: (s == 0).mean()).sort_values(ascending=False)

    city_fail = (
        train.groupby("City")["target"]
        .agg(count="size", fail_rate=lambda s: (s == 0).mean())
        .sort_values(["fail_rate", "count"], ascending=[False, False])
    )

    zip_fail = (
        train.groupby("Zip")["target"]
        .agg(count="size", fail_rate=lambda s: (s == 0).mean())
        .sort_values(["fail_rate", "count"], ascending=[False, False])
    )

    viol_count_vs_fail = (
        train.groupby("viol_count")["target"]
        .agg(count="size", fail_rate=lambda s: (s == 0).mean())
        .sort_index()
    )

    common_by_facility = (
        train.explode("viol_codes")
        .dropna(subset=["viol_codes"])
        .groupby(["Facility Type", "viol_codes"])
        .size()
        .reset_index(name="count")
        .sort_values(["Facility Type", "count"], ascending=[True, False])
    )

    geo_fail = train[["Latitude", "Longitude", "target"]].copy()
    geo_fail["fail"] = (geo_fail["target"] == 0).astype(int)

    insp_fail = (
        train.groupby("Inspection Type")["target"]
        .agg(count="size", fail_rate=lambda s: (s == 0).mean())
        .sort_values(["fail_rate", "count"], ascending=[False, False])
    )

    # Code-level fail association using simple fail-rate uplift over baseline.
    exploded = train.explode("viol_codes").dropna(subset=["viol_codes"]).copy()
    base_fail = (train["target"] == 0).mean()
    code_stats = (
        exploded.groupby("viol_codes")["target"]
        .agg(count="size", fail_rate=lambda s: (s == 0).mean())
        .assign(fail_uplift=lambda d: d["fail_rate"] - base_fail)
        .sort_values(["fail_uplift", "count"], ascending=[False, False])
    )

    answers = {
        "dataset": {
            "train_shape": list(train.shape),
            "test_shape": list(test.shape),
            "target_distribution": train["target"].value_counts().to_dict(),
            "cv_f1_mean": round(cv_mean, 5),
            "cv_f1_std": round(cv_std, 5),
        },
        "q1_top_10_violation_codes_associated_with_failures": top10_fail_codes.to_dict(),
        "q2_facility_type_with_highest_unique_violation_codes": {
            "facility_type": fac_code_counts.index[0] if len(fac_code_counts) else None,
            "unique_violation_codes": int(fac_code_counts.iloc[0]) if len(fac_code_counts) else 0,
        },
        "q3_risk_category_highest_failure_rate": {
            "risk": risk_fail_rate.index[0] if len(risk_fail_rate) else None,
            "failure_rate": float(risk_fail_rate.iloc[0]) if len(risk_fail_rate) else None,
        },
        "q4_months_weekdays_most_failed": {
            "month_highest_failure_rate": int(month_fail.index[0]) if len(month_fail) else None,
            "month_failure_rate": float(month_fail.iloc[0]) if len(month_fail) else None,
            "weekday_highest_failure_rate": int(weekday_fail.index[0]) if len(weekday_fail) else None,
            "weekday_failure_rate": float(weekday_fail.iloc[0]) if len(weekday_fail) else None,
        },
        "q5_outcomes_across_city_zip": {
            "top_cities_by_failure_rate_min30": city_fail[city_fail["count"] >= 30].head(10).reset_index().to_dict(orient="records"),
            "top_zips_by_failure_rate_min30": zip_fail[zip_fail["count"] >= 30].head(10).reset_index().to_dict(orient="records"),
        },
        "q6_more_violations_more_failure": {
            "correlation_violcount_fail": float(np.corrcoef(train["viol_count"], (train["target"] == 0).astype(int))[0, 1]),
            "fail_rate_by_viol_count_head": viol_count_vs_fail.head(12).reset_index().to_dict(orient="records"),
        },
        "q7_common_violations_by_facility_type_top3_each": (
            common_by_facility.groupby("Facility Type").head(3).to_dict(orient="records")
        ),
        "q8_geographic_trends": {
            "mean_lat_lon_failed": failed[["Latitude", "Longitude"]].mean().to_dict(),
            "mean_lat_lon_passed": train[train["target"] == 1][["Latitude", "Longitude"]].mean().to_dict(),
        },
        "q9_inspection_type_vs_outcome_top10": insp_fail.head(10).reset_index().to_dict(orient="records"),
        "q10_strongest_failure_indicator_codes_top10": code_stats.head(10).reset_index().to_dict(orient="records"),
    }

    with open(ROOT / "eda_answers.json", "w", encoding="utf-8") as f:
        json.dump(answers, f, indent=2)

    with open(ROOT / "model_report.txt", "w", encoding="utf-8") as f:
        f.write(f"5-fold CV F1 mean: {cv_mean:.5f}\n")
        f.write(f"5-fold CV F1 std: {cv_std:.5f}\n")
        f.write(f"Train rows: {len(train)}\n")
        f.write(f"Test rows: {len(test)}\n")
        f.write("Top 10 fail-associated codes (frequency among failed inspections):\n")
        for code, cnt in top10_fail_codes.items():
            f.write(f"  - {code}: {cnt}\n")

    print("Done. Files created: submission.csv, eda_answers.json, model_report.txt")


if __name__ == "__main__":
    main()
