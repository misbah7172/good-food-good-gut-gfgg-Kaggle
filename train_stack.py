import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import lightgbm as lgb
from catboost import CatBoostClassifier

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


def oof_target_encode(train_series, y, test_series, n_splits=5, alpha=10, noise_level=0.01):
    # Returns oof_encoded for train, and averaged encodings for test
    global_mean = y.mean()
    n = len(train_series)
    oof = pd.Series(index=train_series.index, dtype=float)
    test_avg = pd.Series(0.0, index=test_series.index)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for tr_idx, va_idx in skf.split(train_series, y):
        tr_vals = train_series.iloc[tr_idx]
        tr_y = y.iloc[tr_idx]
        va_vals = train_series.iloc[va_idx]

        stats = tr_vals.to_frame('val').join(tr_y).groupby('val')['target'].agg(['sum','count'])
        # smoothing
        stats['encoded'] = (stats['sum'] + alpha * global_mean) / (stats['count'] + alpha)
        mapping = stats['encoded'].to_dict()

        # train oof for val
        oof.iloc[va_idx] = va_vals.map(mapping).fillna(global_mean)

        # test fold encoding
        test_enc = test_series.map(mapping).fillna(global_mean)
        test_avg += test_enc / n_splits

    # add small noise to train oof
    oof = oof * (1 + noise_level * np.random.randn(len(oof)))
    return oof, test_avg


def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    train = add_engineered_features(train)
    test = add_engineered_features(test)
    y = train['target'].astype(int)

    # choose categorical columns to target-encode
    te_cols = ['City', 'Zip']
    te_oof = {}
    te_test = {}
    for c in te_cols:
        s = train[c].astype(str)
        s_test = test[c].astype(str)
        oof_enc, test_enc = oof_target_encode(s, y, s_test, n_splits=5, alpha=30, noise_level=0.01)
        te_oof[c + '_te'] = oof_enc
        te_test[c + '_te'] = test_enc

    for k, v in te_oof.items():
        train[k] = v
    for k, v in te_test.items():
        test[k] = v

    # violation sparse features
    top_codes = train['viol_codes'].explode().dropna().astype(str).value_counts().head(100).index.tolist()
    train['viol_dict'] = build_violation_dicts(train['viol_codes'], top_codes)
    test['viol_dict'] = build_violation_dicts(test['viol_codes'], top_codes)

    # features
    base_features = [
        'Facility Type', 'Risk', 'City', 'State', 'Inspection Type',
        'Zip', 'Latitude', 'Longitude', 'year', 'month', 'weekday',
        'viol_count', 'unique_viol_count', 'has_any_violation',
    ]
    # add target-encoded columns
    base_features += [c + '_te' for c in te_cols]

    cat_cols = ['Facility Type', 'Risk', 'City', 'State', 'Inspection Type']
    num_cols = ['Zip', 'Latitude', 'Longitude', 'year', 'month', 'weekday', 'viol_count', 'unique_viol_count', 'has_any_violation'] + [c + '_te' for c in te_cols]

    X = train[base_features].copy()
    X_test = test[base_features].copy()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_cat = np.zeros(len(train))
    oof_lgb = np.zeros(len(train))
    test_cat_preds = np.zeros(len(test))
    test_lgb_preds = np.zeros(len(test))

    vec = DictVectorizer(sparse=True)

    fold = 0
    for tr_idx, va_idx in skf.split(X, y):
        fold += 1
        print(f"Fold {fold}")
        X_tr = X.iloc[tr_idx].copy()
        X_va = X.iloc[va_idx].copy()
        y_tr = y.iloc[tr_idx]
        y_va = y.iloc[va_idx]

        # violation dicts
        d_tr = train['viol_dict'].iloc[tr_idx]
        d_va = train['viol_dict'].iloc[va_idx]
        d_test = test['viol_dict']

        # vectorize violation features per fold
        Xv_tr = vec.fit_transform(d_tr)
        Xv_va = vec.transform(d_va)
        Xv_test = vec.transform(d_test)

        # Preprocess numeric and categorical for LightGBM
        # We'll one-hot low-cardinality categories and leave target-encoded city/zip
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
        ohe_cols = ['Facility Type', 'Risk', 'State', 'Inspection Type']
        Xo_tr = ohe.fit_transform(X_tr[ohe_cols])
        Xo_va = ohe.transform(X_va[ohe_cols])
        Xo_test = ohe.transform(X_test[ohe_cols])

        # scale numerics
        num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler(with_mean=False))])
        Xn_tr = num_pipe.fit_transform(X_tr[num_cols])
        Xn_va = num_pipe.transform(X_va[num_cols])
        Xn_test = num_pipe.transform(X_test[num_cols])

        from scipy.sparse import hstack
        Xlgb_tr = hstack([Xo_tr, Xn_tr, Xv_tr]).tocsr()
        Xlgb_va = hstack([Xo_va, Xn_va, Xv_va]).tocsr()
        Xlgb_test = hstack([Xo_test, Xn_test, Xv_test]).tocsr()

        # LightGBM
        lgbm = lgb.LGBMClassifier(objective='binary', n_estimators=1000, learning_rate=0.04, num_leaves=64, random_state=42, n_jobs=-1, class_weight='balanced')
        lgbm.fit(Xlgb_tr, y_tr)
        oof_lgb[va_idx] = lgbm.predict_proba(Xlgb_va)[:, 1]
        test_lgb_preds += lgbm.predict_proba(Xlgb_test)[:, 1] / skf.n_splits

        # CatBoost: prepare DataFrame with categorical columns and numeric columns, then append violation features
        Xv_tr_dense = Xv_tr.toarray()
        Xv_va_dense = Xv_va.toarray()
        Xv_test_dense = Xv_test.toarray()

        df_cat_tr = pd.concat([
            X_tr[cat_cols].reset_index(drop=True),
            pd.DataFrame(X_tr[num_cols].values, columns=num_cols),
            pd.DataFrame(Xv_tr_dense)
        ], axis=1)
        df_cat_va = pd.concat([
            X_va[cat_cols].reset_index(drop=True),
            pd.DataFrame(X_va[num_cols].values, columns=num_cols),
            pd.DataFrame(Xv_va_dense)
        ], axis=1)
        df_cat_test = pd.concat([
            X_test[cat_cols].reset_index(drop=True),
            pd.DataFrame(X_test[num_cols].values, columns=num_cols),
            pd.DataFrame(Xv_test_dense)
        ], axis=1)

        # categorical feature names for CatBoost
        cat_feature_names = cat_cols

        cat = CatBoostClassifier(iterations=800, learning_rate=0.05, depth=6, random_seed=42, verbose=False, auto_class_weights='Balanced')
        cat.fit(df_cat_tr, y_tr, cat_features=cat_feature_names)
        oof_cat[va_idx] = cat.predict_proba(df_cat_va)[:, 1]
        test_cat_preds += cat.predict_proba(df_cat_test)[:, 1] / skf.n_splits

        print(f" Fold {fold} done: LGB OOF mean {oof_lgb[va_idx].mean():.4f}, CAT OOF mean {oof_cat[va_idx].mean():.4f}")

    # Stacker: use OOF preds as features
    meta_X = np.vstack([oof_lgb, oof_cat]).T
    meta_y = y.values
    # simple logistic meta-model
    meta = LogisticRegression(class_weight='balanced', max_iter=1000)
    meta_scores = []
    # evaluate via CV on meta
    skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, va_idx in skf2.split(meta_X, meta_y):
        meta.fit(meta_X[tr_idx], meta_y[tr_idx])
        p = meta.predict(meta_X[va_idx])
        meta_scores.append(f1_score(meta_y[va_idx], p))
    print(f"Meta CV F1 mean: {np.mean(meta_scores):.5f}, std: {np.std(meta_scores):.5f}")

    # optimize threshold on OOF
    best_thr = 0.5
    best_f1 = 0
    meta.fit(meta_X, meta_y)
    meta_oof_prob = meta.predict_proba(meta_X)[:, 1]
    for thr in np.linspace(0.1, 0.9, 81):
        f = f1_score(meta_y, (meta_oof_prob >= thr).astype(int))
        if f > best_f1:
            best_f1 = f
            best_thr = thr
    print(f"Best OOF threshold for stack: {best_thr:.3f} with F1 {best_f1:.5f}")

    # prepare test meta features
    test_meta_X = np.vstack([test_lgb_preds, test_cat_preds]).T
    test_meta_prob = meta.predict_proba(test_meta_X)[:, 1]
    test_pred = (test_meta_prob >= best_thr).astype(int)

    submission = pd.DataFrame({'id': test['id'].astype(int), 'target': test_pred})
    sample = pd.read_csv(SAMPLE_PATH)
    submission = sample[['id']].merge(submission, on='id', how='left')
    submission.to_csv(ROOT / 'submission_stack.csv', index=False)

    report = {
        'lgb_cv_mean_oof': float(np.mean(oof_lgb)),
        'cat_cv_mean_oof': float(np.mean(oof_cat)),
        'meta_cv_mean_f1': float(np.mean(meta_scores)),
        'meta_oof_best_thr': float(best_thr),
        'meta_oof_best_f1': float(best_f1)
    }
    with open(ROOT / 'model_report_stack.json', 'w') as f:
        json.dump(report, f, indent=2)

    print('Created submission_stack.csv and model_report_stack.json')


if __name__ == '__main__':
    main()
