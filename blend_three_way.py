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


def oof_target_encode(train_series, y, test_series, n_splits=5, alpha=30, noise_level=0.01):
    global_mean = y.mean()
    oof = pd.Series(index=train_series.index, dtype=float)
    test_avg = pd.Series(0.0, index=test_series.index)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for tr_idx, va_idx in skf.split(train_series, y):
        tr_vals = train_series.iloc[tr_idx]
        tr_y = y.iloc[tr_idx]
        va_vals = train_series.iloc[va_idx]
        stats = tr_vals.to_frame('val').join(tr_y).groupby('val')['target'].agg(['sum','count'])
        stats['encoded'] = (stats['sum'] + alpha * global_mean) / (stats['count'] + alpha)
        mapping = stats['encoded'].to_dict()
        oof.iloc[va_idx] = va_vals.map(mapping).fillna(global_mean)
        test_enc = test_series.map(mapping).fillna(global_mean)
        test_avg += test_enc / n_splits
    oof = oof * (1 + noise_level * np.random.randn(len(oof)))
    return oof, test_avg


def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    train = add_engineered_features(train)
    test = add_engineered_features(test)
    y = train['target'].astype(int)

    # prepare features and TE
    feature_cols = [
        'Facility Type', 'Risk', 'City', 'State', 'Inspection Type',
        'Zip', 'Latitude', 'Longitude', 'year', 'month', 'weekday',
        'viol_count', 'unique_viol_count', 'has_any_violation'
    ]
    for c in ['City', 'Zip']:
        s = train[c].astype(str)
        s_test = test[c].astype(str)
        oof_enc, test_enc = oof_target_encode(s, y, s_test, n_splits=5, alpha=30, noise_level=0.01)
        train[c + '_te'] = oof_enc
        test[c + '_te'] = test_enc
        feature_cols.append(c + '_te')

    top_codes = train['viol_codes'].explode().dropna().astype(str).value_counts().head(100).index.tolist()
    train['viol_dict'] = build_violation_dicts(train['viol_codes'], top_codes)
    test['viol_dict'] = build_violation_dicts(test['viol_codes'], top_codes)

    X = train[feature_cols].copy()
    X_test = test[feature_cols].copy()

    cat_cols = ['Facility Type', 'Risk', 'City', 'State', 'Inspection Type']
    num_cols = ['Zip', 'Latitude', 'Longitude', 'year', 'month', 'weekday', 'viol_count', 'unique_viol_count', 'has_any_violation', 'City_te', 'Zip_te']

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # compute original LGBM OOF/test (train_lgbm parameters)
    orig_oof = np.zeros(len(X))
    orig_test = np.zeros(len(test))
    vec = DictVectorizer(sparse=True)
    for tr_idx, va_idx in skf.split(X, y):
        X_tr = X.iloc[tr_idx].copy()
        X_va = X.iloc[va_idx].copy()
        y_tr = y.iloc[tr_idx]

        d_tr = train['viol_dict'].iloc[tr_idx]
        d_va = train['viol_dict'].iloc[va_idx]
        d_test = test['viol_dict']

        Xv_tr = vec.fit_transform(d_tr)
        Xv_va = vec.transform(d_va)
        Xv_test = vec.transform(d_test)

        pre = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
        ohe_cols = ['Facility Type', 'Risk', 'State', 'Inspection Type']
        Xo_tr = pre.fit_transform(X_tr[ohe_cols])
        Xo_va = pre.transform(X_va[ohe_cols])
        Xo_test = pre.transform(X_test[ohe_cols])

        num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler(with_mean=False))])
        Xn_tr = num_pipe.fit_transform(X_tr[num_cols])
        Xn_va = num_pipe.transform(X_va[num_cols])
        Xn_test = num_pipe.transform(X_test[num_cols])

        from scipy.sparse import hstack
        Xall_tr = hstack([Xo_tr, Xn_tr, Xv_tr]).tocsr()
        Xall_va = hstack([Xo_va, Xn_va, Xv_va]).tocsr()
        Xall_test = hstack([Xo_test, Xn_test, Xv_test]).tocsr()

        clf = lgb.LGBMClassifier(objective='binary', n_estimators=2000, learning_rate=0.03, num_leaves=64, random_state=42, n_jobs=-1, class_weight='balanced')
        clf.fit(Xall_tr, y_tr)
        orig_oof[va_idx] = clf.predict_proba(Xall_va)[:, 1]
        orig_test += clf.predict_proba(Xall_test)[:, 1] / skf.n_splits

    # compute tuned LGBM OOF/test
    with open(ROOT / 'model_report_lgbm_optuna.json', 'r') as f:
        tuned = json.load(f)['best_params']
    tuned_oof = np.zeros(len(X))
    tuned_test = np.zeros(len(test))
    vec2 = DictVectorizer(sparse=True)
    for tr_idx, va_idx in skf.split(X, y):
        X_tr = X.iloc[tr_idx].copy()
        X_va = X.iloc[va_idx].copy()
        y_tr = y.iloc[tr_idx]

        d_tr = train['viol_dict'].iloc[tr_idx]
        d_va = train['viol_dict'].iloc[va_idx]
        d_test = test['viol_dict']

        Xv_tr = vec2.fit_transform(d_tr)
        Xv_va = vec2.transform(d_va)
        Xv_test = vec2.transform(d_test)

        pre = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
        ohe_cols = ['Facility Type', 'Risk', 'State', 'Inspection Type']
        Xo_tr = pre.fit_transform(X_tr[ohe_cols])
        Xo_va = pre.transform(X_va[ohe_cols])
        Xo_test = pre.transform(X_test[ohe_cols])

        num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler(with_mean=False))])
        Xn_tr = num_pipe.fit_transform(X_tr[num_cols])
        Xn_va = num_pipe.transform(X_va[num_cols])
        Xn_test = num_pipe.transform(X_test[num_cols])

        from scipy.sparse import hstack
        Xall_tr = hstack([Xo_tr, Xn_tr, Xv_tr]).tocsr()
        Xall_va = hstack([Xo_va, Xn_va, Xv_va]).tocsr()
        Xall_test = hstack([Xo_test, Xn_test, Xv_test]).tocsr()

        params = tuned.copy()
        params.update({'n_jobs': 1})
        clf = lgb.LGBMClassifier(**params)
        clf.fit(Xall_tr, y_tr)
        tuned_oof[va_idx] = clf.predict_proba(Xall_va)[:, 1]
        tuned_test += clf.predict_proba(Xall_test)[:, 1] / skf.n_splits

    # compute stack meta (re-run minimal stack flow to get meta probs)
    oof_lgb = np.zeros(len(X))
    oof_cat = np.zeros(len(X))
    test_lgb = np.zeros(len(test))
    test_cat = np.zeros(len(test))

    vec3 = DictVectorizer(sparse=True)
    for tr_idx, va_idx in skf.split(X, y):
        X_tr = X.iloc[tr_idx].copy()
        X_va = X.iloc[va_idx].copy()
        y_tr = y.iloc[tr_idx]

        d_tr = train['viol_dict'].iloc[tr_idx]
        d_va = train['viol_dict'].iloc[va_idx]
        d_test = test['viol_dict']

        Xv_tr = vec3.fit_transform(d_tr)
        Xv_va = vec3.transform(d_va)
        Xv_test = vec3.transform(d_test)

        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
        ohe_cols = ['Facility Type', 'Risk', 'State', 'Inspection Type']
        Xo_tr = ohe.fit_transform(X_tr[ohe_cols])
        Xo_va = ohe.transform(X_va[ohe_cols])
        Xo_test = ohe.transform(X_test[ohe_cols])

        num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler(with_mean=False))])
        Xn_tr = num_pipe.fit_transform(X_tr[num_cols])
        Xn_va = num_pipe.transform(X_va[num_cols])
        Xn_test = num_pipe.transform(X_test[num_cols])

        from scipy.sparse import hstack
        Xlgb_tr = hstack([Xo_tr, Xn_tr, Xv_tr]).tocsr()
        Xlgb_va = hstack([Xo_va, Xn_va, Xv_va]).tocsr()
        Xlgb_test = hstack([Xo_test, Xn_test, Xv_test]).tocsr()

        lgbm = lgb.LGBMClassifier(objective='binary', n_estimators=1000, learning_rate=0.04, num_leaves=64, random_state=42, n_jobs=-1, class_weight='balanced')
        lgbm.fit(Xlgb_tr, y_tr)
        oof_lgb[va_idx] = lgbm.predict_proba(Xlgb_va)[:, 1]
        test_lgb += lgbm.predict_proba(Xlgb_test)[:, 1] / skf.n_splits

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

        cat = CatBoostClassifier(iterations=800, learning_rate=0.05, depth=6, random_seed=42, verbose=False, auto_class_weights='Balanced')
        cat.fit(df_cat_tr, y_tr, cat_features=cat_cols)
        oof_cat[va_idx] = cat.predict_proba(df_cat_va)[:, 1]
        test_cat += cat.predict_proba(df_cat_test)[:, 1] / skf.n_splits

    meta = LogisticRegression(class_weight='balanced', max_iter=1000)
    meta_X = np.vstack([oof_lgb, oof_cat]).T
    meta.fit(meta_X, y.values)
    meta_oof = meta.predict_proba(meta_X)[:, 1]
    test_meta = meta.predict_proba(np.vstack([test_lgb, test_cat]).T)[:, 1]

    # search coarse 21x21 grid for weights w_orig, w_tuned, w_meta = 1 - w1 - w2
    best = {'w_orig': 0, 'w_tuned': 0, 'w_meta': 1, 'thr': 0.5, 'f1': -1}
    grid = np.linspace(0, 1, 21)
    for w1 in grid:
        for w2 in grid:
            if w1 + w2 > 1:
                continue
            w3 = 1 - w1 - w2
            blended = w1 * orig_oof + w2 * tuned_oof + w3 * meta_oof
            best_thr = 0.5
            best_f = 0
            for thr in np.linspace(0.1, 0.9, 81):
                f = f1_score(y, (blended >= thr).astype(int))
                if f > best_f:
                    best_f = f
                    best_thr = thr
            if best_f > best['f1']:
                best.update({'w_orig': float(w1), 'w_tuned': float(w2), 'w_meta': float(w3), 'thr': float(best_thr), 'f1': float(best_f)})

    print('Best three-way blend (coarse):', best)

    # refine around best with finer grid
    w1_center = best['w_orig']
    w2_center = best['w_tuned']
    w1_grid = np.clip(np.linspace(max(0, w1_center - 0.1), min(1, w1_center + 0.1), 21), 0, 1)
    w2_grid = np.clip(np.linspace(max(0, w2_center - 0.1), min(1, w2_center + 0.1), 21), 0, 1)
    for w1 in w1_grid:
        for w2 in w2_grid:
            if w1 + w2 > 1:
                continue
            w3 = 1 - w1 - w2
            blended = w1 * orig_oof + w2 * tuned_oof + w3 * meta_oof
            best_thr = 0.5
            best_f = 0
            for thr in np.linspace(0.1, 0.9, 161):
                f = f1_score(y, (blended >= thr).astype(int))
                if f > best_f:
                    best_f = f
                    best_thr = thr
            if best_f > best['f1']:
                best.update({'w_orig': float(w1), 'w_tuned': float(w2), 'w_meta': float(w3), 'thr': float(best_thr), 'f1': float(best_f)})

    print('Best three-way blend (refined):', best)

    # write submission for test blend
    blended_test = best['w_orig'] * orig_test + best['w_tuned'] * tuned_test + best['w_meta'] * test_meta
    test_pred = (blended_test >= best['thr']).astype(int)
    submission = pd.DataFrame({'id': test['id'].astype(int), 'target': test_pred})
    sample = pd.read_csv(SAMPLE_PATH)
    submission = sample[['id']].merge(submission, on='id', how='left')
    submission.to_csv(ROOT / 'submission_blend_three_way.csv', index=False)
    with open(ROOT / 'model_report_blend_three_way.json', 'w') as f:
        json.dump(best, f, indent=2)
    print('Wrote submission_blend_three_way.csv and model_report_blend_three_way.json')


if __name__ == '__main__':
    main()
