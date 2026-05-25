import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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


# Prepare data and precomputed TE + violation dicts
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
train = add_engineered_features(train)
test = add_engineered_features(test)
y = train['target'].astype(int)

# precompute OOF target encoding for City/Zip
for c in ['City', 'Zip']:
    s = train[c].astype(str)
    s_test = test[c].astype(str)
    oof_enc, test_enc = oof_target_encode(s, y, s_test, n_splits=5, alpha=30, noise_level=0.01)
    train[c + '_te'] = oof_enc
    test[c + '_te'] = test_enc

# violation features
top_codes = train['viol_codes'].explode().dropna().astype(str).value_counts().head(100).index.tolist()
train['viol_dict'] = build_violation_dicts(train['viol_codes'], top_codes)
test['viol_dict'] = build_violation_dicts(test['viol_codes'], top_codes)

feature_cols = [
    'Facility Type', 'Risk', 'City', 'State', 'Inspection Type',
    'Zip', 'Latitude', 'Longitude', 'year', 'month', 'weekday',
    'viol_count', 'unique_viol_count', 'has_any_violation', 'City_te', 'Zip_te'
]

X_full = train[feature_cols].copy()
X_test_full = test[feature_cols].copy()

cat_cols = ['Facility Type', 'Risk', 'City', 'State', 'Inspection Type']
num_cols = ['Zip', 'Latitude', 'Longitude', 'year', 'month', 'weekday', 'viol_count', 'unique_viol_count', 'has_any_violation', 'City_te', 'Zip_te']

vec = DictVectorizer(sparse=True)

# We'll run Optuna to maximize CV F1
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
        'num_leaves': trial.suggest_int('num_leaves', 16, 128),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 200),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.4, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0),
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': 1,
    }

    fold_scores = []
    for tr_idx, va_idx in skf.split(X_full, y):
        X_tr = X_full.iloc[tr_idx].copy()
        X_va = X_full.iloc[va_idx].copy()
        y_tr = y.iloc[tr_idx]
        y_va = y.iloc[va_idx]

        d_tr = train['viol_dict'].iloc[tr_idx]
        d_va = train['viol_dict'].iloc[va_idx]
        d_test = test['viol_dict']

        Xv_tr = vec.fit_transform(d_tr)
        Xv_va = vec.transform(d_va)

        pre_ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
        ohe_cols = ['Facility Type', 'Risk', 'State', 'Inspection Type']
        Xo_tr = pre_ohe.fit_transform(X_tr[ohe_cols])
        Xo_va = pre_ohe.transform(X_va[ohe_cols])

        num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler(with_mean=False))])
        Xn_tr = num_pipe.fit_transform(X_tr[num_cols])
        Xn_va = num_pipe.transform(X_va[num_cols])

        from scipy.sparse import hstack
        Xall_tr = hstack([Xo_tr, Xn_tr, Xv_tr]).tocsr()
        Xall_va = hstack([Xo_va, Xn_va, Xv_va]).tocsr()

        clf = lgb.LGBMClassifier(**params)
        clf.fit(Xall_tr, y_tr)
        pred = clf.predict(Xall_va)
        f = f1_score(y_va, pred)
        fold_scores.append(f)

    return float(np.mean(fold_scores))


if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    print('Best trial:')
    print(study.best_trial.params)

    best_params = study.best_trial.params
    best_params.update({'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1})

    # Train final model with best params
    # prepare full matrices
    d_all = train['viol_dict']
    d_test = test['viol_dict']
    Xv_all = vec.fit_transform(d_all)
    Xv_test = vec.transform(d_test)

    pre_ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    ohe_cols = ['Facility Type', 'Risk', 'State', 'Inspection Type']
    Xo_all = pre_ohe.fit_transform(X_full[ohe_cols])
    Xo_test = pre_ohe.transform(X_test_full[ohe_cols])

    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler(with_mean=False))])
    Xn_all = num_pipe.fit_transform(X_full[num_cols])
    Xn_test = num_pipe.transform(X_test_full[num_cols])

    from scipy.sparse import hstack
    Xfull = hstack([Xo_all, Xn_all, Xv_all]).tocsr()
    Xtest_full = hstack([Xo_test, Xn_test, Xv_test]).tocsr()

    final = lgb.LGBMClassifier(**best_params)
    final.fit(Xfull, y)
    test_prob = final.predict_proba(Xtest_full)[:, 1]

    # find threshold on OOF by re-running CV predictions
    oof = np.zeros(len(X_full))
    for tr_idx, va_idx in skf.split(X_full, y):
        X_tr = X_full.iloc[tr_idx].copy()
        X_va = X_full.iloc[va_idx].copy()
        y_tr = y.iloc[tr_idx]

        d_tr = train['viol_dict'].iloc[tr_idx]
        d_va = train['viol_dict'].iloc[va_idx]

        Xv_tr = vec.fit_transform(d_tr)
        Xv_va = vec.transform(d_va)

        Xo_tr = pre_ohe.fit_transform(X_tr[ohe_cols])
        Xo_va = pre_ohe.transform(X_va[ohe_cols])

        Xn_tr = num_pipe.fit_transform(X_tr[num_cols])
        Xn_va = num_pipe.transform(X_va[num_cols])

        Xtr = hstack([Xo_tr, Xn_tr, Xv_tr]).tocsr()
        Xva = hstack([Xo_va, Xn_va, Xv_va]).tocsr()

        m = lgb.LGBMClassifier(**best_params)
        m.fit(Xtr, y_tr)
        oof[va_idx] = m.predict_proba(Xva)[:, 1]

    best_thr = 0.5
    best_f1 = 0
    for thr in np.linspace(0.1, 0.9, 81):
        f = f1_score(y, (oof >= thr).astype(int))
        if f > best_f1:
            best_f1 = f
            best_thr = thr

    test_pred = (test_prob >= best_thr).astype(int)
    submission = pd.DataFrame({'id': test['id'].astype(int), 'target': test_pred})
    sample = pd.read_csv(SAMPLE_PATH)
    submission = sample[['id']].merge(submission, on='id', how='left')
    submission.to_csv(ROOT / 'submission_lgbm_optuna.csv', index=False)

    report = {'best_params': best_params, 'best_oof_threshold': float(best_thr), 'best_oof_f1': float(best_f1)}
    with open(ROOT / 'model_report_lgbm_optuna.json', 'w') as f:
        json.dump(report, f, indent=2)

    print('Created submission_lgbm_optuna.csv and model_report_lgbm_optuna.json')
