#!/usr/bin/env python3
"""Reusable global mapping, held-out-stimulus, and personalization functions.

Public entry points are provided under scripts/analysis/. This module contains
only the model-fitting and bootstrap functions used by those entry points."""

from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from joblib import Parallel, delayed

SEED = 20260825
BOOT_N = 5000
PERS_REPEATS = 200
OUTCOMES = ['valence', 'liking']
NUMERIC = ['joy', 'inspiration', 'tenderness', 'sadness', 'fear', 'disgust', 'arousal', 'familiarity']
NOMINAL = ['context', 'emotion_category']
ALPHAS = np.array([0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0], dtype=float)
KS = [4, 8, 12, 16, 20, 30]
MODELS = {'M0_intercept': ([], []), 'M1_nominal': ([], NOMINAL), 'M2_affective': (NUMERIC, []), 'M3_affective_nominal': (NUMERIC, NOMINAL)}

def onehot():
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=False)

def make_ridge(num_cols, cat_cols, alpha: float | None=None, ridgecv: bool=False):
    transformers = []
    if num_cols:
        transformers.append(('num', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), list(num_cols)))
    if cat_cols:
        transformers.append(('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('onehot', onehot())]), list(cat_cols)))
    prep = ColumnTransformer(transformers, remainder='drop')
    if ridgecv:
        reg = RidgeCV(alphas=ALPHAS, fit_intercept=True)
    else:
        reg = Ridge(alpha=float(alpha if alpha is not None else 1.0), fit_intercept=True)
    return Pipeline([('prep', prep), ('ridge', reg)])

def safe_pearson(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    if len(y) < 2 or np.nanstd(y) == 0 or np.nanstd(p) == 0:
        return np.nan
    return float(np.corrcoef(y, p)[0, 1])

def metrics(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    return {'n': int(len(y)), 'r2': float(r2_score(y, p)) if len(y) >= 2 else np.nan, 'mae': float(mean_absolute_error(y, p)) if len(y) else np.nan, 'rmse': float(math.sqrt(mean_squared_error(y, p))) if len(y) else np.nan, 'pearson_r': safe_pearson(y, p)}

def sufficient_stats(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    return {'n': int(len(y)), 'sum_y': float(np.sum(y)), 'sum_y2': float(np.sum(y * y)), 'sum_p': float(np.sum(p)), 'sum_p2': float(np.sum(p * p)), 'sum_yp': float(np.sum(y * p)), 'sum_abs_err': float(np.sum(np.abs(y - p))), 'sum_sq_err': float(np.sum((y - p) ** 2))}

def metrics_from_stats(s):
    n = float(s['n'])
    if n <= 0:
        return {'r2': np.nan, 'mae': np.nan, 'rmse': np.nan, 'pearson_r': np.nan}
    tss = float(s['sum_y2'] - s['sum_y'] ** 2 / n)
    r2 = float(1.0 - s['sum_sq_err'] / tss) if tss > 0 else np.nan
    mae = float(s['sum_abs_err'] / n)
    rmse = float(math.sqrt(s['sum_sq_err'] / n))
    vy = float(s['sum_y2'] - s['sum_y'] ** 2 / n)
    vp = float(s['sum_p2'] - s['sum_p'] ** 2 / n)
    cov = float(s['sum_yp'] - s['sum_y'] * s['sum_p'] / n)
    r = float(cov / math.sqrt(vy * vp)) if vy > 0 and vp > 0 else np.nan
    return {'r2': r2, 'mae': mae, 'rmse': rmse, 'pearson_r': r}

def aggregate_stats(rows):
    keys = ['n', 'sum_y', 'sum_y2', 'sum_p', 'sum_p2', 'sum_yp', 'sum_abs_err', 'sum_sq_err']
    out = {k: 0.0 for k in keys}
    for r in rows:
        for k in keys:
            out[k] += float(r[k])
    out['n'] = int(round(out['n']))
    return out

def choose_alpha(train: pd.DataFrame, outcome: str, num_cols, cat_cols, group_col: str, n_jobs: int):
    X = train[list(num_cols) + list(cat_cols)]
    y = train[outcome].to_numpy(float)
    groups = train[group_col].astype(str).to_numpy()
    n_groups = pd.Series(groups).nunique()
    if n_groups < 2:
        return (1.0, np.nan)
    n_splits = min(5, int(n_groups))
    cv = GroupKFold(n_splits=n_splits)
    pipe = make_ridge(num_cols, cat_cols, alpha=1.0)
    gs = GridSearchCV(pipe, param_grid={'ridge__alpha': list(ALPHAS)}, scoring='neg_mean_squared_error', cv=cv, n_jobs=n_jobs, refit=True, error_score='raise', return_train_score=False)
    gs.fit(X, y, groups=groups)
    return (float(gs.best_params_['ridge__alpha']), float(-gs.best_score_))

def execute_global_lopo(df, n_jobs):
    pred_rows = []
    alpha_rows = []
    subjects = sorted(df.subject_id.unique())
    for outcome in OUTCOMES:
        for target in subjects:
            train = df[df.subject_id != target].copy()
            test = df[df.subject_id == target].copy()
            ytest = test[outcome].to_numpy(float)
            p0 = np.repeat(train[outcome].mean(), len(test))
            for row, pred in zip(test.itertuples(index=False), p0):
                pred_rows.append({'canonical_row_index': row.canonical_row_index, 'subject_id': row.subject_id, 'stimulus_id': row.stimulus_id, 'outcome': outcome, 'outer_unit': target, 'model': 'M0_intercept', 'y': getattr(row, outcome), 'pred': float(pred)})
            alpha_rows.append({'analysis': 'LOPO', 'outcome': outcome, 'outer_unit': target, 'model': 'M0_intercept', 'alpha': np.nan, 'inner_cv_mse': np.nan})
            for model in ['M1_nominal', 'M2_affective', 'M3_affective_nominal']:
                nums, cats = MODELS[model]
                alpha, mse = choose_alpha(train, outcome, nums, cats, 'subject_id', n_jobs)
                fit = make_ridge(nums, cats, alpha=alpha)
                fit.fit(train[nums + cats], train[outcome].to_numpy(float))
                pred = fit.predict(test[nums + cats])
                alpha_rows.append({'analysis': 'LOPO', 'outcome': outcome, 'outer_unit': target, 'model': model, 'alpha': alpha, 'inner_cv_mse': mse})
                for row, pp in zip(test.itertuples(index=False), pred):
                    pred_rows.append({'canonical_row_index': row.canonical_row_index, 'subject_id': row.subject_id, 'stimulus_id': row.stimulus_id, 'outcome': outcome, 'outer_unit': target, 'model': model, 'y': getattr(row, outcome), 'pred': float(pp)})
    return (pd.DataFrame(pred_rows), pd.DataFrame(alpha_rows))

def execute_stimulus_loso(df, n_jobs):
    pred_rows = []
    alpha_rows = []
    stimuli = sorted(df.stimulus_id.unique())
    for outcome in OUTCOMES:
        for target in stimuli:
            train = df[df.stimulus_id != target].copy()
            test = df[df.stimulus_id == target].copy()
            for model in ['M1_nominal', 'M2_affective', 'M3_affective_nominal']:
                nums, cats = MODELS[model]
                alpha, mse = choose_alpha(train, outcome, nums, cats, 'stimulus_id', n_jobs)
                fit = make_ridge(nums, cats, alpha=alpha)
                fit.fit(train[nums + cats], train[outcome].to_numpy(float))
                pred = fit.predict(test[nums + cats])
                alpha_rows.append({'analysis': 'LOSO_stimulus', 'outcome': outcome, 'outer_unit': target, 'model': model, 'alpha': alpha, 'inner_cv_mse': mse})
                for row, pp in zip(test.itertuples(index=False), pred):
                    pred_rows.append({'canonical_row_index': row.canonical_row_index, 'subject_id': row.subject_id, 'stimulus_id': row.stimulus_id, 'outcome': outcome, 'outer_unit': target, 'model': model, 'y': getattr(row, outcome), 'pred': float(pp)})
    return (pd.DataFrame(pred_rows), pd.DataFrame(alpha_rows))

def personal_seed(k, rep, subject_number):
    return int(SEED + k * 1000003 + rep * 1009 + int(subject_number) * 37)

def _personalization_subject_k_outcome(g, outcome, k, global_map):
    sid = str(g.subject_id.iloc[0])
    subj_num = int(g.subject_number.iloc[0])
    g = g.sort_values('canonical_row_index').reset_index(drop=True)
    metrics_rows = []
    stat_rows = []
    alpha_rows = []
    for rep in range(PERS_REPEATS):
        rng = np.random.default_rng(personal_seed(k, rep, subj_num))
        idx = rng.choice(np.arange(len(g)), size=k, replace=False)
        mask = np.zeros(len(g), dtype=bool)
        mask[idx] = True
        cal = g.loc[mask].copy()
        test = g.loc[~mask].copy()
        fit = make_ridge(NUMERIC, NOMINAL, ridgecv=True)
        fit.fit(cal[NUMERIC + NOMINAL], cal[outcome].to_numpy(float))
        pp = fit.predict(test[NUMERIC + NOMINAL])
        alpha = float(fit.named_steps['ridge'].alpha_)
        y = test[outcome].to_numpy(float)
        gg = np.asarray([global_map[int(i)] for i in test.canonical_row_index], float)
        mg = metrics(y, gg)
        mp = metrics(y, pp)
        metrics_rows.append({'outcome': outcome, 'k': k, 'repeat': rep, 'subject_id': sid, 'n_test': len(test), 'global_r2': mg['r2'], 'personal_r2': mp['r2'], 'delta_r2': mp['r2'] - mg['r2'], 'global_mae': mg['mae'], 'personal_mae': mp['mae'], 'delta_mae_global_minus_personal': mg['mae'] - mp['mae'], 'personal_alpha': alpha})
        sg = sufficient_stats(y, gg)
        sp = sufficient_stats(y, pp)
        stat_rows.append({'outcome': outcome, 'k': k, 'repeat': rep, 'subject_id': sid, 'model': 'global', **sg})
        stat_rows.append({'outcome': outcome, 'k': k, 'repeat': rep, 'subject_id': sid, 'model': 'personalized', **sp})
        alpha_rows.append({'outcome': outcome, 'k': k, 'repeat': rep, 'subject_id': sid, 'alpha': alpha})
    return (metrics_rows, stat_rows, alpha_rows)

def execute_personalization(df, lopo_m3_pred, n_jobs):
    jobs = []
    for outcome in OUTCOMES:
        go = lopo_m3_pred[lopo_m3_pred.outcome == outcome]
        global_map = dict(zip(go.canonical_row_index.astype(int), go.pred.astype(float)))
        if len(global_map) != len(df):
            raise RuntimeError('global M3 comparator does not cover canonical rows exactly once')
        for k in KS:
            for sid, g in df.groupby('subject_id'):
                if len(g) >= k + 5:
                    jobs.append((g.copy(), outcome, k, global_map))
    results = Parallel(n_jobs=n_jobs, prefer='processes', verbose=0)((delayed(_personalization_subject_k_outcome)(g, o, k, gm) for g, o, k, gm in jobs))
    metrics_rows = []
    stat_rows = []
    alpha_rows = []
    for a, b, c in results:
        metrics_rows.extend(a)
        stat_rows.extend(b)
        alpha_rows.extend(c)
    return (pd.DataFrame(metrics_rows), pd.DataFrame(stat_rows), pd.DataFrame(alpha_rows))

def bootstrap_personalization(metric_df, stat_df, n_boot=BOOT_N, seed=SEED):
    summary = []
    rng = np.random.default_rng(seed + 991)
    for outcome in OUTCOMES:
        for k in KS:
            m = metric_df[(metric_df.outcome == outcome) & (metric_df.k == k)].copy()
            s = stat_df[(stat_df.outcome == outcome) & (stat_df.k == k)].copy()
            subjects = sorted(m.subject_id.unique())
            pmae = m.groupby('subject_id')['delta_mae_global_minus_personal'].mean()
            mae_point = float(pmae.mean())
            r2s = {}
            for model in ['global', 'personalized']:
                agg = aggregate_stats([r for _, r in s[s.model == model].iterrows()])
                r2s[model] = metrics_from_stats(agg)['r2']
            dr2_point = float(r2s['personalized'] - r2s['global'])
            boot_mae = []
            boot_dr2 = []
            cache = {}
            for model in ['global', 'personalized']:
                sm = s[s.model == model]
                for sid, sg in sm.groupby('subject_id'):
                    cache[model, sid] = aggregate_stats([r for _, r in sg.iterrows()])
            for _ in range(n_boot):
                samp = rng.choice(subjects, size=len(subjects), replace=True)
                boot_mae.append(float(np.mean([pmae.loc[sid] for sid in samp])))
                bm = {}
                for model in ['global', 'personalized']:
                    agg = aggregate_stats([cache[model, sid] for sid in samp])
                    bm[model] = metrics_from_stats(agg)['r2']
                boot_dr2.append(bm['personalized'] - bm['global'])
            ba = np.asarray(boot_mae, float)
            br = np.asarray(boot_dr2, float)
            summary.append({'outcome': outcome, 'k': k, 'eligible_subjects': len(subjects), 'repeats': PERS_REPEATS, 'minimum_test_trials': 5, 'delta_mae_point': mae_point, 'delta_mae_ci_low': float(np.percentile(ba, 2.5)), 'delta_mae_ci_high': float(np.percentile(ba, 97.5)), 'pooled_delta_r2_point': dr2_point, 'pooled_delta_r2_ci_low': float(np.percentile(br, 2.5)), 'pooled_delta_r2_ci_high': float(np.percentile(br, 97.5)), 'positive_mae_ci': bool(np.percentile(ba, 2.5) > 0), 'positive_pooled_delta_r2': bool(dr2_point > 0)})
    out = pd.DataFrame(summary)
    support = []
    for outcome in OUTCOMES:
        g = out[(out.outcome == outcome) & out.k.isin([20, 30])]
        ok = bool(len(g) == 2 and g.positive_mae_ci.all() and g.positive_pooled_delta_r2.all())
        support.append({'outcome': outcome, 'incremental_personalization_supported': ok, 'gate': 'k20_and_k30: lower95_delta_MAE>0 AND pooled_delta_R2>0'})
    return (out, pd.DataFrame(support))
