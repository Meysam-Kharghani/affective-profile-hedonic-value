#!/usr/bin/env python3
"""Reusable robustness-analysis functions.

Implements simultaneous participant-and-stimulus holdout and within-participant
centering analyses used by scripts/analysis/run_robustness.py."""

from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 20260826
BOOT_N = 5000
OUTCOMES = ['valence', 'liking']
NUMERIC = ['joy', 'inspiration', 'tenderness', 'sadness', 'fear', 'disgust', 'arousal', 'familiarity']
NOMINAL = ['context', 'emotion_category']
ALPHAS = np.array([0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0], float)
MODELS = {'M1_nominal': ([], NOMINAL), 'M2_affective': (NUMERIC, []), 'M3_affective_nominal': (NUMERIC, NOMINAL)}
WNUM = [f'wp_{c}' for c in NUMERIC]
WMODELS = {'W1_nominal': ([], NOMINAL), 'W2_within_affective': (WNUM, []), 'W3_within_affective_nominal': (WNUM, NOMINAL)}

def onehot():
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=False)

def make_ridge(num_cols, cat_cols, alpha=1.0):
    tr = []
    if num_cols:
        tr.append(('num', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), list(num_cols)))
    if cat_cols:
        tr.append(('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('onehot', onehot())]), list(cat_cols)))
    prep = ColumnTransformer(tr, remainder='drop')
    return Pipeline([('prep', prep), ('ridge', Ridge(alpha=float(alpha), fit_intercept=True))])

def choose_alpha(train, outcome, num_cols, cat_cols, group_col, n_jobs):
    groups = train[group_col].astype(str).to_numpy()
    ng = pd.Series(groups).nunique()
    if ng < 2:
        return (1.0, np.nan)
    cv = GroupKFold(n_splits=min(5, int(ng)))
    pipe = make_ridge(num_cols, cat_cols, 1.0)
    gs = GridSearchCV(pipe, {'ridge__alpha': list(ALPHAS)}, scoring='neg_mean_squared_error', cv=cv, n_jobs=n_jobs, refit=True, error_score='raise')
    X = train[list(num_cols) + list(cat_cols)]
    y = train[outcome].to_numpy(float)
    gs.fit(X, y, groups=groups)
    return (float(gs.best_params_['ridge__alpha']), float(-gs.best_score_))

def metrics(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    return {'n': len(y), 'r2': float(r2_score(y, p)), 'mae': float(mean_absolute_error(y, p)), 'rmse': float(math.sqrt(mean_squared_error(y, p)))}

def weighted_metrics(y, p, w):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    w = np.asarray(w, float)
    good = np.isfinite(y) & np.isfinite(p) & np.isfinite(w) & (w > 0)
    y = y[good]
    p = p[good]
    w = w[good]
    if len(y) < 2 or w.sum() <= 0:
        return {'r2': np.nan, 'mae': np.nan, 'rmse': np.nan}
    ym = np.sum(w * y) / np.sum(w)
    tss = np.sum(w * (y - ym) ** 2)
    sse = np.sum(w * (y - p) ** 2)
    return {'r2': float(1 - sse / tss) if tss > 0 else np.nan, 'mae': float(np.sum(w * np.abs(y - p)) / np.sum(w)), 'rmse': float(math.sqrt(np.sum(w * (y - p) ** 2) / np.sum(w)))}

def percentile_ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return (float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))) if len(x) else (np.nan, np.nan)

def make_outer_folds(df: pd.DataFrame):
    subjects = np.array(sorted(df.subject_id.astype(str).unique()))
    stimuli = np.array(sorted(df.stimulus_id.astype(str).unique()))
    skf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    tkf = KFold(n_splits=5, shuffle=True, random_state=SEED + 1)
    sfolds = [set(subjects[ix]) for _, ix in skf.split(subjects)]
    tfolds = [set(stimuli[ix]) for _, ix in tkf.split(stimuli)]
    rows = []
    for si, S in enumerate(sfolds):
        for ti, T in enumerate(tfolds):
            test = df.subject_id.astype(str).isin(S) & df.stimulus_id.astype(str).isin(T)
            train = ~df.subject_id.astype(str).isin(S) & ~df.stimulus_id.astype(str).isin(T)
            rows.append((si, ti, S, T, train.to_numpy(), test.to_numpy()))
    return (rows, sfolds, tfolds)

def _fit_double_cell(df, si, ti, S, T, trainmask, testmask, outcome, model, n_jobs):
    train = df.loc[trainmask].copy()
    test = df.loc[testmask].copy()
    nums, cats = MODELS[model]
    alpha, mse = choose_alpha(train, outcome, nums, cats, 'subject_id', n_jobs)
    pipe = make_ridge(nums, cats, alpha)
    pipe.fit(train[nums + cats], train[outcome].to_numpy(float))
    pred = pipe.predict(test[nums + cats])
    rows = []
    for rr, pp in zip(test.itertuples(index=False), pred):
        rows.append({'canonical_row_index': rr.canonical_row_index, 'subject_id': rr.subject_id, 'stimulus_id': rr.stimulus_id, 'subject_fold': si, 'stimulus_fold': ti, 'outcome': outcome, 'model': model, 'y': getattr(rr, outcome), 'pred': float(pp)})
    return (rows, {'subject_fold': si, 'stimulus_fold': ti, 'outcome': outcome, 'model': model, 'alpha': alpha, 'inner_cv_mse': mse, 'n_train': len(train), 'n_test': len(test), 'n_train_subjects': train.subject_id.nunique(), 'n_train_stimuli': train.stimulus_id.nunique()})

def execute_double_heldout(df, n_jobs):
    folds, sfolds, tfolds = make_outer_folds(df)
    tasks = []
    for si, ti, S, T, tr, te in folds:
        if te.sum() == 0:
            raise RuntimeError(f'empty double-heldout cell {si}/{ti}')
        for outcome in OUTCOMES:
            for model in MODELS:
                tasks.append((si, ti, S, T, tr, te, outcome, model))
    res = Parallel(n_jobs=n_jobs, verbose=0)((delayed(_fit_double_cell)(df, *task, n_jobs=1) for task in tasks))
    pred = pd.DataFrame([x for rows, _ in res for x in rows])
    alpha = pd.DataFrame([a for _, a in res])
    counts = pred.groupby(['outcome', 'model', 'canonical_row_index']).size()
    if len(counts) != len(df) * len(OUTCOMES) * len(MODELS) or not (counts == 1).all():
        raise RuntimeError('double-heldout prediction coverage is not exactly once')
    return (pred, alpha, sfolds, tfolds)

def summarize_predictions(pred):
    rows = []
    for (outcome, model), g in pred.groupby(['outcome', 'model'], sort=True):
        m = metrics(g.y, g.pred)
        rows.append({'outcome': outcome, 'model': model, **m})
    return pd.DataFrame(rows)

def bootstrap_double(pred, n_boot=BOOT_N, seed=SEED):
    subjects = sorted(pred.subject_id.astype(str).unique())
    stimuli = sorted(pred.stimulus_id.astype(str).unique())
    smap = {s: i for i, s in enumerate(subjects)}
    tmap = {s: i for i, s in enumerate(stimuli)}
    rng = np.random.default_rng(seed)
    rows = []
    for outcome in OUTCOMES:
        m1 = pred[(pred.outcome == outcome) & (pred.model == 'M1_nominal')].sort_values('canonical_row_index')
        m3 = pred[(pred.outcome == outcome) & (pred.model == 'M3_affective_nominal')].sort_values('canonical_row_index')
        if not np.array_equal(m1.canonical_row_index.to_numpy(), m3.canonical_row_index.to_numpy()):
            raise RuntimeError('M1/M3 double-heldout rows misaligned')
        y = m1.y.to_numpy(float)
        p1 = m1.pred.to_numpy(float)
        p3 = m3.pred.to_numpy(float)
        si = np.array([smap[str(x)] for x in m1.subject_id])
        ti = np.array([tmap[str(x)] for x in m1.stimulus_id])
        dr = []
        dm = []
        r3 = []
        ns = len(subjects)
        nt = len(stimuli)
        for b in range(n_boot):
            cs = np.bincount(rng.integers(0, ns, ns), minlength=ns)
            ct = np.bincount(rng.integers(0, nt, nt), minlength=nt)
            w = cs[si] * ct[ti]
            a = weighted_metrics(y, p1, w)
            c = weighted_metrics(y, p3, w)
            dr.append(c['r2'] - a['r2'])
            dm.append(a['mae'] - c['mae'])
            r3.append(c['r2'])
        cr = percentile_ci(dr)
        cm = percentile_ci(dm)
        c3 = percentile_ci(r3)
        obs1 = metrics(y, p1)
        obs3 = metrics(y, p3)
        rows.append({'outcome': outcome, 'delta_r2_M3_minus_M1': obs3['r2'] - obs1['r2'], 'delta_r2_ci_low': cr[0], 'delta_r2_ci_high': cr[1], 'mae_improvement_M1_minus_M3': obs1['mae'] - obs3['mae'], 'mae_improvement_ci_low': cm[0], 'mae_improvement_ci_high': cm[1], 'M3_r2': obs3['r2'], 'M3_r2_ci_low': c3[0], 'M3_r2_ci_high': c3[1]})
    return pd.DataFrame(rows)

def add_within_person_columns(df):
    x = df.copy()
    for c in NUMERIC:
        x[f'wp_{c}'] = x[c] - x.groupby('subject_id')[c].transform('mean')
    for y in OUTCOMES:
        x[f'wp_{y}'] = x[y] - x.groupby('subject_id')[y].transform('mean')
    return x

def execute_within_lopo(df, n_jobs):
    x = add_within_person_columns(df)
    pred_rows = []
    alpha_rows = []
    subjects = sorted(x.subject_id.unique())
    for outcome in OUTCOMES:
        wy = f'wp_{outcome}'
        for target in subjects:
            train = x[x.subject_id != target].copy()
            test = x[x.subject_id == target].copy()
            for model, (nums, cats) in WMODELS.items():
                alpha, mse = choose_alpha(train, wy, nums, cats, 'subject_id', n_jobs)
                pipe = make_ridge(nums, cats, alpha)
                pipe.fit(train[nums + cats], train[wy].to_numpy(float))
                p = pipe.predict(test[nums + cats])
                for rr, pp in zip(test.itertuples(index=False), p):
                    pred_rows.append({'canonical_row_index': rr.canonical_row_index, 'subject_id': rr.subject_id, 'stimulus_id': rr.stimulus_id, 'outcome': outcome, 'model': model, 'y': getattr(rr, wy), 'pred': float(pp)})
                alpha_rows.append({'outcome': outcome, 'target_subject': target, 'model': model, 'alpha': alpha, 'inner_cv_mse': mse})
    return (pd.DataFrame(pred_rows), pd.DataFrame(alpha_rows), x)

def bootstrap_participant_primary(pred, n_boot=BOOT_N, seed=SEED + 11):
    rng = np.random.default_rng(seed)
    rows = []
    for outcome in OUTCOMES:
        m1 = pred[(pred.outcome == outcome) & (pred.model == 'W1_nominal')].copy()
        m3 = pred[(pred.outcome == outcome) & (pred.model == 'W3_within_affective_nominal')].copy()
        subs = sorted(m1.subject_id.unique())
        stats = []
        for s in subs:
            a = m1[m1.subject_id == s]
            c = m3[m3.subject_id == s]
            if not np.array_equal(a.sort_values('canonical_row_index').canonical_row_index.to_numpy(), c.sort_values('canonical_row_index').canonical_row_index.to_numpy()):
                raise RuntimeError('within M1/M3 subject rows mismatch')
            a = a.sort_values('canonical_row_index')
            c = c.sort_values('canonical_row_index')
            stats.append((a.y.to_numpy(float), a.pred.to_numpy(float), c.pred.to_numpy(float)))
        dr = []
        dm = []
        r3 = []
        n = len(subs)
        for b in range(n_boot):
            draw = rng.integers(0, n, n)
            ys = []
            p1 = []
            p3 = []
            for i in draw:
                ys.append(stats[i][0])
                p1.append(stats[i][1])
                p3.append(stats[i][2])
            y = np.concatenate(ys)
            q1 = np.concatenate(p1)
            q3 = np.concatenate(p3)
            a = metrics(y, q1)
            c = metrics(y, q3)
            dr.append(c['r2'] - a['r2'])
            dm.append(a['mae'] - c['mae'])
            r3.append(c['r2'])
        aa = metrics(m1.y, m1.pred)
        cc = metrics(m3.y, m3.pred)
        cr = percentile_ci(dr)
        cm = percentile_ci(dm)
        c3 = percentile_ci(r3)
        rows.append({'outcome': outcome, 'delta_r2_W3_minus_W1': cc['r2'] - aa['r2'], 'delta_r2_ci_low': cr[0], 'delta_r2_ci_high': cr[1], 'mae_improvement_W1_minus_W3': aa['mae'] - cc['mae'], 'mae_improvement_ci_low': cm[0], 'mae_improvement_ci_high': cm[1], 'W3_r2': cc['r2'], 'W3_r2_ci_low': c3[0], 'W3_r2_ci_high': c3[1]})
    return pd.DataFrame(rows)
