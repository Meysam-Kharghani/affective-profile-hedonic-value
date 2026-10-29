#!/usr/bin/env python3
"""Reusable context-dependent analysis functions.

Implements cross-context transfer, participant-bootstrap intervals, split-half
stability, descriptive relative bidirectionality, and descriptive model-implied
surfaces for the fixed context-analysis cohort."""

from __future__ import annotations
import argparse, hashlib, json, math, os, shutil, sys, tempfile
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

SEED = 20260825
BOOT_N = 5000
STABILITY_REPEATS = 1000
SURFACE_GRID_N = 21
OUTCOMES = ['valence', 'liking']
AFFECTS = ['joy', 'inspiration', 'tenderness', 'sadness', 'fear', 'disgust', 'arousal', 'familiarity']
CATEGORIES = ['disgust', 'fear', 'inspiration', 'joy', 'neutral', 'sadness', 'tenderness']
ALPHAS = np.array([0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0], dtype=float)

def onehot():
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=False)

def make_ridge(num_cols, cat_cols, alpha: float | None=None, ridgecv: bool=False):
    trans = []
    if num_cols:
        trans.append(('num', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), list(num_cols)))
    if cat_cols:
        trans.append(('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('onehot', onehot())]), list(cat_cols)))
    prep = ColumnTransformer(trans, remainder='drop')
    reg = RidgeCV(alphas=ALPHAS, fit_intercept=True) if ridgecv else Ridge(alpha=float(alpha if alpha is not None else 1.0), fit_intercept=True)
    return Pipeline([('prep', prep), ('ridge', reg)])

def choose_alpha(train: pd.DataFrame, outcome: str, num_cols, cat_cols, group_col: str, n_jobs: int):
    groups = train[group_col].astype(str).to_numpy()
    ng = pd.Series(groups).nunique()
    if ng < 2:
        return (1.0, np.nan)
    cv = GroupKFold(n_splits=min(5, int(ng)))
    pipe = make_ridge(num_cols, cat_cols, alpha=1.0)
    gs = GridSearchCV(pipe, {'ridge__alpha': list(ALPHAS)}, scoring='neg_mean_squared_error', cv=cv, n_jobs=n_jobs, refit=True, error_score='raise')
    X = train[list(num_cols) + list(cat_cols)]
    y = train[outcome].to_numpy(float)
    gs.fit(X, y, groups=groups)
    return (float(gs.best_params_['ridge__alpha']), float(-gs.best_score_))

def safe_pearson(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a = a[m]
    b = b[m]
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])

def safe_spearman(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a = a[m]
    b = b[m]
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(spearmanr(a, b).statistic)

def metric_dict(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    return {'n': int(len(y)), 'r2': float(r2_score(y, p)), 'mae': float(mean_absolute_error(y, p)), 'rmse': float(math.sqrt(mean_squared_error(y, p))), 'pearson_r': safe_pearson(y, p)}

def sufficient_stats(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    return {'n': int(len(y)), 'sum_y': float(y.sum()), 'sum_y2': float((y * y).sum()), 'sum_p': float(p.sum()), 'sum_p2': float((p * p).sum()), 'sum_yp': float((y * p).sum()), 'sum_abs_err': float(np.abs(y - p).sum()), 'sum_sq_err': float(((y - p) ** 2).sum())}

def aggregate_stats(rows):
    keys = ['n', 'sum_y', 'sum_y2', 'sum_p', 'sum_p2', 'sum_yp', 'sum_abs_err', 'sum_sq_err']
    out = {k: 0.0 for k in keys}
    for r in rows:
        for k in keys:
            out[k] += float(r[k])
    out['n'] = int(round(out['n']))
    return out

def metrics_from_stats(s):
    n = float(s['n'])
    tss = s['sum_y2'] - s['sum_y'] ** 2 / n
    vy = tss
    vp = s['sum_p2'] - s['sum_p'] ** 2 / n
    cov = s['sum_yp'] - s['sum_y'] * s['sum_p'] / n
    return {'r2': float(1 - s['sum_sq_err'] / tss) if tss > 0 else np.nan, 'mae': float(s['sum_abs_err'] / n), 'rmse': float(math.sqrt(s['sum_sq_err'] / n)), 'pearson_r': float(cov / math.sqrt(vy * vp)) if vy > 0 and vp > 0 else np.nan}

def cross_context(df: pd.DataFrame, n_jobs: int):
    preds = []
    alphas = []
    pmetrics = []
    directions = [('imagery', 'video'), ('video', 'imagery')]
    for outcome in OUTCOMES:
        for src, dst in directions:
            dname = f'{src}_to_{dst}'
            for sid in sorted(df.subject_id.unique()):
                train = df[(df.subject_id != sid) & (df.context == src)].copy()
                test = df[(df.subject_id == sid) & (df.context == dst)].copy()
                alpha, mse = choose_alpha(train, outcome, AFFECTS, ['emotion_category'], 'subject_id', n_jobs)
                model = make_ridge(AFFECTS, ['emotion_category'], alpha=alpha)
                model.fit(train[AFFECTS + ['emotion_category']], train[outcome].to_numpy(float))
                pp = model.predict(test[AFFECTS + ['emotion_category']])
                yy = test[outcome].to_numpy(float)
                md = metric_dict(yy, pp)
                pmetrics.append({'outcome': outcome, 'direction': dname, 'subject_id': sid, **md})
                alphas.append({'outcome': outcome, 'direction': dname, 'target_subject': sid, 'alpha': alpha, 'inner_cv_mse': mse, 'n_train': len(train), 'n_test': len(test)})
                for row, p in zip(test.itertuples(index=False), pp):
                    preds.append({'outcome': outcome, 'direction': dname, 'subject_id': sid, 'canonical_row_index': row.canonical_row_index, 'y': getattr(row, outcome), 'pred': float(p)})
    return (pd.DataFrame(preds), pd.DataFrame(alphas), pd.DataFrame(pmetrics))

def bootstrap_cross_context(pred: pd.DataFrame, pmetrics: pd.DataFrame):
    rng = np.random.default_rng(SEED + 313)
    rows = []
    asym = []
    for outcome in OUTCOMES:
        perdir = {}
        for direction in ['imagery_to_video', 'video_to_imagery']:
            g = pred[(pred.outcome == outcome) & (pred.direction == direction)]
            subjects = sorted(g.subject_id.unique())
            cache = {sid: sufficient_stats(sg.y, sg.pred) for sid, sg in g.groupby('subject_id')}
            point = metrics_from_stats(aggregate_stats(list(cache.values())))
            boot = {k: [] for k in ['r2', 'mae', 'rmse', 'pearson_r']}
            for _ in range(BOOT_N):
                samp = rng.choice(subjects, size=len(subjects), replace=True)
                mm = metrics_from_stats(aggregate_stats([cache[s] for s in samp]))
                for k in boot:
                    boot[k].append(mm[k])
            rec = {'outcome': outcome, 'direction': direction, 'participants': len(subjects), 'trials': len(g), **point}
            for k, v in boot.items():
                a = np.asarray(v, float)
                rec[f'{k}_ci_low'] = float(np.nanpercentile(a, 2.5))
                rec[f'{k}_ci_high'] = float(np.nanpercentile(a, 97.5))
            rows.append(rec)
            perdir[direction] = cache
        pm = pmetrics[pmetrics.outcome == outcome].pivot(index='subject_id', columns='direction', values='mae').dropna()
        delta = (pm['imagery_to_video'] - pm['video_to_imagery']).to_numpy(float)
        subs = pm.index.to_numpy()
        b = []
        for _ in range(BOOT_N):
            idx = rng.integers(0, len(delta), size=len(delta))
            b.append(float(delta[idx].mean()))
        lo, hi = np.percentile(b, [2.5, 97.5])
        point = float(delta.mean())
        asym.append({'outcome': outcome, 'metric': 'participant_MAE_imagery_to_video_minus_video_to_imagery', 'participants': len(delta), 'point': point, 'ci_low': float(lo), 'ci_high': float(hi), 'asymmetry_supported': bool(lo > 0 or hi < 0), 'favored_direction_if_supported': ('video_to_imagery' if point > 0 else 'imagery_to_video') if lo > 0 or hi < 0 else 'none'})
    return (pd.DataFrame(rows), pd.DataFrame(asym))

def trait_names():
    out = []
    for cat in CATEGORIES:
        for y in OUTCOMES:
            out.append((f'category_mean::{cat}::{y}', 'category_specific_valuation'))
    for y in OUTCOMES:
        out.append((f'dispersion::{y}', 'dispersion'))
    for a in AFFECTS:
        for y in OUTCOMES:
            out.append((f'coupling::{a}::{y}', 'affect_hedonic_coupling'))
    for y in OUTCOMES:
        out.append((f'context_video_minus_imagery::{y}', 'global_context_contrast'))
    return out

def _traits_from_arrays(vals: np.ndarray, cats: np.ndarray, ctx: np.ndarray, idx: np.ndarray):
    x = vals[idx]
    c = cats[idx]
    k = ctx[idx]
    rec = np.full(34, np.nan, float)
    q = 0
    for cat in CATEGORIES:
        mm = c == cat
        for yi in range(2):
            rec[q] = float(np.mean(x[mm, 8 + yi])) if np.any(mm) else np.nan
            q += 1
    for yi in range(2):
        v = x[:, 8 + yi]
        rec[q] = float(np.std(v, ddof=1)) if len(v) >= 2 else np.nan
        q += 1
    for affect_index in range(8):
        for yi in range(2):
            rec[q] = safe_pearson(x[:, affect_index], x[:, 8 + yi])
            q += 1
    for yi in range(2):
        vv = x[k == 'video', 8 + yi]
        ii = x[k == 'imagery', 8 + yi]
        rec[q] = float(np.mean(vv) - np.mean(ii)) if len(vv) and len(ii) else np.nan
        q += 1
    return rec

def split_half_stability(df: pd.DataFrame):
    names = trait_names()
    tnames = [x[0] for x in names]
    families = [x[1] for x in names]
    subjects = sorted(df.subject_id.unique())
    caches = {}
    for sid in subjects:
        g = df[df.subject_id == sid].sort_values('canonical_row_index')
        caches[sid] = (g[AFFECTS + OUTCOMES].to_numpy(float), g.emotion_category.astype(str).to_numpy(), g.context.astype(str).to_numpy(), int(g.subject_number.iloc[0]))
    repeat_rows = []
    for rep in range(STABILITY_REPEATS):
        amat = []
        bmat = []
        for sid in subjects:
            vals, cats, ctx, sn = caches[sid]
            rng = np.random.default_rng(SEED + rep * 10007 + sn * 101)
            perm = rng.permutation(len(vals))
            nA = len(vals) // 2
            amat.append(_traits_from_arrays(vals, cats, ctx, perm[:nA]))
            bmat.append(_traits_from_arrays(vals, cats, ctx, perm[nA:]))
        A = np.vstack(amat)
        B = np.vstack(bmat)
        for j, t in enumerate(tnames):
            av = A[:, j]
            bv = B[:, j]
            m = np.isfinite(av) & np.isfinite(bv)
            repeat_rows.append({'repeat': rep, 'trait': t, 'family': families[j], 'n_participants': int(m.sum()), 'pearson_r': safe_pearson(av, bv), 'spearman_r': safe_spearman(av, bv)})
    rr = pd.DataFrame(repeat_rows)
    summ = []
    for (trait, family), g in rr.groupby(['trait', 'family'], sort=False):
        pr = g.pearson_r.to_numpy(float)
        sr = g.spearman_r.to_numpy(float)
        med = float(np.nanmedian(pr))
        summ.append({'trait': trait, 'family': family, 'repeats': len(g), 'median_participants': float(np.nanmedian(g.n_participants)), 'median_pearson_r': med, 'pearson_p2_5': float(np.nanpercentile(pr, 2.5)), 'pearson_p97_5': float(np.nanpercentile(pr, 97.5)), 'median_spearman_r': float(np.nanmedian(sr)), 'spearman_p2_5': float(np.nanpercentile(sr, 2.5)), 'spearman_p97_5': float(np.nanpercentile(sr, 97.5)), 'stable_trait_descriptor': bool(med >= 0.5)})
    ss = pd.DataFrame(summ)
    fs = []
    for family, g in ss.groupby('family'):
        n = len(g)
        ns = int(g.stable_trait_descriptor.sum())
        fs.append({'family': family, 'n_traits': n, 'n_stable_traits': ns, 'stable_fraction': ns / n, 'partially_stable_family': bool(ns >= math.ceil(n / 2)), 'rule': 'at least half constituent traits have median Pearson r >= 0.50'})
    return (rr, ss, pd.DataFrame(fs))

def relative_hedonic_bidirectionality(df: pd.DataFrame):
    rows = []
    d = df.copy()
    for y in OUTCOMES:
        d[f'{y}_pc_centered'] = d[y] - d.groupby(['subject_id', 'context'])[y].transform('mean')
        for cat in CATEGORIES:
            g = d[d.emotion_category == cat].copy()
            c = f'{y}_pc_centered'
            pooled_pos = float((g[c] > 0).mean())
            pooled_neg = float((g[c] < 0).mean())
            pooled_zero = float((g[c] == 0).mean())
            pp = []
            both = 0
            for sid, sg in g.groupby('subject_id'):
                pos = float((sg[c] > 0).mean())
                neg = float((sg[c] < 0).mean())
                pp.append((pos, neg))
                if (sg[c] > 0).any() and (sg[c] < 0).any():
                    both += 1
            pb_pos = float(np.mean([x[0] for x in pp]))
            pb_neg = float(np.mean([x[1] for x in pp]))
            rows.append({'outcome': y, 'emotion_category': cat, 'trials': len(g), 'participants': g.subject_id.nunique(), 'pooled_positive_prop': pooled_pos, 'pooled_negative_prop': pooled_neg, 'pooled_zero_prop': pooled_zero, 'participant_balanced_positive_prop': pb_pos, 'participant_balanced_negative_prop': pb_neg, 'participants_with_both_signs': both, 'participant_both_signs_fraction': both / max(1, g.subject_id.nunique()), 'meaningful_bidirectionality_descriptor': bool(pb_pos >= 0.1 and pb_neg >= 0.1)})
    return pd.DataFrame(rows)

def model_implied_surfaces(df: pd.DataFrame):
    grid_rows = []
    slope_rows = []
    alpha_rows = []
    for sid, g in df.groupby('subject_id'):
        g = g.copy()
        med = {a: float(g[a].median()) for a in AFFECTS}
        cats = g[['context', 'emotion_category']].copy().reset_index(drop=True)
        for y in OUTCOMES:
            model = make_ridge(AFFECTS, ['context', 'emotion_category'], ridgecv=True)
            model.fit(g[AFFECTS + ['context', 'emotion_category']], g[y].to_numpy(float))
            alpha = float(model.named_steps['ridge'].alpha_)
            alpha_rows.append({'subject_id': sid, 'outcome': y, 'alpha': alpha, 'n_trials': len(g)})
            for focal in AFFECTS:
                lo = float(g[focal].quantile(0.05))
                hi = float(g[focal].quantile(0.95))
                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                    continue
                vals = np.linspace(lo, hi, SURFACE_GRID_N)
                preds = []
                for v in vals:
                    base = pd.DataFrame({a: np.repeat(med[a], len(cats)) for a in AFFECTS})
                    base[focal] = v
                    base['context'] = cats.context.values
                    base['emotion_category'] = cats.emotion_category.values
                    pr = model.predict(base[AFFECTS + ['context', 'emotion_category']])
                    pm = float(np.mean(pr))
                    preds.append(pm)
                    grid_rows.append({'subject_id': sid, 'outcome': y, 'focal_affect': focal, 'grid_index': len(preds) - 1, 'focal_value': float(v), 'predicted_mean': pm, 'predicted_sd_over_observed_nominal_mix': float(np.std(pr, ddof=0)), 'focal_p05': lo, 'focal_p95': hi, 'other_numeric_rule': 'within-participant empirical medians', 'nominal_rule': 'average over participant observed context/category mix', 'interpretation': 'DESCRIPTIVE_MODEL_IMPLIED_ONLY'})
                slope_rows.append({'subject_id': sid, 'outcome': y, 'focal_affect': focal, 'focal_p05': lo, 'focal_p95': hi, 'predicted_at_p05': preds[0], 'predicted_at_p95': preds[-1], 'delta_predicted_p95_minus_p05': preds[-1] - preds[0], 'average_slope_per_rating_unit': (preds[-1] - preds[0]) / (hi - lo), 'alpha': alpha})
    return (pd.DataFrame(grid_rows), pd.DataFrame(slope_rows), pd.DataFrame(alpha_rows))
