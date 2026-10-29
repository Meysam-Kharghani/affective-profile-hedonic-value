#!/usr/bin/env python3
"""Reusable low-dimensional sensitivity functions.

Implements the held-out-participant low-dimensional models and participant-level
bootstrap comparisons used by scripts/analysis/run_lowdim.py."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

ALPHAS=np.array([0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1.0,3.0,10.0,30.0,100.0,300.0,1000.0,3000.0,10000.0,30000.0,100000.0],float)
AFFECT8=["joy","inspiration","tenderness","sadness","fear","disgust","arousal","familiarity"]
OUTCOMES=["valence","liking"]
MODELS={
    "affect_balance":(["affect_balance"],None),
    "positive_negative_2D":(["positive_affect","negative_affect"],None),
    "balance_arousal_familiarity_3D":(["affect_balance","arousal","familiarity"],None),
    "PCA1_of_affect8":(AFFECT8,1),
    "PCA2_of_affect8":(AFFECT8,2),
    "full_affect8":(AFFECT8,None),
}

def scale_fit(x):
    med=np.nanmedian(x,axis=0)
    xi=np.where(np.isnan(x),med,x)
    mean=xi.mean(axis=0); sd=xi.std(axis=0)
    sd=np.where(sd>0,sd,1.0)
    return med,mean,sd

def scale_apply(x,p):
    med,mean,sd=p
    xi=np.where(np.isnan(x),med,x)
    return (xi-mean)/sd

def pca_fit(x,k):
    mean=x.mean(axis=0)
    _,_,vt=np.linalg.svd(x-mean,full_matrices=False)
    return mean,vt[:k].T

def pca_apply(x,p):
    mean,w=p
    return (x-mean)@w

def ridge_fit(x,y,alpha):
    xm=x.mean(axis=0); ym=y.mean()
    xc=x-xm; yc=y-ym
    beta=np.linalg.solve(xc.T@xc + float(alpha)*np.eye(x.shape[1]), xc.T@yc)
    return beta, float(ym-xm@beta)

def predict(x,fit):
    beta,intercept=fit
    return x@beta+intercept

def fit_one_outer(train,test,outcome,cols,pca_k):
    xtr0=train[cols].to_numpy(float); ytr=train[outcome].to_numpy(float)
    xte0=test[cols].to_numpy(float)
    groups=train.subject_id.astype(str).to_numpy()
    mse=np.zeros(len(ALPHAS),float)
    for it,iv in GroupKFold(n_splits=5).split(xtr0,ytr,groups):
        sc=scale_fit(xtr0[it])
        xa=scale_apply(xtr0[it],sc); xv=scale_apply(xtr0[iv],sc)
        if pca_k:
            pc=pca_fit(xa,pca_k); xa=pca_apply(xa,pc); xv=pca_apply(xv,pc)
        for j,a in enumerate(ALPHAS):
            pv=predict(xv,ridge_fit(xa,ytr[it],a))
            mse[j]+=np.mean((ytr[iv]-pv)**2)
    alpha=float(ALPHAS[np.argmin(mse)])
    sc=scale_fit(xtr0)
    xtr=scale_apply(xtr0,sc); xte=scale_apply(xte0,sc)
    if pca_k:
        pc=pca_fit(xtr,pca_k); xtr=pca_apply(xtr,pc); xte=pca_apply(xte,pc)
    return predict(xte,ridge_fit(xtr,ytr,alpha)), alpha

def lopo(df,outcome,model):
    cols,pca_k=MODELS[model]
    rows=[]
    for sid in sorted(df.subject_id.unique()):
        train=df[df.subject_id!=sid]
        test=df[df.subject_id==sid]
        pred,alpha=fit_one_outer(train,test,outcome,cols,pca_k)
        for y,p in zip(test[outcome].to_numpy(float),pred):
            rows.append((sid,float(y),float(p),alpha))
    return pd.DataFrame(rows,columns=["subject_id","y","pred","alpha"])

def subject_stats(g):
    rows=[]
    for sid,s in g.groupby("subject_id"):
        y=s.y.to_numpy(float); p=s.pred.to_numpy(float)
        rows.append([sid,len(y),y.sum(),(y*y).sum(),np.abs(y-p).sum(),((y-p)**2).sum()])
    return pd.DataFrame(rows,columns=["subject_id","n","sum_y","sum_y2","sum_abs_err","sum_sq_err"]).set_index("subject_id")

def metrics_from_totals(a):
    n=a[:,0]; tss=a[:,2]-(a[:,1]**2)/n
    return 1-a[:,4]/tss, a[:,3]/n

def bootstrap_delta(full,base,nboot=5000,seed=20261027):
    sf=subject_stats(full); sb=subject_stats(base)
    subjects=sorted(sf.index)
    f=sf.loc[subjects][["n","sum_y","sum_y2","sum_abs_err","sum_sq_err"]].to_numpy(float)
    b=sb.loc[subjects][["n","sum_y","sum_y2","sum_abs_err","sum_sq_err"]].to_numpy(float)
    rng=np.random.default_rng(seed)
    counts=rng.multinomial(len(subjects),np.ones(len(subjects))/len(subjects),size=nboot)
    rf,mf=metrics_from_totals(counts@f); rb,mb=metrics_from_totals(counts@b)
    prf,pmf=metrics_from_totals(f.sum(axis=0,keepdims=True))
    prb,pmb=metrics_from_totals(b.sum(axis=0,keepdims=True))
    return {
        "delta_r2":float(prf[0]-prb[0]),
        "r2_ci_low":float(np.percentile(rf-rb,2.5)),
        "r2_ci_high":float(np.percentile(rf-rb,97.5)),
        "mae_improvement":float(pmb[0]-pmf[0]),
        "mae_ci_low":float(np.percentile(mb-mf,2.5)),
        "mae_ci_high":float(np.percentile(mb-mf,97.5)),
    }
