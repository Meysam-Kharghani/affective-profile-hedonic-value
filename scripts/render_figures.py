#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser()
ap.add_argument('--output',default=str(ROOT/'figures'))
args=ap.parse_args()
OUT=Path(args.output); OUT.mkdir(parents=True,exist_ok=True)
META={'Software':'Matplotlib'}


def performance(outcome):
    l=pd.read_csv(ROOT/'results/source_data/lopo_model_summary.csv')
    h=pd.read_csv(ROOT/'results/source_data/heldout_stimulus_model_summary.csv')
    d=pd.read_csv(ROOT/'results/source_data/double_heldout_model_summary.csv')
    rows=[]
    for regime,df in [('LOPO',l),('Held-out stimulus',h),('Double holdout',d)]:
        g=df[(df.outcome==outcome)&df.model.isin(['M1_nominal','M2_affective','M3_affective_nominal'])]
        for _,r in g.iterrows(): rows.append((regime,r.model,r.r2))
    z=pd.DataFrame(rows,columns=['regime','model','r2'])
    labels=[f"{r}\n{m.replace('M1_nominal','Category').replace('M2_affective','Affective').replace('M3_affective_nominal','Affective + category')}" for r,m in zip(z.regime,z.model)]
    fig,ax=plt.subplots(figsize=(12,5)); ax.bar(range(len(z)),z.r2); ax.set_xticks(range(len(z)),labels,rotation=55,ha='right'); ax.set_ylabel('Pooled R²'); ax.set_title(f'{outcome.capitalize()}: model performance'); ax.axhline(0,linewidth=.8); fig.tight_layout(); fig.savefig(OUT/f'Figure_{outcome}_model_performance.png',dpi=200,metadata=META); plt.close(fig)


def lowdim(outcome):
    z=pd.read_csv(ROOT/'results/source_data/low_dimensional_sensitivity.csv'); z=z[z.outcome==outcome].copy()
    labels=z.model.str.replace('_',' ',regex=False)
    fig,ax=plt.subplots(figsize=(10,5)); ax.bar(range(len(z)),z.r2); ax.set_xticks(range(len(z)),labels,rotation=45,ha='right'); ax.set_ylabel('LOPO R²'); ax.set_title(f'{outcome.capitalize()}: low-dimensional sensitivity'); fig.tight_layout(); fig.savefig(OUT/f'Figure_{outcome}_low_dimensional_sensitivity.png',dpi=200,metadata=META); plt.close(fig)


def cross_context(outcome):
    z=pd.read_csv(ROOT/'results/main_tables/Table6_cross_context_transfer.csv'); z=z[z.outcome==outcome].copy()
    labels=z.direction.replace({'imagery_to_video':'Imagery → video','video_to_imagery':'Video → imagery'})
    y=z.r2.to_numpy(float); lo=y-z.r2_ci_low.to_numpy(float); hi=z.r2_ci_high.to_numpy(float)-y
    fig,ax=plt.subplots(figsize=(6.5,4.5)); ax.bar(range(len(z)),y); ax.errorbar(range(len(z)),y,yerr=np.vstack([lo,hi]),fmt='none',capsize=5); ax.set_xticks(range(len(z)),labels); ax.set_ylabel('Cross-context pooled R²'); ax.set_title(f'{outcome.capitalize()}: cross-context transfer'); ax.axhline(0,linewidth=.8); fig.tight_layout(); fig.savefig(OUT/f'Figure_{outcome}_cross_context_transfer.png',dpi=200,metadata=META); plt.close(fig)


def personalization(outcome):
    z=pd.read_csv(ROOT/'results/source_data/personalization_bootstrap_summary.csv'); z=z[z.outcome==outcome].sort_values('k')
    x=z.k.to_numpy(int); y=z.delta_mae_point.to_numpy(float); lo=y-z.delta_mae_ci_low.to_numpy(float); hi=z.delta_mae_ci_high.to_numpy(float)-y
    fig,ax=plt.subplots(figsize=(7,4.5)); ax.errorbar(x,y,yerr=np.vstack([lo,hi]),marker='o',capsize=4); ax.axhline(0,linewidth=.8); ax.set_xlabel('Calibration trials (k)'); ax.set_ylabel('ΔMAE vs global M3\n(positive = lower personalized MAE)'); ax.set_title(f'{outcome.capitalize()}: personalization'); fig.tight_layout(); fig.savefig(OUT/f'Figure_{outcome}_personalization.png',dpi=200,metadata=META); plt.close(fig)

for outcome in ['valence','liking']:
    performance(outcome); lowdim(outcome); cross_context(outcome); personalization(outcome)
print('PASS: figures rendered')
