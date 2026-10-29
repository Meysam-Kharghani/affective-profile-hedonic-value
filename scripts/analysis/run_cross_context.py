#!/usr/bin/env python3
"""Run cross-context and context-dependent analyses on the fixed analysis cohort."""
from __future__ import annotations
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np
import pandas as pd


def import_module(path: Path):
    spec=importlib.util.spec_from_file_location('cross_context_analysis',path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def save(df,path,gzip=False):
    df.to_csv(path,index=False,compression='gzip' if gzip else None)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--canonical',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--cohort-file',type=Path,default=None)
    ap.add_argument('--n-jobs',type=int,default=-1)
    ap.add_argument('--skip-secondary',action='store_true')
    args=ap.parse_args()
    root=Path(__file__).resolve().parents[2]
    module=import_module(root/'src/analysis_pipeline/04_cross_context_stability_bidirectionality_surfaces.py')
    cohort_file=args.cohort_file or root/'analysis_specification/context_analysis_cohort.csv'
    cohort=pd.read_csv(cohort_file)
    flag=cohort['included_in_context_analysis']
    if flag.dtype == bool:
        include_mask=flag
    else:
        include_mask=flag.astype(str).str.lower().isin(['true','1','yes'])
    include=set(cohort.loc[include_mask,'participant_id'].astype(str))
    df=pd.read_csv(args.canonical)
    df=df[df.subject_id.astype(str).isin(include)].copy().reset_index(drop=True)
    if df.subject_id.nunique()!=45 or len(df)!=1890:
        raise RuntimeError(f'Expected fixed context cohort of 45 participants and 1,890 trials; found {df.subject_id.nunique()} participants and {len(df)} trials.')
    counts=pd.crosstab(df.subject_id,df.context)
    if not ((counts.get('imagery',0)==21)&(counts.get('video',0)==21)).all():
        raise RuntimeError('Each context-analysis participant must contribute 21 imagery and 21 video trials.')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    pred,alpha,pmetrics=module.cross_context(df,args.n_jobs)
    save(pred,args.output_dir/'cross_context_predictions.csv.gz',True)
    save(alpha,args.output_dir/'cross_context_alpha_selection.csv')
    save(pmetrics,args.output_dir/'cross_context_participant_metrics.csv')
    boot,asym=module.bootstrap_cross_context(pred,pmetrics)
    save(boot,args.output_dir/'cross_context_bootstrap_summary.csv')
    save(asym,args.output_dir/'cross_context_directional_asymmetry.csv')
    rows=[]
    for (outcome,direction),g in pred.groupby(['outcome','direction']):
        y=g.y.to_numpy(float); p=g.pred.to_numpy(float)
        sse=float(np.sum((y-p)**2)); sst=float(np.sum((y-y.mean())**2))
        rows.append({'outcome':outcome,'direction':direction,'n':len(g),'r2':1-sse/sst,'mae':float(np.mean(np.abs(y-p))),'rmse':float(np.sqrt(np.mean((y-p)**2))),'pearson_r':float(np.corrcoef(y,p)[0,1])})
    save(pd.DataFrame(rows),args.output_dir/'cross_context_pooled_metrics.csv')
    if not args.skip_secondary:
        repeat,summary,family=module.split_half_stability(df.copy())
        save(repeat,args.output_dir/'split_half_stability_repeats.csv.gz',True)
        save(summary,args.output_dir/'split_half_stability_summary.csv')
        save(family,args.output_dir/'split_half_stability_family_summary.csv')
        save(module.relative_hedonic_bidirectionality(df.copy()),args.output_dir/'relative_hedonic_bidirectionality.csv')
        grid,slopes,alphas=module.model_implied_surfaces(df.copy())
        save(grid,args.output_dir/'model_implied_surface_grid.csv.gz',True)
        save(slopes,args.output_dir/'model_implied_surface_slopes.csv')
        save(alphas,args.output_dir/'model_implied_surface_alpha.csv')
    print('PASS: cross-context and context-dependent analyses')


if __name__=='__main__':
    main()
