#!/usr/bin/env python3
"""Run personalization against the global affective-plus-nominal comparator."""
from __future__ import annotations
import argparse, importlib, os, shutil, sys, tempfile
from pathlib import Path
import pandas as pd


def prepare_module(source: Path):
    temp=tempfile.TemporaryDirectory(prefix='personalization_module_')
    tempdir=Path(temp.name)
    shutil.copy2(source,tempdir/'personalization_analysis.py')
    tempstr=str(tempdir)
    sys.path.insert(0,tempstr)
    existing=os.environ.get('PYTHONPATH','')
    os.environ['PYTHONPATH']=tempstr if not existing else tempstr+os.pathsep+existing
    module=importlib.import_module('personalization_analysis')
    return temp,module


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--canonical',type=Path,required=True)
    ap.add_argument('--global-predictions',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--n-jobs',type=int,default=-1)
    args=ap.parse_args()
    root=Path(__file__).resolve().parents[2]
    temp,module=prepare_module(root/'src/analysis_pipeline/02_global_mapping_and_personalization.py')
    try:
        df=pd.read_csv(args.canonical)
        pred=pd.read_csv(args.global_predictions)
        if len(df)!=2225 or df.subject_id.nunique()!=53:
            raise RuntimeError('Expected 2,225 core trials from 53 participants.')
        m3=pred[pred.model=='M3_affective_nominal'].copy()
        if len(m3)!=len(df)*len(module.OUTCOMES):
            raise RuntimeError('Global comparator predictions are incomplete.')
        args.output_dir.mkdir(parents=True,exist_ok=True)
        metrics,stats,alphas=module.execute_personalization(df.copy(),m3.copy(),args.n_jobs)
        metrics.to_csv(args.output_dir/'personalization_metrics.csv.gz',index=False,compression='gzip')
        stats.to_csv(args.output_dir/'personalization_sufficient_stats.csv.gz',index=False,compression='gzip')
        alphas.to_csv(args.output_dir/'personalization_alpha_selection.csv.gz',index=False,compression='gzip')
        summary,support=module.bootstrap_personalization(metrics,stats,n_boot=module.BOOT_N,seed=module.SEED)
        summary.to_csv(args.output_dir/'personalization_bootstrap_summary.csv',index=False)
        support.to_csv(args.output_dir/'personalization_support_gate.csv',index=False)
        print('PASS: personalization analysis')
    finally:
        temp.cleanup()


if __name__=='__main__':
    main()
