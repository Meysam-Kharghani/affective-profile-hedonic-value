#!/usr/bin/env python3
from __future__ import annotations
import argparse,importlib.util
from pathlib import Path
import pandas as pd

def loadmod(path,name):
 s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--canonical',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--n-jobs',type=int,default=-1); ap.add_argument('--bootstrap',type=int,default=5000); a=ap.parse_args()
 root=Path(__file__).resolve().parents[2]; m=loadmod(root/'src/analysis_pipeline/03_robustness.py','rob_orig'); df=pd.read_csv(a.canonical); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
 if len(df)!=2225 or df.subject_id.nunique()!=53: raise RuntimeError('Core cohort mismatch.')
 dp,da,_,_=m.execute_double_heldout(df,a.n_jobs); m.summarize_predictions(dp).to_csv(out/'double_heldout_model_summary.csv',index=False); m.bootstrap_double(dp,n_boot=a.bootstrap,seed=m.SEED).to_csv(out/'double_heldout_bootstrap.csv',index=False); dp.to_csv(out/'double_heldout_predictions.csv.gz',index=False,compression='gzip'); da.to_csv(out/'double_heldout_alpha_selection.csv',index=False)
 wp,wa,_=m.execute_within_lopo(df,a.n_jobs); m.summarize_predictions(wp).to_csv(out/'within_person_model_summary.csv',index=False); m.bootstrap_participant_primary(wp,n_boot=a.bootstrap,seed=m.SEED+11).to_csv(out/'within_person_bootstrap.csv',index=False); wp.to_csv(out/'within_person_predictions.csv.gz',index=False,compression='gzip'); wa.to_csv(out/'within_person_alpha_selection.csv',index=False)
 print('PASS: robustness analysis')
if __name__=='__main__': main()
