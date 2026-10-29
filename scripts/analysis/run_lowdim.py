#!/usr/bin/env python3
from __future__ import annotations
import argparse,importlib.util
from pathlib import Path
import pandas as pd
from sklearn.metrics import r2_score,mean_absolute_error

def loadmod(path,name):
 s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--canonical',required=True); ap.add_argument('--output',required=True); ap.add_argument('--bootstrap',type=int,default=5000); a=ap.parse_args()
 root=Path(__file__).resolve().parents[2]; m=loadmod(root/'src/analysis_pipeline/05_low_dimensional_sensitivity.py','low_orig'); df=pd.read_csv(a.canonical)
 if len(df)!=2225 or df.subject_id.nunique()!=53: raise RuntimeError('Core cohort mismatch.')
 df['positive_affect']=df[['joy','inspiration','tenderness']].mean(axis=1); df['negative_affect']=df[['sadness','fear','disgust']].mean(axis=1); df['affect_balance']=df.positive_affect-df.negative_affect
 fitted={}; rows=[]
 for o in m.OUTCOMES:
  for model in m.MODELS:
   g=m.lopo(df,o,model); fitted[o,model]=g; rows.append({'outcome':o,'model':model,'r2':r2_score(g.y,g.pred),'mae':mean_absolute_error(g.y,g.pred),'n_trials':len(g),'n_participants':g.subject_id.nunique()})
 out=pd.DataFrame(rows); extra=[]
 for o in m.OUTCOMES:
  full=fitted[o,'full_affect8']
  for model in m.MODELS:
   if model=='full_affect8': continue
   extra.append({'outcome':o,'model':model,**m.bootstrap_delta(full,fitted[o,model],nboot=a.bootstrap,seed=20261027+sum(map(ord,o+model)))})
 out=out.merge(pd.DataFrame(extra),on=['outcome','model'],how='left'); Path(a.output).parent.mkdir(parents=True,exist_ok=True); out.to_csv(a.output,index=False); print('PASS: low-dimensional sensitivity analysis')
if __name__=='__main__': main()
