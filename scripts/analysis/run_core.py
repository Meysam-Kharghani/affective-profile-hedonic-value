#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

AFFECTS = ['joy','inspiration','tenderness','sadness','fear','disgust','arousal','familiarity']
OUTCOMES = ['valence','liking']


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonicalize(path: Path):
    d = pd.read_csv(path)
    required = ['participant_id','stimulus_code','emotion_category',*AFFECTS,*OUTCOMES]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise RuntimeError(f'Missing columns: {missing}')
    if len(d) != 2226 or d.participant_id.nunique() != 53:
        raise RuntimeError('Expected 2,226 labeled rows from 53 participants before complete-case filtering.')
    x = pd.DataFrame({
        'subject_id': d.participant_id.astype(str),
        'stimulus_id': d.stimulus_code.astype(str),
        'context': 'unresolved',
        'emotion_category': d.emotion_category.astype(str),
    })
    x['subject_number'] = x.subject_id.str.extract(r'(\d+)')[0].astype(int)
    for c in AFFECTS + OUTCOMES:
        x[c] = pd.to_numeric(d[c], errors='coerce')
    x = x.loc[~x[AFFECTS + OUTCOMES].isna().any(axis=1)].reset_index(drop=True)
    x.insert(0, 'canonical_row_index', np.arange(len(x), dtype=int))
    x['is_analysis_eligible'] = True
    if len(x) != 2225:
        raise RuntimeError(f'Expected 2,225 complete-case rows; found {len(x)}.')
    return x


def pooled(pred):
    rows=[]
    for (outcome, model), g in pred.groupby(['outcome','model'], sort=False):
        y=g.y.to_numpy(float); p=g.pred.to_numpy(float)
        sse=((y-p)**2).sum(); sst=((y-y.mean())**2).sum()
        rows.append({'outcome':outcome,'model':model,'n':len(g),'r2':1-sse/sst,'mae':np.abs(y-p).mean(),'pearson_r':np.corrcoef(y,p)[0,1]})
    return pd.DataFrame(rows)


def subject_stats(pred,outcome,model,subjects):
    g=pred[(pred.outcome==outcome)&(pred.model==model)]
    rows=[]
    for sid in subjects:
        z=g[g.subject_id==sid]; y=z.y.to_numpy(float); p=z.pred.to_numpy(float)
        rows.append([sid,len(z),y.sum(),(y*y).sum(),((y-p)**2).sum(),np.abs(y-p).sum()])
    return pd.DataFrame(rows,columns=['sid','n','sy','sy2','sse','sae']).set_index('sid').loc[subjects]


def metric(s,c):
    n=float(c@s.n); sy=float(c@s.sy); sy2=float(c@s.sy2); sse=float(c@s.sse); sae=float(c@s.sae); sst=sy2-sy*sy/n
    return 1-sse/sst,sae/n


def contrasts(pred,nboot,seed):
    rng=np.random.default_rng(seed)
    pairs=[('M2_affective','M1_nominal'),('M3_affective_nominal','M1_nominal'),('M3_affective_nominal','M2_affective')]
    rows=[]
    for outcome in OUTCOMES:
        subjects=np.array(sorted(pred[pred.outcome==outcome].subject_id.unique()))
        cache={m:subject_stats(pred,outcome,m,subjects) for m in pred[pred.outcome==outcome].model.unique()}
        ones=np.ones(len(subjects),int)
        for a,b in pairs:
            pa,pb=metric(cache[a],ones),metric(cache[b],ones); dr=[]; dm=[]
            for _ in range(nboot):
                c=np.bincount(rng.integers(0,len(subjects),len(subjects)),minlength=len(subjects))
                aa,bb=metric(cache[a],c),metric(cache[b],c); dr.append(aa[0]-bb[0]); dm.append(bb[1]-aa[1])
            rows.append({'outcome':outcome,'contrast':f'{a}_minus_{b}','delta_r2':pa[0]-pb[0],'delta_r2_ci_low':np.quantile(dr,.025),'delta_r2_ci_high':np.quantile(dr,.975),'mae_improvement':pb[1]-pa[1],'mae_improvement_ci_low':np.quantile(dm,.025),'mae_improvement_ci_high':np.quantile(dm,.975)})
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--behavior', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--n-jobs', type=int, default=-1)
    ap.add_argument('--bootstrap', type=int, default=5000)
    args=ap.parse_args()
    root=Path(__file__).resolve().parents[2]
    module=load_module(root/'src/analysis_pipeline/02_global_mapping_and_personalization.py','global_mapping')
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    df=canonicalize(Path(args.behavior)); df.to_csv(out/'canonical_core.csv',index=False)
    lp,la=module.execute_global_lopo(df,args.n_jobs)
    lp.to_csv(out/'lopo_predictions.csv.gz',index=False,compression='gzip')
    la.to_csv(out/'lopo_alpha_selection.csv',index=False)
    pooled(lp).to_csv(out/'lopo_model_summary.csv',index=False)
    contrasts(lp,args.bootstrap,module.SEED).to_csv(out/'lopo_bootstrap_contrasts.csv',index=False)
    sp,sa=module.execute_stimulus_loso(df,args.n_jobs)
    sp.to_csv(out/'heldout_stimulus_predictions.csv.gz',index=False,compression='gzip')
    sa.to_csv(out/'heldout_stimulus_alpha_selection.csv',index=False)
    pooled(sp).to_csv(out/'heldout_stimulus_model_summary.csv',index=False)
    print(f'PASS: core analysis ({len(df)} trials; {df.subject_id.nunique()} participants)')


if __name__=='__main__':
    main()
