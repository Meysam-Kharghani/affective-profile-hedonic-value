#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
R=Path(__file__).resolve().parents[1]
required=[
'DATA_INTEGRITY.md',
'analysis_specification/analysis_plan.json',
'analysis_specification/context_analysis_cohort.csv',
'results/main_tables/Table1_LOPO_model_performance.csv',
'results/main_tables/Table5_analysis_status.csv',
'results/main_tables/Table6_cross_context_transfer.csv',
'results/main_tables/Table7_personalization_support_gate.csv',
'results/supplementary_tables/TableS4_data_integrity_audit.csv',
'results/supplementary_tables/TableS6_split_half_family_stability.csv',
'results/supplementary_tables/TableS10_context_recovery_summary.csv',
]
missing=[x for x in required if not (R/x).exists()]
if missing: raise SystemExit('Missing required files: '+str(missing))
core=pd.read_csv(R/'results/main_tables/Table1_LOPO_model_performance.csv')
assert (core.n==2225).all()
for out,cat,aff in [('valence',0.49,0.67),('liking',0.47,0.68)]:
    g=core[core.outcome==out].set_index('model')
    assert g.loc['M1_nominal','r2']>cat
    assert g.loc['M2_affective','r2']>aff
    assert g.loc['M2_affective','r2']>g.loc['M1_nominal','r2']
cross=pd.read_csv(R/'results/main_tables/Table6_cross_context_transfer.csv')
assert len(cross)==4 and (cross.participants==45).all() and (cross.trials==945).all() and (cross.r2>0.55).all()
pers=pd.read_csv(R/'results/main_tables/Table7_personalization_support_gate.csv')
assert len(pers)==2 and not pers.incremental_personalization_supported.astype(bool).any()
cohort=pd.read_csv(R/'analysis_specification/context_analysis_cohort.csv')
assert cohort.included_in_context_analysis.astype(bool).sum()==45
print('PASS: scientific result verification')
