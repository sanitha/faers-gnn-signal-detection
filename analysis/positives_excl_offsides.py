import os as _o; ROOT = (_o.environ.get('PROJECT_ROOT') or _o.path.abspath(_o.path.join(_o.path.dirname(_o.path.abspath(globals().get('__file__', 'analysis/x.py'))), '..'))).rstrip('/') + '/'  # repo root; override with PROJECT_ROOT
import pandas as pd
B=ROOT + 'datasets/'
o=pd.read_csv(B+'offsides_twosides/OFFSIDES.csv.gz',compression='gzip',usecols=['drug_concept_name','condition_concept_name'])
x=pd.read_pickle('l01_events.pkl')
o['dn']=o.drug_concept_name.str.lower().str.replace(r'\b(hydrochloride|hcl|sodium|sulfate|mesylate|anhydrous|disodium|phosphate|citrate|acetate|maleate)\b','',regex=True).str.replace(r'\s+',' ',regex=True).str.strip()
op=set(zip(o.dn,o.condition_concept_name.str.lower().str.strip()))
cnt=x.groupby(['dn','event','year']).size().unstack(fill_value=0)
ins=x.groupby(['dn','event']).in_sider.max().reindex(cnt.index)
offs=pd.Series([(a,b) in op for a,b in cnt.index],index=cnt.index)
rows=[]
for N in (3,5,10):
  for Y in range(2019,2026):
    prior=cnt[[c for c in cnt.columns if c<Y]].sum(axis=1); base=(prior<N)&(cnt[Y]>=N)
    rows.append((N,Y,int((base&~ins).sum()),int((base&~ins&~offs).sum())))
r=pd.DataFrame(rows,columns=['N','Y','not_in_SIDER','not_in_SIDER_or_OFFSIDES'])
print(r.groupby('N')[['not_in_SIDER','not_in_SIDER_or_OFFSIDES']].sum())
print(r[r.N==3].to_string(index=False)); r.to_csv('positives_by_novelty_definition.csv',index=False)
