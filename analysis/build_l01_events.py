import os as _o; ROOT = (_o.environ.get('PROJECT_ROOT') or _o.path.abspath(_o.path.join(_o.path.dirname(_o.path.abspath(globals().get('__file__', 'analysis/x.py'))), '..'))).rstrip('/') + '/'  # repo root; override with PROJECT_ROOT
"""Match FAERS suspect drugs to SIDER ATC L01 drugs, explode events, flag SIDER label pairs.
Outputs: l01_drug_map.csv (audit table), l01_events.pkl, positives_rolling_origin.csv"""
import pandas as pd, re
B=ROOT + 'datasets/'
atc=pd.read_csv(B+'sider_4.1/drug_atc.tsv',sep='\t',header=None,names=['cid','atc'])
nm=pd.read_csv(B+'sider_4.1/drug_names.tsv',sep='\t',header=None,names=['cid','name'])
l01=atc[atc.atc.str.startswith('L01')].merge(nm,on='cid')
# manual synonyms: SIDER name -> canonical generic used in FAERS (documented in report appendix)
SYN={'actinomycin':'dactinomycin','retinoic':'tretinoin','5-azacytidine':'azacitidine',"5-aza-2'-deoxycytidine":'decitabine',
     '5-fu':'fluorouracil','bcnu':'carmustine','hexamethylmelamine':'altretamine','mthpc':'temoporfin','photofrin':'porfimer',
     'arsenic':'arsenic trioxide'}
EXCLUDE={'methyl','anthracycline','5-aminolevulinic'}   # not a specific L01 drug / false-match prone
canon={SYN.get(n,n) for n in l01.name.str.lower().str.strip()}-EXCLUDE
STOP={'hydrochloride','hcl','sodium','disodium','sulfate','sulphate','mesylate','maleate','tartrate','phosphate','citrate',
      'acetate','besylate','tosylate','malate','anhydrous','injection','solution','for','tablet','tablets','film-coated',
      'capsule','capsules','oral','intravenous','comparator','dihydrate','trihydrate','monohydrate','hydrate','bromide',
      'chloride','nitrate','succinate','fumarate','lactate','pamoate','and','powder','lyophilized','concentrate'}
def key(s):
    s=re.sub(r'[\^\(\),\.]',' ',s.lower()); toks=[t for t in re.split(r'[\s]+',s) if t and t not in STOP]
    return ' '.join(toks)
d=pd.read_csv(B+'fda_adverse_events_2015_2026_CLEAN.csv',usecols=['report_id','year','suspect_drug','reactions','num_drugs'],low_memory=False)
import re as _re
CANON_SORTED=sorted(canon,key=len,reverse=True)
def match(s):
    """Split multi-name entries on ; / , then accept a part if it starts with a canonical L01 name (word boundary)."""
    hits=set()
    for part in _re.split(r'[;/]',str(s)):
        k=key(part)
        for c in CANON_SORTED:
            if k==c or k.startswith(c+' '):
                hits.add(c); break
    return hits
m=d.suspect_drug.map(match)
d['n_hits']=m.map(len)
print('reports matching >1 distinct L01 drug (excluded as ambiguous):',int((d.n_hits>1).sum()))
d['dn']=m.map(lambda h: next(iter(h)) if len(h)==1 else None)
d=d[d.dn.notna()].copy()
d.groupby('dn').agg(reports=('report_id','size'),raw_name_variants=('suspect_drug','nunique')).sort_values('reports',ascending=False).to_csv('l01_drug_map.csv')
print('L01 canonical names',len(canon),'matched in FAERS',d.dn.nunique(),'reports',len(d),'unique report ids',d.report_id.nunique())
print(d.groupby('year').size().to_string())
x=d.assign(event=d.reactions.str.split('; ')).explode('event')
x['event']=x.event.str.lower().str.strip(); x=x[x.event.ne('')]
se=pd.read_csv(B+'sider_4.1/meddra_all_se.tsv.gz',sep='\t',header=None,names=['c1','c2','u1','mtype','u2','event'],compression='gzip')
se=se[se.mtype=='PT'].merge(nm.rename(columns={'cid':'c1'}),on='c1')
se['dn']=se.name.str.lower().str.strip().map(lambda n:SYN.get(n,n)).map(key); se['event']=se.event.str.lower().str.strip()
sp=set(zip(se.dn,se.event))
x['in_sider']=[(a,b) in sp for a,b in zip(x.dn,x.event)]
print('rows',len(x),'share in SIDER',x.in_sider.mean().round(3),'| 5+ drug reports',(d.num_drugs>=5).mean().round(3))
x.to_pickle('l01_events.pkl')
cnt=x.groupby(['dn','event','year']).size().unstack(fill_value=0)
ins=x.groupby(['dn','event']).in_sider.max().reindex(cnt.index)
poly=x[x.num_drugs>=5].groupby(['dn','event','year']).size().unstack(fill_value=0).reindex(cnt.index,fill_value=0)
out=[]
for N in (3,5,10):
    for Y in range(2019,2026):
        prior=cnt[[c for c in cnt.columns if c<Y]].sum(axis=1)
        pos=(prior<N)&(cnt[Y]>=N)&(~ins)
        out.append((N,Y,int(pos.sum()),int((pos&(poly[Y]>=N)).sum()) ))
r=pd.DataFrame(out,columns=['N','Y','positives','pos_with_N_reports_from_5plus_drug_reports'])
print(r.pivot(index='Y',columns='N',values='positives').to_string()); print(r.groupby('N').positives.sum().to_string())
r.to_csv('positives_rolling_origin.csv',index=False); print('pairs',len(cnt))
