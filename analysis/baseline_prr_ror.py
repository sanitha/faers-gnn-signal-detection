import os as _o; ROOT = (_o.environ.get('PROJECT_ROOT') or _o.path.abspath(_o.path.join(_o.path.dirname(_o.path.abspath(globals().get('__file__', 'analysis/x.py'))), '..'))).rstrip('/') + '/'  # repo root; override with PROJECT_ROOT
"""Pilot PRR/ROR baseline under the rolling-origin protocol (counts from years < Y only).
Candidates: all L01 drug x event pairs with event seen in any FAERS report before Y, prior count < N.
Test set A excludes SIDER pairs; test set B excludes SIDER and OFFSIDES pairs. Outputs baseline_prr_ror.csv"""
import pandas as pd, numpy as np, re, sys
from sklearn.metrics import average_precision_score, roc_auc_score
R=ROOT; B=R; D=R+'datasets/'
x=pd.read_pickle(B+'analysis/l01_events.pkl')           # L01 report-event rows (dn, event, year, report_id, num_drugs)
full=pd.read_csv(D+'fda_adverse_events_2015_2026_CLEAN.csv',usecols=['report_id','year','reactions'],low_memory=False)
fe=full.assign(event=full.reactions.str.split('; ')).explode('event'); fe['event']=fe.event.str.lower().str.strip(); fe=fe[fe.event.notna()&fe.event.ne('')]
rep_year=full.set_index('report_id').year
# SIDER and OFFSIDES pair sets (same normalisation as build scripts)
exec(open(R+'analysis/build_l01_events.py').read().split('d=pd.read_csv')[0]); B=R
se=pd.read_csv(D+'sider_4.1/meddra_all_se.tsv.gz',sep='\t',header=None,names=['c1','c2','u1','mtype','u2','event'],compression='gzip')
se=se[se.mtype=='PT'].merge(nm.rename(columns={'cid':'c1'}),on='c1')
se['dn']=se.name.str.lower().str.strip().map(lambda n:SYN.get(n,n)).map(key); se['event']=se.event.str.lower().str.strip()
sider=set(zip(se.dn,se.event))
o=pd.read_csv(D+'offsides_twosides/OFFSIDES.csv.gz',compression='gzip',usecols=['drug_concept_name','condition_concept_name'])
o['dn']=o.drug_concept_name.str.lower().str.replace(r'\b(hydrochloride|hcl|sodium|sulfate|mesylate|anhydrous|disodium|phosphate|citrate|acetate|maleate)\b','',regex=True).str.replace(r'\s+',' ',regex=True).str.strip()
offs=set(zip(o.dn,o.condition_concept_name.str.lower().str.strip()))
drugs=sorted(x.dn.unique()); dix={d:i for i,d in enumerate(drugs)}
res=[]
def pr_at_k(y,s,K):
    idx=np.argsort(-s,kind='stable')[:K]; return y[idx].sum()/K, y[idx].sum()/max(y.sum(),1)
for N in (3,5):
  for Y in range(2019,2026):
    prior=x[x.year<Y]; cur=x[x.year==Y]
    events=pd.Index(fe[fe.year<Y].event.unique()); eix={e:i for i,e in enumerate(events)}
    nrep=(full.year<Y).sum()
    Eb=fe[fe.year<Y].groupby('event').report_id.nunique().reindex(events).values.astype(float)
    Da=prior.groupby('dn').report_id.nunique().reindex(drugs).fillna(0).values.astype(float)
    A=np.zeros((len(drugs),len(events)))
    p=prior[prior.event.isin(eix)].groupby(['dn','event']).size()
    for (dn,ev),c in p.items(): A[dix[dn],eix[ev]]=c
    a=A; Dd=Da[:,None]; E=Eb[None,:]
    with np.errstate(divide='ignore',invalid='ignore'):
        prr=(a/Dd)/((E-a)/(nrep-Dd)); ror=(a*(nrep-Dd-E+a))/((Dd-a)*(E-a))
    prr=np.nan_to_num(prr,nan=0,posinf=0); ror=np.nan_to_num(ror,nan=0,posinf=0)
    C=np.zeros_like(A); q=cur[cur.event.isin(eix)].groupby(['dn','event']).size()
    for (dn,ev),c in q.items(): C[dix[dn],eix[ev]]=c
    # pairs whose event is outside the vocabulary are unreachable positives
    all_cur=cur.groupby(['dn','event']).size(); all_prior=prior.groupby(['dn','event']).size().reindex(all_cur.index).fillna(0)
    unreach=int(((all_prior<N)&(all_cur>=N)&~pd.Series([(a_,b_) in sider for a_,b_ in all_cur.index],index=all_cur.index)&~pd.Series([e not in eix for _,e in all_cur.index],index=all_cur.index)==False).sum()) if False else None
    elig=(A<N)
    s_mask=np.zeros_like(A,dtype=bool); o_mask=np.zeros_like(A,dtype=bool)
    ev_list=list(events)
    # known-pair masks
    for (dn,ev) in sider:
        if dn in dix and ev in eix: s_mask[dix[dn],eix[ev]]=True
    for (dn,ev) in offs:
        if dn in dix and ev in eix: o_mask[dix[dn],eix[ev]]=True
    for name,mask in (('A',elig&~s_mask),('B',elig&~s_mask&~o_mask),('A_seen',elig&~s_mask&(A>=1)),('B_seen',elig&~s_mask&~o_mask&(A>=1))):
        y=((C>=N)&mask)[mask].astype(int); 
        if y.sum()==0: continue
        row={'N':N,'Y':Y,'test_set':name,'candidates':int(mask.sum()),'positives':int(y.sum()),'base_rate':float('%.3g'%y.mean()),'prior0_positives':int((((A==0)&(C>=N))&mask).sum())}
        for nm_,sc in (('PRR',prr),('ROR',ror)):
            s=sc[mask]; row[nm_+'_AUPRC']=round(average_precision_score(y,s),4); row[nm_+'_AUROC']=round(roc_auc_score(y,s),4)
            for K in (10,50,100):
                pk,rk=pr_at_k(y,s,K); row[f'{nm_}_P@{K}']=round(pk,3); row[f'{nm_}_R@{K}']=round(rk,3)
        res.append(row); print(N,Y,name,row['positives'],row['PRR_AUPRC'],row['ROR_AUPRC'],flush=True)
r=pd.DataFrame(res); r.to_csv(B+'analysis/baseline_prr_ror.csv',index=False)
print(r.groupby(['N','test_set'])[['positives','base_rate','PRR_AUPRC','ROR_AUPRC','PRR_AUROC','ROR_AUROC']].mean().round(4).to_string())
