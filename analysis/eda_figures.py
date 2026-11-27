import os as _o; ROOT = (_o.environ.get('PROJECT_ROOT') or _o.path.abspath(_o.path.join(_o.path.dirname(_o.path.abspath(globals().get('__file__', 'analysis/x.py'))), '..'))).rstrip('/') + '/'  # repo root; override with PROJECT_ROOT
"""EDA figures and stats for the interim report. Outputs to final_thesis/figures and analysis/eda_stats.json"""
import pandas as pd, numpy as np, json, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
B=ROOT; F=B+'final_thesis/figures/'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
C1,C2,C3='#1f4e79','#c0504d','#7f7f7f'
x=pd.read_pickle(B+'analysis/l01_events.pkl')
full=pd.read_csv(B+'datasets/fda_adverse_events_2015_2026_CLEAN.csv',usecols=['report_id','year','serious','patient_sex','patient_age_years','age_group','num_drugs','num_reactions','country'],low_memory=False)
rep=x.drop_duplicates('report_id')[['report_id','year','dn','num_drugs']].merge(full[['report_id','serious','patient_sex','patient_age_years','age_group','num_reactions','country']],on='report_id',how='left')
S={}
S['all_reports']=len(full); S['l01_reports']=len(rep); S['l01_share']=round(len(rep)/len(full),4)
S['l01_drugs']=int(rep.dn.nunique()); S['pairs']=int(x.groupby(['dn','event']).ngroups); S['events']=int(x.event.nunique()); S['event_rows']=len(x)
S['median_num_drugs_l01']=float(rep.num_drugs.median()); S['mean_num_drugs_l01']=round(float(rep.num_drugs.mean()),2)
S['share_5plus']=round(float((rep.num_drugs>=5).mean()),3); S['share_5plus_all']=round(float((full.num_drugs>=5).mean()),3)
S['serious_yes']=round(float((rep.serious=='Yes').mean()),3)
S['sex']=rep.patient_sex.value_counts(normalize=True).round(3).to_dict()
S['age_median']=float(rep.patient_age_years.median()); S['age_missing']=round(float(rep.patient_age_years.isna().mean()),3)
S['top_countries']=rep.country.value_counts(normalize=True).head(5).round(3).to_dict()
S['in_sider_rows']=round(float(x.in_sider.mean()),3)
# Fig1 reports per year
fig,ax=plt.subplots(figsize=(6.2,3.4)); y=rep.groupby('year').size(); ax.bar(y.index,y.values,color=C1)
for a,b in y.items(): ax.text(a,b+25,str(b),ha='center',fontsize=7)
ax.set_xlabel('Year of report'); ax.set_ylabel('L01 reports'); ax.set_xticks(y.index); fig.tight_layout(); fig.savefig(F+'fig_reports_per_year.png'); plt.close()
S['reports_per_year']=y.to_dict()
# Fig2 top drugs
d=rep.dn.value_counts().head(15)[::-1]; fig,ax=plt.subplots(figsize=(6.2,4.2)); ax.barh(d.index,d.values,color=[C2 if n in('methotrexate','celecoxib','ofatumumab') else C1 for n in d.index])
ax.set_xlabel('Reports'); fig.tight_layout(); fig.savefig(F+'fig_top_drugs.png'); plt.close()
# Fig3 num_drugs distribution
fig,ax=plt.subplots(figsize=(6.2,3.4)); v=rep.num_drugs.clip(upper=40); ax.hist(v,bins=np.arange(0.5,41.5,1),color=C1); ax.axvline(4.5,color=C2,ls='--'); ax.text(5,ax.get_ylim()[1]*0.9,'5 or more drugs',color=C2,fontsize=8)
ax.set_xlabel('Drugs per report (capped at 40)'); ax.set_ylabel('Reports'); fig.tight_layout(); fig.savefig(F+'fig_num_drugs.png'); plt.close()
S['num_drugs_max']=int(rep.num_drugs.max()); S['num_drugs_gt40']=int((rep.num_drugs>40).sum())
# Fig4 pair frequency
cnt=x.groupby(['dn','event']).size(); S['pairs_1_report']=round(float((cnt==1).mean()),3); S['pairs_ge3']=int((cnt>=3).sum()); S['pairs_ge10']=int((cnt>=10).sum())
fig,ax=plt.subplots(figsize=(6.2,3.4)); ax.hist(cnt.clip(upper=30),bins=np.arange(0.5,31.5,1),color=C1); ax.set_yscale('log'); ax.set_xlabel('Reports per drug-event pair (capped at 30)'); ax.set_ylabel('Pairs (log)'); fig.tight_layout(); fig.savefig(F+'fig_pair_frequency.png'); plt.close()
# Fig5 top events
e=x.event.value_counts().head(15)[::-1]; fig,ax=plt.subplots(figsize=(6.2,4.2)); ax.barh(e.index,e.values,color=C1); ax.set_xlabel('Event mentions'); fig.tight_layout(); fig.savefig(F+'fig_top_events.png'); plt.close()
# Fig6 positives per year by N and test set
p=pd.read_csv(B+'analysis/positives_by_novelty_definition.csv'); p3=p[p.N==3]
fig,ax=plt.subplots(figsize=(6.2,3.6)); w=0.38; ax.bar(p3.Y-w/2,p3.not_in_SIDER,w,color=C1,label='Test set A: not in SIDER'); ax.bar(p3.Y+w/2,p3.not_in_SIDER_or_OFFSIDES,w,color=C2,label='Test set B: not in SIDER or OFFSIDES')
ax.set_xlabel('Test year Y (N = 3)'); ax.set_ylabel('Positive pairs'); ax.legend(frameon=False,fontsize=8); fig.tight_layout(); fig.savefig(F+'fig_positives_by_year.png'); plt.close()
# Fig7 OFFSIDES leakage share
r=p.groupby('N')[['not_in_SIDER','not_in_SIDER_or_OFFSIDES']].sum(); r['leak']=1-r.not_in_SIDER_or_OFFSIDES/r.not_in_SIDER; S['offsides_leak_share_by_N']=r.leak.round(3).to_dict()
fig,ax=plt.subplots(figsize=(5.2,3.4)); ax.bar([f'N={n}' for n in r.index],r.leak*100,color=C2); 
for i,v in enumerate(r.leak*100): ax.text(i,v+1,f'{v:.0f}%',ha='center')
ax.set_ylabel('SIDER-novel positives also in OFFSIDES (%)'); ax.set_ylim(0,100); fig.tight_layout(); fig.savefig(F+'fig_offsides_leakage.png'); plt.close()
# Fig8 missingness
m=full.loc[full.report_id.isin(rep.report_id),['serious','patient_sex','patient_age_years','age_group']].isna().mean()*100
S['missing_l01_pct']=m.round(1).to_dict()
# Fig9 age/sex
fig,ax=plt.subplots(figsize=(6.2,3.4)); ax.hist(rep.patient_age_years.dropna(),bins=np.arange(0,101,5),color=C1); ax.set_xlabel('Patient age (years)'); ax.set_ylabel('Reports'); fig.tight_layout(); fig.savefig(F+'fig_age.png'); plt.close()
json.dump(S,open(B+'analysis/eda_stats.json','w'),indent=1,default=str); print(json.dumps(S,indent=1,default=str))
