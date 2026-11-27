import os as _o; ROOT = (_o.environ.get('PROJECT_ROOT') or _o.path.abspath(_o.path.join(_o.path.dirname(_o.path.abspath(globals().get('__file__', 'analysis/x.py'))), '..'))).rstrip('/') + '/'  # repo root; override with PROJECT_ROOT
"""Draw schematic figures: pipeline, graph schema, temporal split, ablation, gantt."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.patches as P
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import datetime as dt, matplotlib.dates as md
F=ROOT + 'figures/'
BL,RD,GR,LB='#1f4e79','#c0504d','#7f7f7f','#dce6f1'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
def box(ax,x,y,w,h,t,fc=LB,ec=BL,fs=8.5,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.08',fc=fc,ec=ec,lw=1.2))
    ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs,weight='bold' if bold else 'normal',wrap=True)
def arr(ax,a,b,c=GR,ls='-'):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,color=c,lw=1.3,linestyle=ls))
# pipeline
fig,ax=plt.subplots(figsize=(7.2,3.9)); ax.set_xlim(0,10); ax.set_ylim(0,5.4); ax.axis('off')
box(ax,0.1,3.9,2.2,1.2,'FAERS sample\n528,000 reports\n2015-2025'); box(ax,0.1,2.2,2.2,1.2,'SIDER 4.1\nlabel pairs'); box(ax,0.1,0.5,2.2,1.2,'OFFSIDES\ncleaned signals')
box(ax,3.0,2.2,2.0,2.2,'Preprocessing\nATC L01 filter\n(73 drugs)\nname and event\nnormalisation',fc='#fff2cc',ec='#bf9000')
box(ax,5.7,3.6,2.0,1.5,'Temporal protocol\nyears < Y for features\nyear Y for labels',fc='#fff2cc',ec='#bf9000')
box(ax,5.7,1.5,2.0,1.5,'Models\nPRR, ROR, XGBoost,\nnode2vec, R-GCN',fc='#e2efd9',ec='#548235')
box(ax,8.2,2.2,1.7,2.2,'Evaluation\nAUPRC, AUROC,\nP@K, R@K, MRR\nablation A-D\npolypharmacy',fc='#fce4d6',ec=RD)
for s in [(2.3,4.5),(2.3,2.8),(2.3,1.1)]: arr(ax,s,(3.0,3.3))
arr(ax,(5.0,3.6),(5.7,4.3)); arr(ax,(5.0,3.0),(5.7,2.4)); arr(ax,(6.7,3.6),(6.7,3.0)); arr(ax,(7.7,2.3),(8.2,3.0))
fig.tight_layout(); fig.savefig(F+'fig_pipeline.png',dpi=180); plt.close()
# graph schema
fig,ax=plt.subplots(figsize=(6.6,3.8)); ax.set_xlim(0,10); ax.set_ylim(0,6); ax.axis('off')
box(ax,0.4,2.0,2.3,1.9,'Drug\nATC class,\nreport count,\npolypharmacy share',bold=True,fs=7.5)
box(ax,7.3,2.0,2.3,1.9,'Adverse event\nMedDRA term,\nfrequency',bold=True,fs=7.5)
box(ax,3.85,4.4,2.3,1.3,'Report\nseriousness, drug count\n(years < Y only)',bold=True,fs=8,fc='#e2efd9',ec='#548235')
arr(ax,(2.0,3.7),(3.85,5.0),BL); ax.text(2.2,4.7,'drug in report',color=BL,fontsize=8)
arr(ax,(6.15,5.0),(8.0,3.7),BL); ax.text(7.2,4.7,'event in report',color=BL,fontsize=8)
arr(ax,(2.7,3.2),(7.3,3.2),'#bf9000'); ax.text(3.6,3.35,'SIDER label pair (static)',color='#bf9000',fontsize=8)
arr(ax,(2.7,2.6),(7.3,2.6),RD); ax.text(3.2,2.15,'OFFSIDES signal (static)',color=RD,fontsize=8)
arr(ax,(2.7,2.9),(7.3,2.9),'black',ls='--'); ax.text(3.4,1.4,'Prediction target: drug-event pair\nthat becomes frequent in year Y (dashed)',fontsize=8)
fig.tight_layout(); fig.savefig(F+'fig_graph_schema.png',dpi=180); plt.close()
# temporal split
fig,ax=plt.subplots(figsize=(7.2,3.6)); ax.set_xlim(2014.5,2026.5); ax.set_ylim(0,8); 
for k,Y in enumerate(range(2019,2026)):
    y=7-k
    ax.barh(y,Y-2015,left=2015,height=0.55,color=BL,alpha=.85); ax.barh(y,1,left=Y,height=0.55,color=RD)
    ax.text(2015.1,y,f'fold Y={Y}',color='white',fontsize=7.5,va='center')
ax.set_yticks([]); ax.set_xticks(range(2015,2026)); ax.set_xlabel('Report year'); 
for s in ['top','right','left']: ax.spines[s].set_visible(False)
ax.legend(handles=[P.Patch(color=BL,label='History: features, edges, prior counts (years < Y)'),P.Patch(color=RD,label='Test year Y: labels (pair count >= N)')],loc='lower right',frameon=False,fontsize=8,bbox_to_anchor=(1,-0.02))
fig.tight_layout(); fig.savefig(F+'fig_temporal_split.png',dpi=180); plt.close()
# ablation
fig,ax=plt.subplots(figsize=(6.0,2.6)); ax.axis('off'); rows=['Run A','Run B','Run C','Run D']; cols=['FAERS','SIDER','OFFSIDES']; M=[[1,0,0],[1,1,0],[1,0,1],[1,1,1]]
ax.set_xlim(0,5); ax.set_ylim(0,5)
for j,c in enumerate(cols): ax.text(1.6+j*1.1,4.4,c,ha='center',weight='bold')
for i,r in enumerate(rows):
    ax.text(0.5,3.5-i*0.9,r,ha='center',va='center',weight='bold')
    for j in range(3):
        on=M[i][j]; ax.add_patch(FancyBboxPatch((1.15+j*1.1,3.2-i*0.9),0.9,0.6,boxstyle='round,pad=0.02',fc=BL if on else '#f2f2f2',ec=BL)); ax.text(1.6+j*1.1,3.5-i*0.9,'yes' if on else 'no',ha='center',va='center',color='white' if on else GR)
fig.tight_layout(); fig.savefig(F+'fig_ablation.png',dpi=180); plt.close()
# gantt
T=[('Interim report (Ch 1-3, EDA)','2026-11-14','2026-11-15',BL),('Supervisor meeting','2026-11-17','2026-11-17',RD),('Baselines: PRR, ROR, XGBoost','2026-11-16','2026-11-18',BL),
   ('node2vec and R-GCN','2026-11-18','2026-11-21',BL),('Ablation A-D and leakage control','2026-11-22','2026-11-24',BL),('Polypharmacy analysis','2026-11-23','2026-11-25',BL),
   ('Dissertation writing and figures','2026-11-22','2026-11-29',GR),('Final checks and submission','2026-11-30','2026-12-01',RD)]
fig,ax=plt.subplots(figsize=(7.4,3.6))
for i,(n,s,e,c) in enumerate(T):
    s=dt.date.fromisoformat(s); e=dt.date.fromisoformat(e)+dt.timedelta(days=1); ax.barh(len(T)-i,(e-s).days,left=md.date2num(s),color=c,height=0.55)
ax.set_yticks(range(len(T),0,-1)); ax.set_yticklabels([t[0] for t in T],fontsize=8); ax.xaxis_date(); ax.xaxis.set_major_formatter(md.DateFormatter('%d %b')); ax.xaxis.set_major_locator(md.DayLocator(interval=2))
ax.axvline(md.date2num(dt.date(2026,11,22)),color=RD,ls='--',lw=1); ax.text(md.date2num(dt.date(2026,11,22))+0.1,0.1,'R-GCN go/no-go',color=RD,fontsize=7.5)
ax.grid(axis='x',alpha=.3); ax.tick_params(axis='x',labelsize=7.5)
for s in ['top','right']: ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(F+'fig_gantt.png',dpi=180); plt.close()
