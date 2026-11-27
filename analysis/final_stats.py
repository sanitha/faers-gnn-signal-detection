"""Final statistics for Chapter 5. Per (setting, label, test): mean over 7 folds of seed-averaged AUPRC for every model,
lift over base rate, and paired comparisons (mean diff, ratio, wins/7, Wilcoxon p) between key pairs of models.
Outputs final_summary.csv and final_paired.csv"""
import pandas as pd, numpy as np, glob, os
from scipy.stats import wilcoxon
def load_rgcn(setting, label):
    fs = []
    if (setting, label) == ('seen', 'y_count'):
        fs = ['rgcn_seeds_p1.csv', 'rgcn_seeds_p2.csv', 'rgcn_nograph.csv']
    else:
        fs = sorted(glob.glob(f'rgcn_{setting}_{label}_*.csv'))
    d = pd.concat([pd.read_csv(f) for f in fs if os.path.exists(f)])
    d = d.drop_duplicates(['Y', 'run', 'seed', 'test'], keep='last'); d['model'] = 'rgcn_' + d.run.replace({'A': 'A', 'sider': 'B', 'offs': 'C', 'D': 'D', 'nograph': 'nograph'})
    return d[['Y', 'model', 'seed', 'test', 'pos', 'base', 'AUPRC', 'AUROC', 'P50']]
def load_xgb(setting, label):
    d = pd.read_csv('xgb_results.csv'); d = d[(d.setting == setting) & (d.label == label)].copy()
    d['model'] = 'xgb_' + d.run.str[0]; d['seed'] = 0; return d[['Y', 'model', 'seed', 'test', 'pos', 'base', 'AUPRC', 'AUROC', 'P50']]
def load_base(setting, label):
    d = pd.read_csv('seen_cold_baselines.csv'); d = d[(d.setting == setting) & (d.label == label) & (d.N == 3)].copy()
    d['model'] = d.baseline; d['seed'] = 0; d['P50'] = d.P50; return d[['Y', 'model', 'seed', 'test', 'pos', 'base', 'AUPRC', 'AUROC', 'P50']]
summ, paired = [], []
PAIRS = [('rgcn_D', 'event_freq'), ('rgcn_D', 'xgb_D'), ('rgcn_A', 'xgb_A'), ('rgcn_D', 'rgcn_nograph'), ('rgcn_A', 'rgcn_nograph'),
         ('rgcn_D', 'rgcn_A'), ('rgcn_B', 'rgcn_A'), ('rgcn_C', 'rgcn_A'), ('xgb_D', 'xgb_A'), ('xgb_D', 'event_freq')]
for setting in ('seen', 'cold'):
    for label in ('y_count', 'y_evans'):
        d = pd.concat([load_rgcn(setting, label), load_xgb(setting, label), load_base(setting, label)])
        nseed = d[d.model.str.startswith('rgcn')].groupby('model').seed.nunique().to_dict()
        for ts in ('test_A', 'test_B'):
            x = d[d.test == ts]; f = x.groupby(['model', 'Y']).agg(AUPRC=('AUPRC', 'mean'), AUROC=('AUROC', 'mean'), P50=('P50', 'mean'), base=('base', 'first'), pos=('pos', 'first')).reset_index()
            m = f.groupby('model').agg(AUPRC=('AUPRC', 'mean'), AUROC=('AUROC', 'mean'), P50=('P50', 'mean'), base=('base', 'mean'), pos=('pos', 'mean'), folds=('Y', 'count'))
            m['lift'] = m.AUPRC / m.base; m['seeds'] = [nseed.get(i, 1) for i in m.index]
            for k, r in m.iterrows(): summ.append(dict(setting=setting, label=label, test=ts, model=k, **r.to_dict()))
            P = f.pivot(index='Y', columns='model', values='AUPRC')
            for a, b in PAIRS:
                if a in P and b in P:
                    z = P[[a, b]].dropna(); dd = (z[a] - z[b]).values
                    if len(dd) < 5: continue
                    p = wilcoxon(dd).pvalue if np.any(dd != 0) else 1.0
                    paired.append(dict(setting=setting, label=label, test=ts, a=a, b=b, folds=len(dd), mean_a=z[a].mean(), mean_b=z[b].mean(), ratio=z[a].mean() / z[b].mean(), wins=int((dd > 0).sum()), p=p))
S = pd.DataFrame(summ); S.to_csv('final_summary.csv', index=False); Pp = pd.DataFrame(paired); Pp.to_csv('final_paired.csv', index=False)
pd.set_option('display.width', 220); pd.set_option('display.max_rows', 500)
print(S[S.model.str.startswith(('rgcn', 'xgb'))].pivot_table(index=['setting', 'label', 'test'], columns='model', values='AUPRC').round(4).to_string())
print(Pp.round(4).to_string())
