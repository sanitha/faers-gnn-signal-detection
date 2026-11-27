"""Paired fold-level comparison (7 folds, Y=2019-2025) of AUPRC; R-GCN averaged over seeds. Seen setting, N=3, y_count."""
import pandas as pd, numpy as np
from scipy.stats import wilcoxon
r = pd.concat([pd.read_csv(f) for f in ['rgcn_seeds_p1.csv', 'rgcn_seeds_p2.csv']])
ng = pd.read_csv('rgcn_nograph.csv'); r = pd.concat([r, ng])
print('seeds per run:', r.groupby('run').seed.nunique().to_dict())
R = r.groupby(['run', 'test', 'Y']).AUPRC.mean()
x = pd.read_csv('xgb_results.csv'); x = x[(x.setting == 'seen') & (x.label == 'y_count')].set_index(['run', 'test', 'Y']).AUPRC
b = pd.read_csv('seen_cold_baselines.csv'); b = b[(b.setting == 'seen') & (b.label == 'y_count') & (b.N == 3)].set_index(['baseline', 'test', 'Y']).AUPRC
def get(name, t):
    if name.startswith('rg:'): return R[name[3:]][t]
    if name.startswith('xgb:'): return x[name[4:]][t]
    return b[name][t]
pairs = [('rg:D', 'event_freq'), ('rg:D', 'xgb:D_all'), ('rg:D', 'rg:nograph'), ('rg:D', 'rg:A'), ('rg:offs', 'rg:A'), ('rg:sider', 'rg:A'), ('rg:A', 'event_freq')]
for t in ['test_A', 'test_B']:
    for a, c in pairs:
        u, v = get(a, t).sort_index(), get(c, t).sort_index(); d = (u - v).values
        p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        print(f'{t} {a:10s} vs {c:11s} mean_diff={d.mean():+.4f} ratio={u.mean()/v.mean():.2f} wins={(d>0).sum()}/7 wilcoxon_p={p:.3f}')
