"""Error analysis, seen setting, count label, N=3, test set A. R-GCN run D and A (mean of seeds 1,4,5) vs event-frequency ranking.
Outputs error_analysis.csv (per-positive ranks) and prints summaries."""
import pandas as pd, numpy as np, pickle
from harness import HERE
F = pickle.load(open(HERE + 'folds_seen.pkl', 'rb')); SEEDS = (1, 4, 5); rows = []
for Y in range(2019, 2026):
    f = F[Y].reset_index(drop=True)
    mk = f.test_A.values; g = f[mk].copy()
    for run in ('A', 'D'):
        s = np.mean([np.load(HERE + f'rgcn_scores/Y{Y}_{run}_s{sd}.npy') for sd in SEEDS], axis=0)[mk]
        g['rank_' + run] = pd.Series(-s, index=g.index).rank(method='first')
    g['rank_event_freq'] = (-g.event_prior_reports).rank(method='first'); g['n'] = len(g); g['Y'] = Y
    rows.append(g)
d = pd.concat(rows); d.to_csv('error_analysis_all.csv', index=False)
p = d[d.y_count].copy()
for k in ('A', 'D', 'event_freq'): p['pct_' + k] = p['rank_' + k] / p.n
print('positives', len(p), 'candidates/fold', int(d.groupby('Y').n.first().mean()))
for k in ('A', 'D', 'event_freq'):
    print(k, 'median percentile rank of positives', round(p['pct_' + k].median(), 3), '| share in top 1%', round((p['pct_' + k] <= .01).mean(), 3), 'top 5%', round((p['pct_' + k] <= .05).mean(), 3), 'top 10%', round((p['pct_' + k] <= .10).mean(), 3), 'top 50%', round((p['pct_' + k] <= .5).mean(), 3))
for k in ('A', 'D', 'event_freq'):
    print(k, 'positives within top 50 per fold (mean)', round((p['rank_' + k] <= 50).sum() / 7, 2), '| top 100', round((p['rank_' + k] <= 100).sum() / 7, 2), '| top 500', round((p['rank_' + k] <= 500).sum() / 7, 2))
# who is missed: bottom half under run D
p['missed'] = p.pct_D > 0.5
print('missed (bottom half) share', round(p.missed.mean(), 3))
p['prior1'] = p.pair_prior == 1
print('by pair_prior (1 vs 2): share missed', p.groupby('pair_prior').missed.mean().round(3).to_dict(), 'n', p.groupby('pair_prior').size().to_dict())
p['ev_tercile'] = pd.qcut(p.event_prior_reports, 3, labels=['rare event', 'mid event', 'common event']); p['drug_tercile'] = pd.qcut(p.drug_prior_reports, 3, labels=['small drug', 'mid drug', 'large drug'])
print('by event frequency:', p.groupby('ev_tercile', observed=True).agg(n=('missed', 'size'), missed=('missed', 'mean'), median_pct=('pct_D', 'median')).round(3).to_dict('index'))
print('by drug volume:', p.groupby('drug_tercile', observed=True).agg(n=('missed', 'size'), missed=('missed', 'mean'), median_pct=('pct_D', 'median')).round(3).to_dict('index'))
print('in OFFSIDES (test A only):', p.groupby('in_offsides').agg(n=('missed', 'size'), missed=('missed', 'mean'), median_pct=('pct_D', 'median')).round(3).to_dict('index'))
# top-ranked false positives under D
fp = d[(~d.y_count) & (d.rank_D <= 100)]
print('top-100 false positives per fold:', len(fp) / 7, '| median pair_prior', fp.pair_prior.median(), '| top drugs', fp.drug.value_counts().head(5).to_dict(), '| top events', fp.event.value_counts().head(5).to_dict())
tp = d[(d.y_count) & (d.rank_D <= 100)]
print('top-100 true positives total', len(tp), '| drugs', tp.drug.value_counts().head(5).to_dict(), '| events', tp.event.value_counts().head(5).to_dict())
