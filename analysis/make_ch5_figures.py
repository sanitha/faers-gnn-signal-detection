import os as _o; ROOT = (_o.environ.get('PROJECT_ROOT') or _o.path.abspath(_o.path.join(_o.path.dirname(_o.path.abspath(globals().get('__file__', 'analysis/x.py'))), '..'))).rstrip('/') + '/'  # repo root; override with PROJECT_ROOT
"""Chapter 5 figures (seen setting, count label, final data). Palette: reference categorical slots 1-3 (blue, orange, aqua), neutral gray for controls.
Static PNGs for the thesis; direct value labels, recessive grid, no dual axis."""
import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'font.family': 'Times New Roman', 'font.size': 10, 'axes.edgecolor': '#52514e', 'axes.linewidth': .6, 'axes.spines.top': False, 'axes.spines.right': False})
BLUE, ORANGE, AQUA, GRAY, INK, MUTED = '#2a78d6', '#eb6834', '#1baf7a', '#8a8a86', '#0b0b0b', '#52514e'
OUT = ROOT + 'figures/'
def fold_table(setting='seen', label='y_count'):
    r = pd.concat([pd.read_csv(f) for f in ['rgcn_seeds_p1.csv', 'rgcn_seeds_p2.csv', 'rgcn_nograph.csv']])
    r['model'] = 'R-GCN ' + r.run.map({'A': 'A', 'sider': 'B', 'offs': 'C', 'D': 'D', 'nograph': 'no graph'})
    x = pd.read_csv('xgb_results.csv'); x = x[(x.setting == setting) & (x.label == label)].copy(); x['model'] = 'XGBoost ' + x.run.str[0]
    b = pd.read_csv('seen_cold_baselines.csv'); b = b[(b.setting == setting) & (b.label == label) & (b.N == 3)].copy()
    b['model'] = b.baseline.map({'prior_PRR': 'PRR (prior counts)', 'event_freq': 'Event frequency', 'drug_volume': 'Drug volume', 'pair_prior': 'Pair count', 'event_n_drugs': 'Drugs per event'})
    d = pd.concat([r[['Y', 'model', 'test', 'AUPRC', 'base']], x[['Y', 'model', 'test', 'AUPRC', 'base']], b[['Y', 'model', 'test', 'AUPRC', 'base']]])
    return d.groupby(['model', 'test', 'Y']).agg(AUPRC=('AUPRC', 'mean'), base=('base', 'first')).reset_index()
F = fold_table(); M = F.groupby(['model', 'test']).AUPRC.mean().unstack(); BR = F.groupby('test').base.mean()
# ---- Figure A: AUPRC by model, two panels
order = ['PRR (prior counts)', 'Drug volume', 'Pair count', 'Event frequency', 'XGBoost D', 'R-GCN no graph', 'R-GCN D']
lab = {'XGBoost D': 'XGBoost (all sources)', 'R-GCN no graph': 'R-GCN without graph', 'R-GCN D': 'R-GCN (all sources)'}
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
for ax, ts, ttl in zip(axs, ('test_A', 'test_B'), ('Test set A', 'Test set B')):
    v = [M.loc[m, ts] for m in order]; y = np.arange(len(order))[::-1]
    ax.barh(y, v, color=BLUE, height=.55, zorder=3)
    for yi, vi in zip(y, v): ax.text(vi + .0008, yi, f'{vi:.3f}', va='center', fontsize=8.5, color=INK)
    ax.axvline(BR[ts], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=2)
    ax.set_yticks(y); ax.set_yticklabels([lab.get(m, m) for m in order]); ax.set_title(f'{ttl} (dashed line: base rate {BR[ts]:.4f})', fontsize=9.5, loc='left'); ax.set_xlim(0, max(v) * 1.22)
    ax.grid(axis='x', color='#e6e5e1', lw=.6, zorder=0); ax.tick_params(length=0); ax.set_xlabel('Mean AUPRC over seven folds')
plt.tight_layout(); plt.savefig(OUT + 'fig5_auprc_models.png', dpi=220); plt.close()
# ---- Figure B: fold-by-fold, test A
fig, ax = plt.subplots(figsize=(6.4, 3.4)); yrs = list(range(2019, 2026))
series = [('R-GCN D', 'R-GCN (all sources)', BLUE, '-'), ('XGBoost D', 'XGBoost (all sources)', ORANGE, '-'), ('Event frequency', 'Event frequency', AQUA, '-'), ('R-GCN no graph', 'R-GCN without graph', GRAY, (0, (4, 3)))]
for m, l, c, ls in series:
    v = F[(F.model == m) & (F.test == 'test_A')].sort_values('Y').AUPRC.values; ax.plot(yrs, v, color=c, lw=2, ls=ls, marker='o', ms=4.5, mec='white', mew=1, label=l, zorder=3)
ax.set_xticks(yrs); ax.set_xlabel('Test year'); ax.set_ylabel('AUPRC (test set A)'); ax.grid(axis='y', color='#e6e5e1', lw=.6); ax.tick_params(length=0)
ax.set_ylim(0, 0.066); ax.legend(frameon=False, fontsize=8.5, ncol=2, loc='upper right'); plt.tight_layout(); plt.savefig(OUT + 'fig5_auprc_by_fold.png', dpi=220); plt.close()
# ---- Figure C: ablation per fold dots + mean, both tests
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=False)
runs = [('R-GCN A', 'A\nFAERS'), ('R-GCN B', 'B\n+SIDER'), ('R-GCN C', 'C\n+OFFSIDES'), ('R-GCN D', 'D\nall')]
for ax, ts, ttl in zip(axs, ('test_A', 'test_B'), ('Test set A', 'Test set B')):
    for i, (m, l) in enumerate(runs):
        v = F[(F.model == m) & (F.test == ts)].AUPRC.values; ax.scatter(np.full(len(v), i) + np.linspace(-.12, .12, len(v)), v, s=14, color=BLUE, alpha=.55, zorder=3, edgecolor='white', linewidth=.5)
        ax.plot([i - .22, i + .22], [v.mean()] * 2, color=INK, lw=2, zorder=4); ax.text(i, v.mean() + (max(F[F.test == ts].AUPRC) * .0), '', fontsize=8)
        ax.text(i + .27, v.mean(), f'{v.mean():.3f}', fontsize=8.5, va='center')
    ax.set_xticks(range(4)); ax.set_xticklabels([l for _, l in runs], fontsize=8.5); ax.set_title(ttl, fontsize=10, loc='left'); ax.grid(axis='y', color='#e6e5e1', lw=.6); ax.tick_params(length=0); ax.set_xlim(-.5, 3.8)
axs[0].set_ylabel('AUPRC (dots: folds; bar: mean)'); plt.tight_layout(); plt.savefig(OUT + 'fig5_ablation.png', dpi=220); plt.close()
# ---- Figure D: rank percentile of positives
d = pd.read_csv('error_analysis_all.csv'); p = d[d.y_count].copy()
fig, ax = plt.subplots(figsize=(6.2, 3.2)); bins = np.linspace(0, 1, 21)
for col, l, c, off in (('rank_D', 'R-GCN (all sources)', BLUE, -.011), ('rank_event_freq', 'Event frequency', AQUA, .011)):
    h, _ = np.histogram(p[col] / p.n, bins=bins); ax.bar(bins[:-1] + .025 + off, h / len(p) * 100, width=.021, color=c, label=l, zorder=3)
ax.set_xlabel('Rank of the positive pair as a share of all candidates (0 = ranked first)'); ax.set_ylabel('Share of positives (%)'); ax.grid(axis='y', color='#e6e5e1', lw=.6, zorder=0); ax.tick_params(length=0)
ax.legend(frameon=False, fontsize=9); plt.tight_layout(); plt.savefig(OUT + 'fig5_rank_percentile.png', dpi=220); plt.close()
# ---- Figure E: polypharmacy (test A)
pr = pd.read_csv('poly_xgb.csv'); rp = pd.read_csv('rgcn_poly.csv')
def pm(df, mcol, key): g = df[(df.test == 'test_A') & (df[mcol] == key)].groupby(['label', 'Y']).AUPRC.mean().groupby('label').mean(); return g
rows = [('R-GCN (all sources)', rp, 'run', 'D'), ('XGBoost (all sources)', pr, 'model', 'XGB_D_all'), ('Event frequency', pr, 'model', 'event_freq')]
fig, ax = plt.subplots(figsize=(6.0, 3.2)); w = .34; x = np.arange(3)
vals = {k: [pm(df, mc, key)[k] for _, df, mc, key in rows] for k in ('y_lt5', 'y_poly5')}
ax.bar(x - w / 2 - .01, vals['y_lt5'], w, color=BLUE, label='Fewer than five drugs', zorder=3); ax.bar(x + w / 2 + .01, vals['y_poly5'], w, color=ORANGE, label='Five or more drugs', zorder=3)
for xi, (a, b) in enumerate(zip(vals['y_lt5'], vals['y_poly5'])): ax.text(xi - w / 2 - .01, a + .0004, f'{a:.4f}', ha='center', fontsize=8.5); ax.text(xi + w / 2 + .01, b + .0004, f'{b:.4f}', ha='center', fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows]); ax.set_ylabel('Mean AUPRC (test set A)'); ax.grid(axis='y', color='#e6e5e1', lw=.6, zorder=0); ax.tick_params(length=0); ax.legend(frameon=False, fontsize=9)
plt.tight_layout(); plt.savefig(OUT + 'fig5_polypharmacy.png', dpi=220); plt.close(); print('figures done'); print(M.round(4).to_string())
