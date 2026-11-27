import os as _o; ROOT = (_o.environ.get('PROJECT_ROOT') or _o.path.abspath(_o.path.join(_o.path.dirname(_o.path.abspath(globals().get('__file__', 'analysis/x.py'))), '..'))).rstrip('/') + '/'  # repo root; override with PROJECT_ROOT
"""Summary figure: per-fold ratio of AUPRC, R-GCN (all sources, mean of 3 seeds) / XGBoost (all sources), for the four settings and both test sets. Log axis, reference line at 1."""
import pandas as pd, numpy as np, glob, os, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'font.family': 'Times New Roman', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#52514e', 'axes.linewidth': .6})
BLUE, ORANGE, INK = '#2a78d6', '#eb6834', '#0b0b0b'
OUT = ROOT + 'final_thesis/figures/fig5_ratio_vs_xgb.png'
def rg(setting, label):
    fs = ['rgcn_seeds_p1.csv', 'rgcn_seeds_p2.csv'] if (setting, label) == ('seen', 'y_count') else sorted(glob.glob(f'rgcn_{setting}_{label}_*.csv'))
    d = pd.concat([pd.read_csv(f) for f in fs]); d = d[d.run == 'D'].drop_duplicates(['Y', 'seed', 'test'], keep='last')
    return d.groupby(['test', 'Y']).AUPRC.mean()
def xg(setting, label):
    d = pd.read_csv('xgb_results.csv'); d = d[(d.setting == setting) & (d.label == label) & (d.run == 'D_all')]; return d.set_index(['test', 'Y']).AUPRC
cells = [('seen', 'y_count', 'Seen, count'), ('seen', 'y_evans', 'Seen, Evans'), ('cold', 'y_count', 'Cold, count'), ('cold', 'y_evans', 'Cold, Evans')]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
for ax, ts, ttl in zip(axs, ('test_A', 'test_B'), ('Test set A', 'Test set B')):
    for i, (s, l, name) in enumerate(cells):
        r = (rg(s, l)[ts] / xg(s, l)[ts]).dropna(); v = r.values
        ax.scatter(np.full(len(v), i) + np.linspace(-.15, .15, len(v)), v, s=14, color=BLUE, alpha=.6, edgecolor='white', linewidth=.4, zorder=3)
        mean_ratio = rg(s, l)[ts].mean() / xg(s, l)[ts].mean()
        ax.plot([i - .27, i + .27], [mean_ratio] * 2, color=INK, lw=2.2, zorder=4); ax.text(i, 0.027, f'{mean_ratio:.2f}', ha='center', fontsize=8.5)
    ax.axhline(1, color=ORANGE, lw=1.2, ls=(0, (4, 3)), zorder=2); ax.set_yscale('log'); ax.set_ylim(0.02, 40)
    ax.set_xticks(range(4)); ax.set_xticklabels([c[2].replace(', ', '\n') for c in cells], fontsize=8.5); ax.set_title(ttl, fontsize=10, loc='left')
    ax.grid(axis='y', color='#e6e5e1', lw=.6); ax.tick_params(length=0); ax.set_yticks([.1, .3, 1, 3, 10]); ax.set_yticklabels(['0.1', '0.3', '1', '3', '10'])
axs[0].set_ylabel('AUPRC ratio, R-GCN / XGBoost\n(above 1: R-GCN higher)')
plt.tight_layout(); plt.savefig(OUT, dpi=220); plt.close(); print('ok')
