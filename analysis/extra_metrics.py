import os as _o; ROOT = (_o.environ.get('PROJECT_ROOT') or _o.path.abspath(_o.path.join(_o.path.dirname(_o.path.abspath(globals().get('__file__', 'analysis/x.py'))), '..'))).rstrip('/') + '/'  # repo root; override with PROJECT_ROOT
"""Remaining design metrics for the primary setting (seen, y_count, N=3): MRR, recall@K (10, 50, 100) and bootstrap 95% CIs
(pairs resampled within fold, 500 reps, fold means), plus a paired bootstrap for R-GCN D minus XGB D.
Models: event_freq, prior_PRR, XGB A and D (retrained, same settings as run_xgb.py), R-GCN A and D (seeds 1,4,5, metrics averaged per seed).
Outputs extra_metrics.csv and extra_paired_boot.csv"""
import pandas as pd, numpy as np, pickle, xgboost as xgb, warnings; warnings.filterwarnings('ignore')
from sklearn.metrics import average_precision_score as ap
HERE = ROOT + 'analysis/'
F = pickle.load(open(HERE + 'folds_seen.pkl', 'rb')); SEEDS = (1, 4, 5); LAB = 'y_count'; B = 500
FE = ['pair_prior', 'drug_prior_reports', 'event_prior_reports', 'event_n_drugs', 'drug_n_events', 'prior_prr', 'share']
RUNS = {'A': FE, 'D': FE + ['in_sider', 'in_offsides']}
def ranks(s): r = np.empty(len(s)); r[np.argsort(-s, kind='stable')] = np.arange(1, len(s) + 1); return r
def point(y, s):
    r = ranks(s)[y == 1]; n = y.sum()
    return dict(AUPRC=ap(y, s), MRR=np.mean(1 / r), R10=(r <= 10).sum() / n, R50=(r <= 50).sum() / n, R100=(r <= 100).sum() / n)
scores = {}   # (Y, model) -> list of score vectors (one per seed)
for Y in range(2019, 2026):
    f = F[Y].reset_index(drop=True)
    scores[(Y, 'event_freq')] = [f.event_prior_reports.values.astype(float)]
    scores[(Y, 'prior_PRR')] = [f.prior_prr.values.astype(float)]
    tr = pd.concat([F[y] for y in range(2016, Y)])
    for run, cols in RUNS.items():
        m = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric='aucpr', n_jobs=8, random_state=0)
        m.fit(tr[cols].astype(float), tr[LAB].astype(int)); scores[(Y, 'xgb_' + run)] = [m.predict_proba(f[cols].astype(float))[:, 1]]
        scores[(Y, 'rgcn_' + run)] = [np.load(HERE + f'rgcn_scores/Y{Y}_{run}_s{sd}.npy') for sd in SEEDS]
    print('fold', Y, flush=True)
models = ['event_freq', 'prior_PRR', 'xgb_A', 'xgb_D', 'rgcn_A', 'rgcn_D']; rng = np.random.RandomState(0)
rows, boot = [], {}
for ts in ('test_A', 'test_B'):
    pt = {m: [] for m in models}; bt = {m: np.zeros((B, 7)) for m in models}
    for k, Y in enumerate(range(2019, 2026)):
        f = F[Y].reset_index(drop=True); mk = f[ts].values; y = f[LAB].values[mk].astype(int)
        S = {m: [s[mk] for s in scores[(Y, m)]] for m in models}
        for m in models: pt[m].append(pd.DataFrame([point(y, s) for s in S[m]]).mean())
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        for b in range(B):
            idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
            for m in models: bt[m][b, k] = np.mean([ap(y[idx], s[idx]) for s in S[m]])
    boot[ts] = bt
    for m in models:
        P = pd.DataFrame(pt[m]).mean(); lo, hi = np.percentile(bt[m].mean(1), [2.5, 97.5])
        rows.append(dict(test=ts, model=m, AUPRC=P.AUPRC, AUPRC_lo=lo, AUPRC_hi=hi, MRR=P.MRR, R10=P.R10, R50=P.R50, R100=P.R100))
R = pd.DataFrame(rows); R.to_csv(HERE + 'extra_metrics.csv', index=False)
pr = []
for ts, bt in boot.items():
    for a, b_ in (('rgcn_D', 'xgb_D'), ('rgcn_A', 'xgb_A'), ('rgcn_D', 'event_freq')):
        d = (bt[a] - bt[b_]).mean(1); pr.append(dict(test=ts, a=a, b=b_, mean_diff=d.mean(), lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5), share_a_better=(d > 0).mean()))
pd.DataFrame(pr).to_csv(HERE + 'extra_paired_boot.csv', index=False)
pd.set_option('display.width', 200); print(R.round(4).to_string()); print(pd.DataFrame(pr).round(4).to_string())
