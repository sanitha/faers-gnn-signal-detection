"""Simple baselines on both labels and both test sets, all 7 folds. Output simple_baselines.csv"""
import pandas as pd, numpy as np, sys
from sklearn.metrics import average_precision_score, roc_auc_score
from harness import Harness
H = Harness(); rows = []
SCORES = {'event_freq': 'event_prior_reports', 'drug_volume': 'drug_prior_reports', 'prior_PRR': 'prior_prr', 'pair_prior': 'pair_prior',
          'event_n_drugs': 'event_n_drugs'}
for N in (3,):
    for Y in range(2019, 2026):
        f = H.fold(Y, N)
        for lab in ('y_count', 'y_evans'):
            for ts in ('test_A', 'test_B'):
                m = f[f[ts]]; y = m[lab].values.astype(int)
                if y.sum() == 0: continue
                for name, col in SCORES.items():
                    s = m[col].values.astype(float)
                    rows.append(dict(N=N, Y=Y, label=lab, test=ts, baseline=name, cand=len(m), pos=int(y.sum()), base=y.mean(),
                                     AUPRC=average_precision_score(y, s), AUROC=roc_auc_score(y, s)))
        print(Y, 'done', len(f), 'cands; pos count/evans A:', int(f[f.test_A].y_count.sum()), int(f[f.test_A].y_evans.sum()), flush=True)
r = pd.DataFrame(rows); r.to_csv('simple_baselines.csv', index=False)
print(r.groupby(['label', 'test', 'baseline'])[['pos', 'base', 'AUPRC', 'AUROC']].mean().round(4).to_string())
