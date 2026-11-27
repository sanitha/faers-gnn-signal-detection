"""Baselines in two settings: SEEN (1 <= prior < N, realistic alert list) and COLD (prior == 0). N=3. Output seen_cold_baselines.csv"""
import pandas as pd, numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from harness import Harness
H = Harness(); rows = []
SC = {'pair_prior': 'pair_prior', 'prior_PRR': 'prior_prr', 'event_freq': 'event_prior_reports', 'event_n_drugs': 'event_n_drugs', 'drug_volume': 'drug_prior_reports'}
def pk(y, s, K):
    idx = np.argsort(-s, kind='stable')[:K]; return y[idx].sum() / K
for N in (3,):
    for Y in range(2019, 2026):
        f = H.fold(Y, N)
        for setting, sm in (('seen', f.pair_prior >= 1), ('cold', f.pair_prior == 0)):
            for lab in ('y_count', 'y_evans'):
                for ts in ('test_A', 'test_B'):
                    m = f[sm & f[ts]]; y = m[lab].values.astype(int)
                    if y.sum() == 0 or y.sum() == len(y): continue
                    for name, col in SC.items():
                        s = m[col].values.astype(float)
                        rows.append(dict(N=N, Y=Y, setting=setting, label=lab, test=ts, baseline=name, cand=len(m), pos=int(y.sum()), base=y.mean(),
                                         AUPRC=average_precision_score(y, s), AUROC=roc_auc_score(y, s), P10=pk(y, s, 10), P50=pk(y, s, 50), P100=pk(y, s, 100)))
        print(Y, 'ok', flush=True)
r = pd.DataFrame(rows); r.to_csv('seen_cold_baselines.csv', index=False)
pd.set_option('display.width', 200)
print(r.groupby(['setting', 'label', 'test', 'baseline'])[['cand', 'pos', 'base', 'AUPRC', 'AUROC', 'P50']].mean().round(4).to_string())
