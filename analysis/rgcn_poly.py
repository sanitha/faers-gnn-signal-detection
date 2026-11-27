"""RQ3 for the R-GCN: score saved seen/y_count test-fold scores on the polypharmacy labels (no retraining).
Scores come from rgcn_scores/Y{Y}_{run}_s{seed}.npy (seeds 1,4,5; graph-free control = nograph). Output rgcn_poly.csv"""
import pandas as pd, numpy as np, pickle, os
from sklearn.metrics import average_precision_score, roc_auc_score
from harness import HERE
from poly import poly_labels
x = pd.read_pickle(HERE + 'l01_events.pkl'); FOLDS = pickle.load(open(HERE + 'folds_seen.pkl', 'rb'))
SEEDS = (1, 4, 5); RUNS = ('A', 'sider', 'offs', 'D', 'nograph'); rows = []
for Y in range(2019, 2026):
    te = FOLDS[Y].copy(); te = te.join(poly_labels(x, te, Y))
    for run in RUNS:
        for seed in SEEDS:
            s = np.load(HERE + f'rgcn_scores/Y{Y}_{run}_s{seed}.npy')
            assert len(s) == len(te), (Y, run, seed, len(s), len(te))
            for lab in ('y_count', 'y_poly5', 'y_lt5'):
                for ts in ('test_A', 'test_B'):
                    mk = te[ts].values; y = te[lab].values[mk].astype(int)
                    if y.sum() == 0: continue
                    rows.append(dict(Y=Y, run=run, seed=seed, label=lab, test=ts, pos=int(y.sum()), base=y.mean(),
                                     AUPRC=average_precision_score(y, s[mk]), AUROC=roc_auc_score(y, s[mk])))
r = pd.DataFrame(rows); r.to_csv('rgcn_poly.csv', index=False)
pd.set_option('display.width', 200)
m = r.groupby(['test', 'label', 'run', 'Y'])[['pos', 'base', 'AUPRC']].mean().reset_index().groupby(['test', 'label', 'run'])[['pos', 'base', 'AUPRC']].mean()
m['lift'] = m.AUPRC / m.base
print(m.round(4).to_string())
