"""XGBoost on harness features. For test year Y, train on folds Y' in [2016, Y-1] (labels from year Y' <= Y-1: no peeking).
Ablation of the tabular model: A = FAERS features only, B = +SIDER flag, C = +OFFSIDES flag, D = both. N=3, label y_count and y_evans.
Fixed modest hyperparameters (no tuning on test folds). Output xgb_results.csv"""
import pandas as pd, numpy as np, xgboost as xgb, warnings; warnings.filterwarnings('ignore')
from sklearn.metrics import average_precision_score, roc_auc_score
from harness import Harness
H = Harness(); N = 3
FE = ['pair_prior', 'drug_prior_reports', 'event_prior_reports', 'event_n_drugs', 'drug_n_events', 'prior_prr', 'share']
RUNS = {'A_faers': FE, 'B_+sider': FE + ['in_sider'], 'C_+offsides': FE + ['in_offsides'], 'D_all': FE + ['in_sider', 'in_offsides']}
folds = {}
for Y in range(2016, 2026):
    f = H.fold(Y, N); f['share'] = f.pair_prior / f.drug_prior_reports.clip(lower=1); folds[Y] = f; print('fold', Y, flush=True)
def pk(y, s, K): return y[np.argsort(-s, kind='stable')[:K]].sum() / K
rng = np.random.RandomState(0); rows = []
for setting in ('seen', 'cold'):
    for lab in ('y_count', 'y_evans'):
        for Y in range(2019, 2026):
            tr = pd.concat([folds[y] for y in range(2016, Y)])
            te = folds[Y]
            sel = (lambda d: d.pair_prior >= 1) if setting == 'seen' else (lambda d: d.pair_prior == 0)
            tr, te = tr[sel(tr)], te[sel(te)]
            if setting == 'cold':   # subsample negatives for speed, keep all positives
                neg = tr[~tr[lab]].sample(200000, random_state=0); tr = pd.concat([tr[tr[lab]], neg])
            for run, cols in RUNS.items():
                m = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                      eval_metric='aucpr', n_jobs=8, random_state=0)
                m.fit(tr[cols].astype(float), tr[lab].astype(int))
                s = m.predict_proba(te[cols].astype(float))[:, 1]
                for ts in ('test_A', 'test_B'):
                    mk = te[ts].values; y = te[lab].values[mk].astype(int)
                    if y.sum() == 0: continue
                    rows.append(dict(setting=setting, label=lab, Y=Y, run=run, test=ts, pos=int(y.sum()), base=y.mean(),
                                     AUPRC=average_precision_score(y, s[mk]), AUROC=roc_auc_score(y, s[mk]), P50=pk(y, s[mk], 50)))
        print(setting, lab, 'done', flush=True)
r = pd.DataFrame(rows); r.to_csv('xgb_results.csv', index=False)
pd.set_option('display.width', 200)
print(r.groupby(['setting', 'label', 'test', 'run'])[['pos', 'base', 'AUPRC', 'AUROC', 'P50']].mean().round(4).to_string())
