"""RQ3 for baselines + XGBoost (seen setting, N=3, test A and B). Output poly_xgb.csv"""
import pandas as pd, numpy as np, pickle, xgboost as xgb, warnings; warnings.filterwarnings('ignore')
from sklearn.metrics import average_precision_score, roc_auc_score
from harness import HERE
from poly import poly_labels
import pandas as pd
x = pd.read_pickle(HERE + 'l01_events.pkl'); FOLDS = pickle.load(open(HERE + 'folds_seen.pkl', 'rb'))
FE = ['pair_prior', 'drug_prior_reports', 'event_prior_reports', 'event_n_drugs', 'drug_n_events', 'prior_prr', 'share']
RUNS = {'XGB_A_faers': FE, 'XGB_D_all': FE + ['in_sider', 'in_offsides']}
rows = []
for Y in range(2019, 2026):
    tr = pd.concat([FOLDS[y] for y in range(2016, Y)]); te = FOLDS[Y].copy()
    te = te.join(poly_labels(x, te, Y))
    scores = {'event_freq': te.event_prior_reports.values.astype(float), 'prior_PRR': te.prior_prr.values.astype(float)}
    for run, cols in RUNS.items():
        m = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, n_jobs=2, random_state=0)
        m.fit(tr[cols].astype(float), tr.y_count.astype(int)); scores[run] = m.predict_proba(te[cols].astype(float))[:, 1]
    for lab in ('y_count', 'y_poly5', 'y_lt5'):
        for ts in ('test_A', 'test_B'):
            mk = te[ts].values; y = te[lab].values[mk].astype(int)
            if y.sum() == 0: continue
            for name, s in scores.items():
                rows.append(dict(Y=Y, label=lab, test=ts, model=name, pos=int(y.sum()), base=y.mean(),
                                 AUPRC=average_precision_score(y, s[mk]), AUROC=roc_auc_score(y, s[mk])))
    print(Y, 'done', flush=True)
r = pd.DataFrame(rows); r.to_csv('poly_xgb.csv', index=False); pd.set_option('display.width', 200)
r['lift'] = r.AUPRC / r.base
print(r.groupby(['test', 'label', 'model'])[['pos', 'base', 'AUPRC', 'AUROC', 'lift']].mean().round(4).to_string())
