"""Sensitivity: evaluation without methotrexate, celecoxib, ofatumumab (seen setting, count label, N=3). Models are the SAME as the main runs
(trained on all drugs); only the evaluated candidate pairs change. R-GCN scores from rgcn_scores (seeds 1,4,5); XGBoost refit with the main settings. Output sensitivity_excl.csv"""
import pandas as pd, numpy as np, pickle, xgboost as xgb, warnings; warnings.filterwarnings('ignore')
from sklearn.metrics import average_precision_score, roc_auc_score
from harness import HERE
F = pickle.load(open(HERE + 'folds_seen.pkl', 'rb')); EX = {'methotrexate', 'celecoxib', 'ofatumumab'}; SEEDS = (1, 4, 5); rows = []
FE = ['pair_prior', 'drug_prior_reports', 'event_prior_reports', 'event_n_drugs', 'drug_n_events', 'prior_prr', 'share']
print('drug names containing excluded:', sorted({d for d in F[2019].drug.unique() if d in EX}))
for Y in range(2019, 2026):
    te = F[Y].reset_index(drop=True); tr = pd.concat([F[y] for y in range(2016, Y)])
    sc = {'event_freq': te.event_prior_reports.values.astype(float)}
    seedsc = {}   # per-seed scores; metrics are computed per seed and then averaged (as in the main results), not on an ensemble
    for run, name in (('A', 'rgcn_A'), ('D', 'rgcn_D'), ('nograph', 'rgcn_nograph')):
        seedsc[name] = [np.load(HERE + f'rgcn_scores/Y{Y}_{run}_s{s}.npy') for s in SEEDS]
    for run, cols in (('xgb_A', FE), ('xgb_D', FE + ['in_sider', 'in_offsides'])):
        m = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, n_jobs=2, random_state=0)
        m.fit(tr[cols].astype(float), tr.y_count.astype(int)); sc[run] = m.predict_proba(te[cols].astype(float))[:, 1]
    keep = ~te.drug.isin(EX).values
    for ts in ('test_A', 'test_B'):
        for subset, mk0 in (('all', np.ones(len(te), bool)), ('excl', keep)):
            mk = te[ts].values & mk0; y = te.y_count.values[mk].astype(int)
            if y.sum() == 0: continue
            allsc = dict(sc); allsc.update({k: v for k, v in seedsc.items()})
            for name, s in allsc.items():
                lst = s if isinstance(s, list) else [s]
                rows.append(dict(Y=Y, test=ts, subset=subset, model=name, pos=int(y.sum()), cand=int(mk.sum()), base=y.mean(),
                                 AUPRC=np.mean([average_precision_score(y, z[mk]) for z in lst]), AUROC=np.mean([roc_auc_score(y, z[mk]) for z in lst])))
    print(Y, 'done', flush=True)
r = pd.DataFrame(rows); r.to_csv('sensitivity_excl.csv', index=False); pd.set_option('display.width', 200)
m = r.groupby(['test', 'subset', 'model'])[['pos', 'cand', 'base', 'AUPRC', 'AUROC']].mean(); m['lift'] = m.AUPRC / m.base
print(m.round(4).to_string())
