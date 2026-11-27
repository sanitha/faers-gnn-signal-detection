"""ROR in the seen-before setting (count label, N=3), same fold tables as the other baselines. Output ror_seen.csv.
2x2 from train-only counts: a = pair_prior, a+b = drug_prior_reports, a+c = event_prior_reports (whole FAERS), n = FAERS reports before Y. Undefined -> 0."""
import pandas as pd, numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from harness import Harness
H = Harness(); rows = []
def pk(y, s, K): return y[np.argsort(-s, kind='stable')[:K]].sum() / K
for Y in range(2019, 2026):
    f = H.fold(Y, 3); f = f[f.pair_prior >= 1]; n = float(H.nrep_year[H.nrep_year.index < Y].sum())
    a = f.pair_prior.values.astype(float); Dd = f.drug_prior_reports.values.astype(float); E = f.event_prior_reports.values.astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        ror = a * (n - Dd - E + a) / ((Dd - a) * (E - a)); prr = (a / Dd) / ((E - a) / (n - Dd))
    ror = np.nan_to_num(ror, nan=0, posinf=0); prr = np.nan_to_num(prr, nan=0, posinf=0)
    for ts in ('test_A', 'test_B'):
        mk = f[ts].values; y = f.y_count.values[mk].astype(int)
        for name, s in (('ROR', ror), ('PRR_check', prr)):
            rows.append(dict(Y=Y, test=ts, model=name, pos=int(y.sum()), base=y.mean(), AUPRC=average_precision_score(y, s[mk]), AUROC=roc_auc_score(y, s[mk]), P50=pk(y, s[mk], 50)))
r = pd.DataFrame(rows); r.to_csv('ror_seen.csv', index=False)
print(r.groupby(['test', 'model'])[['base', 'AUPRC', 'AUROC', 'P50']].mean().round(4).to_string())
