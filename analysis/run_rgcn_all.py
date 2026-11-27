import pandas as pd, sys
from rgcn import run_fold
rows = []
for Y in range(2019, 2026):
    for run in ('A', 'sider', 'offs', 'D'):
        rows += run_fold(Y, run, verbose=False)
        pd.DataFrame(rows).to_csv('rgcn_results.csv', index=False); print(Y, run, 'done', flush=True)
