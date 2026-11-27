import pandas as pd, torch
torch.set_num_threads(2)
from rgcn import run_fold
rows = []
for seed in (1, 4, 5):
    for Y in range(2019, 2026):
        rows += run_fold(Y, 'nograph', verbose=False, seed=seed)
        pd.DataFrame(rows).to_csv('rgcn_nograph.csv', index=False); print(seed, Y, 'done', flush=True)
