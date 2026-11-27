import pandas as pd, sys, torch
torch.set_num_threads(4)
from rgcn import run_fold
seeds = [int(a) for a in sys.argv[1].split(',')]; tag = sys.argv[2]; rows = []
for seed in seeds:
    for Y in range(2019, 2026):
        for run in ('A', 'sider', 'offs', 'D'):
            rows += run_fold(Y, run, verbose=False, seed=seed)
            pd.DataFrame(rows).to_csv(f'rgcn_seeds_{tag}.csv', index=False); print(seed, Y, run, 'done', flush=True)
