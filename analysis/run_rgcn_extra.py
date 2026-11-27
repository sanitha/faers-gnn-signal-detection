"""usage: RG_SETTING=cold RG_LABEL=y_evans python run_rgcn_extra.py <seeds> <tag> ; writes rgcn_<setting>_<label>_<tag>.csv"""
import pandas as pd, sys, torch
torch.set_num_threads(4)
import rgcn
from rgcn import run_fold, SETTING, LABEL
seeds = [int(a) for a in sys.argv[1].split(',')]; tag = sys.argv[2]; rows = []
RUNS = sys.argv[3].split(',') if len(sys.argv) > 3 else ['A', 'sider', 'offs', 'D']
for seed in seeds:
    for Y in range(2019, 2026):
        for run in RUNS:
            rows += run_fold(Y, run, verbose=False, seed=seed)
            pd.DataFrame(rows).to_csv(f'rgcn_{SETTING}_{LABEL}_{tag}.csv', index=False); print(seed, Y, run, 'done', flush=True)
