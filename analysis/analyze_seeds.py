import pandas as pd, numpy as np, glob
pd.set_option("display.width", 200)
r = pd.concat([pd.read_csv(f) for f in glob.glob("rgcn_seeds_p*.csv")] + [pd.read_csv("rgcn_nograph.csv")])
r = r[r.seed.isin([1, 4, 5])]
print("seeds per run:", r.groupby("run").seed.nunique().to_dict())
fold = r.groupby(["test", "run", "Y"]).AUPRC.mean().reset_index()      # mean over seeds within fold
print("\nMean AUPRC over folds (seed-averaged); sd across seeds of the fold-mean in brackets")
sd = r.groupby(["test", "run", "seed"]).AUPRC.mean().groupby(["test", "run"]).std()
m = fold.groupby(["test", "run"]).AUPRC.mean()
print(pd.concat([m.rename("AUPRC"), sd.rename("sd_seed"), r.groupby(["test","run"]).AUROC.mean().rename("AUROC"), r.groupby(["test","run"]).P50.mean().rename("P50")], axis=1).round(4).to_string())
p = fold.pivot_table(index="Y", columns=["test", "run"], values="AUPRC"); print(p.round(3).to_string())
print("\nFolds where run beats nograph (of 7):")
for t in ("test_A", "test_B"):
    for run in ("A", "sider", "offs", "D"):
        print(t, run, int((p[(t, run)] > p[(t, "nograph")]).sum()))
