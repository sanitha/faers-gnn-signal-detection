# Graph neural networks for early drug-adverse event signal detection

Leakage-aware temporal evaluation of a relational graph convolutional network (R-GCN) for ranking drug-adverse event pairs that later become frequently reported, using public FAERS, SIDER and OFFSIDES data for oncology drugs (ATC class L01). The code supports an academic research project. The study is retrospective, and a positive here is a reporting pattern, not a confirmed adverse reaction. Nothing in this repository shows an effect on patient safety.

## Question
Under a strict "past only" evaluation, does a graph model rank emerging drug-event pairs better than the proportional reporting ratio (PRR), the reporting odds ratio (ROR), simple frequency scores and XGBoost? Which data source adds most? Does the answer change for reports that list five or more drugs? See `DESIGN.md` for the protocol.

## Main results (seen-before pairs, count label, N = 3, mean over 7 test years 2019 to 2025)

| Model | AUPRC, test set A | AUPRC, test set B (strict) |
|---|---|---|
| PRR | 0.003 | 0.004 |
| Event frequency | 0.014 | 0.023 |
| XGBoost (all sources) | 0.018 | 0.042 |
| R-GCN without graph | 0.013 | 0.019 |
| R-GCN (all sources) | 0.029 | 0.047 |

The base rate is about 0.005. The graph model scored about 2.3 to 2.5 times its graph-free version in all seven years. Its edge over XGBoost was clear within folds in test set A (paired bootstrap difference +0.012, interval 0.005 to 0.021) and not established in test set B. SIDER and OFFSIDES added no clear gain, and no advantage specific to many-drug reports was found. With a second label, and for pairs with no earlier report, XGBoost was equal or better. Absolute scores are low. All numbers are in `results/`.

## Repository layout
```
analysis/   Python scripts (run from this folder)
results/    CSV outputs of the reported runs, for reference
figures/    figure scripts write here
datasets/   download script and instructions; data files are not committed
DESIGN.md   evaluation design
DATASETS.md data sources, licences and links
```

## How to rerun
1. Create an environment (Python 3.12) and install `pip install -r requirements.txt`. PyTorch Geometric may need the install steps for your platform.
2. Download the data: `cd datasets && bash download_data.sh`, then check the layout described in `datasets/README.md`.
3. Run the scripts from `analysis/`. Outputs are written to the current folder. Set `PROJECT_ROOT` only if you move the scripts away from the repository layout.

| Step | Script | Produces |
|---|---|---|
| Profile data (optional) | `profile_faers.py` | file profile |
| Match L01 drugs, build events and positive counts | `build_l01_events.py` | `l01_events.pkl`, `l01_drug_map.csv`, `positives_rolling_origin.csv` |
| Baselines | `run_simple_baselines.py`, `run_seen_baselines.py`, `run_ror_seen.py`, `baseline_prr_ror.py` | `simple_baselines.csv`, `seen_cold_baselines.csv`, `ror_seen.csv`, `baseline_prr_ror.csv` |
| XGBoost | `run_xgb.py`, `run_poly_xgb.py` | `xgb_results.csv`, `poly_xgb.csv` |
| R-GCN (four source runs, seeds) | `run_rgcn_seeds.py <seeds> <tag>`, `run_rgcn_all.py` | `rgcn_seeds_<tag>.csv`, saved scores in `rgcn_scores/` |
| R-GCN graph-free control | `run_nograph.py` | `rgcn_nograph.csv` |
| R-GCN other settings | `RG_SETTING=cold RG_LABEL=y_evans python run_rgcn_extra.py <seeds> <tag>` | `rgcn_<setting>_<label>_<tag>.csv` |
| Polypharmacy, sensitivity, error analysis | `rgcn_poly.py`, `sensitivity_excl.py`, `error_analysis.py` | `rgcn_poly.csv`, `sensitivity_excl.csv`, `error_analysis_all.csv` |
| Rank metrics and bootstrap intervals | `extra_metrics.py` | `extra_metrics.csv`, `extra_paired_boot.csv` |
| Statistics and tables | `stats_compare.py`, `final_stats.py` | `final_summary.csv`, `final_paired.csv` |
| Figures | `eda_figures.py`, `diagram_figures.py`, `make_ch5_figures.py`, `make_fig_ratio.py` | PNG files in `figures/` |

The R-GCN runs are the slowest step. Run times depend on the machine. The fold cache (`folds_seen.pkl`, `folds_cold.pkl`) is created on the first R-GCN run.

## Limitations
One suspect drug per report and at most five reactions per report in the FAERS sample. The outcome is report frequency, a proxy label. SIDER and OFFSIDES are older snapshots without per-entry dates. 73 drugs, including some not used only for cancer (methotrexate, celecoxib, ofatumumab). Hyperparameters were fixed and three seeds were used. Not carried out: node2vec baseline, model runs at other thresholds, retraining without the three non-oncology drugs, evaluation with degree-matched negatives. Bootstrap intervals reflect resampling within folds only.

## Licence
Code: MIT (see `LICENSE`). The datasets are not included and keep their own terms (see `DATASETS.md`).
