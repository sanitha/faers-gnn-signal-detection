# Evaluation design

**Unit of analysis.** A drug-event pair. Drugs are the 73 ATC class L01 drugs found in the FAERS sample. Events are MedDRA preferred terms as written in the `reactions` field, lower-cased.

**Ground truth.** For test year Y and threshold N (primary N = 3):
- prior count = reports of the pair in years before Y; year-Y count = reports in year Y
- eligible candidate = prior count below N
- positive = eligible and year-Y count at least N; negative = eligible and year-Y count below N
- A second label (`y_evans`) uses the Evans criteria on year-Y data: at least 3 reports, PRR at least 2, chi-squared at least 4

The model flags pairs that become frequently reported in year Y, using only data from before Y. It does not predict label changes.

**Rolling origin.** Test years 2019 to 2025 (seven folds). All features and graph edges use years before the test year only. Training for test year Y uses folds 2016 to Y-1.

**Test sets.** A: positives absent from SIDER 4.1 (721 over the seven folds at N = 3). B (strict): absent from SIDER and OFFSIDES (186).

**Settings.** `seen`: pairs with at least one earlier report (realistic alert list). `cold`: pairs with no earlier report.

**Leakage rules.** SIDER (about 2015) and OFFSIDES (built from FAERS up to about 2012) are treated as known-before sources. Leakage control: compare scores on test set A and test set B for the runs with OFFSIDES edges.

**Ablation.** Run A: FAERS only. B: FAERS + SIDER. C: FAERS + OFFSIDES. D: all three.

**Models.** Baselines: PRR, ROR, event frequency, drug volume, XGBoost. Main model: R-GCN (PyTorch Geometric) for drug-event link scoring, with a graph-free control. Hyperparameters were fixed in advance, not tuned. R-GCN runs use three seeds.

**Metrics.** AUPRC (headline, with the base rate), AUROC, precision at 50, recall at K, mean reciprocal rank, and bootstrap intervals (primary setting only).

**Polypharmacy.** Year-Y reports are split at 5 or more drugs versus fewer, and the labels are recomputed in each subset. The FAERS file has one suspect drug per report, so polypharmacy is measured only by the number of drugs.
