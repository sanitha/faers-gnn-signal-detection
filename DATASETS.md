# Dataset links

Nothing in `datasets/` is committed. Download each source from its original provider.
Check each provider's terms before reuse; they are not covered by this repository's MIT licence.

| Source | What is used | Where to get it | Notes |
|---|---|---|---|
| FAERS sample (Kaggle) | `fda_adverse_events_2015_2026_CLEAN.csv`, 528,000 reports, 48,000 per year for 2015 to 2025 | https://www.kaggle.com/datasets/kanchana1990/fda-drug-adverse-event-reports-2015-to-2026-faers | Derived from the US FDA Adverse Event Reporting System. One suspect drug per report, at most five reactions. Follow the dataset page's terms. |
| SIDER 4.1 | `drug_names.tsv`, `drug_atc.tsv`, `meddra_all_se.tsv.gz` | https://web.archive.org/web/20261004183122/https://sideeffects.embl.de/download/ | Kuhn et al. (2016), Nucleic Acids Research. Release of 21 October 2015, no per-entry dates. The original site (sideeffects.embl.de) was not reachable when this repository was prepared, so the link is an Internet Archive snapshot of its download page. The files fetched through the snapshot were checked byte for byte against the copies used in the analysis. The download page states a Creative Commons Attribution-Share Alike 4.0 licence. |
| OFFSIDES | `OFFSIDES.csv.gz` | https://tatonettilab.org/offsides/ (direct file: https://tatonettilab-resources.s3.us-west-1.amazonaws.com/nsides/OFFSIDES.csv.gz) | Tatonetti et al. (2012), Science Translational Medicine. Built from FAERS reports of earlier years. No snapshot date in the README. |
| TWOSIDES | not used in the reported results | https://tatonettilab.org/offsides/ | Available for future work on drug-drug interactions. |

References: Kuhn, M., Letunic, I., Jensen, L.J. and Bork, P. (2016) The SIDER database of drugs and side effects. Nucleic Acids Research, 44(D1). Tatonetti, N.P., Ye, P.P., Daneshjou, R. and Altman, R.B. (2012) Data-driven prediction of drug effects and interactions. Science Translational Medicine, 4(125).
