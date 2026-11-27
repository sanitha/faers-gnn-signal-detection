# Datasets (not stored in this repository)

The data are public but belong to their original providers, and the files are large (about 1 GB in total).
See `../DATASETS.md` for sources, licences and the exact files needed, then run `bash download_data.sh`
from this folder. Expected layout after download:

```
datasets/
  fda_adverse_events_2015_2026_CLEAN.csv      (Kaggle, manual or Kaggle CLI)
  sider_4.1/drug_names.tsv
  sider_4.1/drug_atc.tsv
  sider_4.1/meddra_all_se.tsv.gz
  offsides_twosides/OFFSIDES.csv.gz
```
