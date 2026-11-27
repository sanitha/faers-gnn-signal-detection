#!/usr/bin/env bash
# Downloads the public datasets used by the analysis. Run from this folder: bash download_data.sh
# FAERS (Kaggle) needs a free Kaggle account and the Kaggle CLI (pip install kaggle, API token in ~/.kaggle/kaggle.json).
set -u
mkdir -p sider_4.1 offsides_twosides

echo "== OFFSIDES =="
curl -L -o offsides_twosides/OFFSIDES.csv.gz https://tatonettilab-resources.s3.us-west-1.amazonaws.com/nsides/OFFSIDES.csv.gz

echo "== SIDER 4.1 (check http://sideeffects.embl.de/download/ if these links have moved) =="
for f in drug_names.tsv drug_atc.tsv meddra_all_se.tsv.gz; do
  curl -L -o "sider_4.1/$f" "http://sideeffects.embl.de/media/download/$f" || echo "Could not fetch $f; download it from the SIDER site by hand."
done

echo "== FAERS sample (Kaggle) =="
if command -v kaggle >/dev/null 2>&1; then
  kaggle datasets download -d kanchana1990/fda-drug-adverse-event-reports-2015-to-2026-faers --unzip -p .
else
  echo "Kaggle CLI not found. Download manually from:"
  echo "https://www.kaggle.com/datasets/kanchana1990/fda-drug-adverse-event-reports-2015-to-2026-faers"
  echo "and place fda_adverse_events_2015_2026_CLEAN.csv in this folder."
fi
