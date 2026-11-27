import os as _o; ROOT = (_o.environ.get('PROJECT_ROOT') or _o.path.abspath(_o.path.join(_o.path.dirname(_o.path.abspath(globals().get('__file__', 'analysis/x.py'))), '..'))).rstrip('/') + '/'  # repo root; override with PROJECT_ROOT
import pandas as pd, sys
p=ROOT + 'datasets/fda_adverse_events_2015_2026_CLEAN.csv'
d=pd.read_csv(p,low_memory=False)
print(d.shape); print(d.year.value_counts().sort_index())
print(d.isna().mean().round(3).to_string())
print(d.suspect_drug.nunique(), d.suspect_drug.str.upper().nunique())
print(d.drug_count_category.value_counts())
print(d.num_drugs.describe())
print(d.suspect_drug.value_counts().head(15))
print(d.reactions.str.split('; ').str.len().describe())
