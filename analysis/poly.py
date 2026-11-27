"""Polypharmacy labels: split year-Y L01 reports at num_drugs >= 5 vs < 5 and recompute the count label inside each subset.
Eligibility (prior < N) and all features stay as in the main protocol, so the SAME scores are evaluated on different labels."""
import pandas as pd, numpy as np
def poly_labels(x, f, Y, N=3):
    cur = x[x.year == Y]
    out = f[['drug', 'event']].copy()
    for name, sub in (('y_poly5', cur[cur.num_drugs >= 5]), ('y_lt5', cur[cur.num_drugs < 5])):
        c = sub.groupby(['dn', 'event']).size().rename(name)
        out = out.merge(c, left_on=['drug', 'event'], right_index=True, how='left')
        out[name] = out[name].fillna(0) >= N
    return out[['y_poly5', 'y_lt5']].set_index(f.index)
