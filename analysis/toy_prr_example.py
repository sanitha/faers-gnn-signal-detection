"""Teaching example: PRR/ROR by hand for one real drug-event pair, plus one 'emerging pair' walk-through.
Background = the 28,634 L01 reports only (the real baseline uses the whole FAERS file as background)."""
import pandas as pd, numpy as np
x = pd.read_pickle('l01_events.pkl')
rep = x.drop_duplicates('report_id')
T = rep.report_id.nunique()

def table(drug, event, d=x):
    t = d.drop_duplicates('report_id').report_id.nunique()
    dr = set(d[d.dn == drug].report_id); ev = set(d[d.event == event].report_id)
    a = len(dr & ev); b = len(dr) - a; c = len(ev) - a; dd = t - a - b - c
    prr = (a / (a + b)) / (c / (c + dd)); ror = (a * dd) / (b * c)
    return a, b, c, dd, prr, ror

print('Total L01 reports:', T)
for drug, event in [('docetaxel', 'neutropenia'), ('methotrexate', 'nausea'), ('sunitinib', 'fatigue')]:
    a, b, c, dd, prr, ror = table(drug, event)
    print(f'\n{drug} x {event}\n  a={a} (drug+event)  b={b} (drug, no event)  c={c} (other drugs+event)  d={dd} (neither)')
    print(f'  PRR = ({a}/{a+b}) / ({c}/{c+dd}) = {prr:.2f}    ROR = ({a}*{dd})/({b}*{c}) = {ror:.2f}')

# Emerging-pair walk-through: Y=2022, N=3
Y, N = 2022, 3
prior = x[x.year < Y].groupby(['dn', 'event']).size(); cur = x[x.year == Y].groupby(['dn', 'event']).size()
df = pd.DataFrame({'prior': prior, 'cur': cur}).fillna(0)
elig = df[df.prior < N]; pos = elig[elig.cur >= N]
print(f'\nY={Y}, N={N}: pairs seen so far={len(df)}, eligible (prior<{N})={len(elig)}, positives={len(pos)}')
print('Some positives (prior reports -> reports in', Y, '):')
print(pos.sort_values('cur', ascending=False).head(8).astype(int).to_string())
print('\nSample of eligible negatives:'); print(elig[elig.cur < N].sample(5, random_state=1).astype(int).to_string())
