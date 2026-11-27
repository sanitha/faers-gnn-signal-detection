import os as _o; ROOT = (_o.environ.get('PROJECT_ROOT') or _o.path.abspath(_o.path.join(_o.path.dirname(_o.path.abspath(globals().get('__file__', 'analysis/x.py'))), '..'))).rstrip('/') + '/'  # repo root; override with PROJECT_ROOT
"""Rolling-origin fold harness (DESIGN.md).
fold(Y, N) -> one row per eligible candidate pair (drug, event) with TRAIN-ONLY features (years < Y)
and two labels computed from year-Y reports:
  y_count : prior < N and year-Y count >= N                       (design lock, label 1)
  y_evans : prior < N and Evans criteria on year-Y data           (label 2: a>=3, PRR>=2, chi2>=4)
Test sets: A = not a SIDER pair; B = not a SIDER and not an OFFSIDES pair.
Candidate events = events seen in any FAERS report before Y (positives with unseen events are counted as unreachable)."""
import pandas as pd, numpy as np, os, pickle
R = ROOT; D = R + 'datasets/'; HERE = R + 'analysis/'
_cache = HERE + 'known_pairs.pkl'

def _known_pairs():
    if os.path.exists(_cache):
        return pickle.load(open(_cache, 'rb'))
    ns = {'__file__': HERE + 'build_l01_events.py'}
    exec(open(HERE + 'build_l01_events.py').read().split('d=pd.read_csv')[0], ns)
    nm, SYN, key = ns['nm'], ns['SYN'], ns['key']
    se = pd.read_csv(D + 'sider_4.1/meddra_all_se.tsv.gz', sep='\t', header=None,
                     names=['c1', 'c2', 'u1', 'mtype', 'u2', 'event'], compression='gzip')
    se = se[se.mtype == 'PT'].merge(nm.rename(columns={'cid': 'c1'}), on='c1')
    se['dn'] = se.name.str.lower().str.strip().map(lambda n: SYN.get(n, n)).map(key)
    se['event'] = se.event.str.lower().str.strip()
    sider = set(zip(se.dn, se.event))
    o = pd.read_csv(D + 'offsides_twosides/OFFSIDES.csv.gz', compression='gzip',
                    usecols=['drug_concept_name', 'condition_concept_name'])
    o['dn'] = (o.drug_concept_name.str.lower()
               .str.replace(r'\b(hydrochloride|hcl|sodium|sulfate|mesylate|anhydrous|disodium|phosphate|citrate|acetate|maleate)\b', '', regex=True)
               .str.replace(r'\s+', ' ', regex=True).str.strip())
    offs = set(zip(o.dn, o.condition_concept_name.str.lower().str.strip()))
    pickle.dump((sider, offs), open(_cache, 'wb'))
    return sider, offs

class Harness:
    def __init__(self):
        self.x = pd.read_pickle(HERE + 'l01_events.pkl')
        full = pd.read_csv(D + 'fda_adverse_events_2015_2026_CLEAN.csv', usecols=['report_id', 'year', 'reactions'], low_memory=False)
        fe = full.assign(event=full.reactions.str.split('; ')).explode('event')
        fe['event'] = fe.event.str.lower().str.strip()
        self.fe = fe[fe.event.notna() & fe.event.ne('')][['report_id', 'year', 'event']]
        self.nrep_year = full.groupby('year').size()
        self.sider, self.offs = _known_pairs()
        self.drugs = sorted(self.x.dn.unique()); self.dix = {d: i for i, d in enumerate(self.drugs)}

    def fold(self, Y, N):
        x, fe, drugs, dix = self.x, self.fe, self.drugs, self.dix
        prior, cur = x[x.year < Y], x[x.year == Y]
        events = pd.Index(fe[fe.year < Y].event.unique()); eix = {e: i for i, e in enumerate(events)}
        nprior = int(self.nrep_year[self.nrep_year.index < Y].sum()); ncur = int(self.nrep_year[Y])
        Eb = fe[fe.year < Y].groupby('event').report_id.nunique().reindex(events).values.astype(float)
        Da = prior.groupby('dn').report_id.nunique().reindex(drugs).fillna(0).values.astype(float)
        def mat(df):
            M = np.zeros((len(drugs), len(events)))
            g = df[df.event.isin(eix)].groupby(['dn', 'event']).size()
            for (d_, e_), c in g.items(): M[dix[d_], eix[e_]] = c
            return M
        A, C = mat(prior), mat(cur)
        # year-Y 2x2 table for Evans label (background = all FAERS reports of year Y)
        Ey = fe[fe.year == Y].groupby('event').report_id.nunique().reindex(events).fillna(0).values.astype(float)[None, :]
        Dy = cur.groupby('dn').report_id.nunique().reindex(drugs).fillna(0).values.astype(float)[:, None]
        a = C; b = Dy - a; c = Ey - a; d = ncur - a - b - c
        with np.errstate(divide='ignore', invalid='ignore'):
            prr_y = (a / Dy) / (c / (ncur - Dy))
            chi2 = ncur * (a * d - b * c) ** 2 / ((a + b) * (c + d) * (a + c) * (b + d))
        evans = (a >= 3) & (np.nan_to_num(prr_y) >= 2) & (np.nan_to_num(chi2) >= 4)
        # train-only prior PRR
        Dd, E = Da[:, None], Eb[None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            prr_p = np.nan_to_num((A / Dd) / ((E - A) / (nprior - Dd)), nan=0, posinf=0)
        s_mask = np.zeros_like(A, dtype=bool); o_mask = np.zeros_like(A, dtype=bool)
        for (d_, e_) in self.sider:
            if d_ in dix and e_ in eix: s_mask[dix[d_], eix[e_]] = True
        for (d_, e_) in self.offs:
            if d_ in dix and e_ in eix: o_mask[dix[d_], eix[e_]] = True
        elig = A < N
        ii, jj = np.nonzero(elig)
        df = pd.DataFrame({
            'drug': np.array(drugs)[ii], 'event': events.values[jj],
            'pair_prior': A[ii, jj], 'drug_prior_reports': Da[ii], 'event_prior_reports': Eb[jj],
            'event_n_drugs': (A > 0).sum(0)[jj], 'drug_n_events': (A > 0).sum(1)[ii],
            'prior_prr': prr_p[ii, jj], 'in_sider': s_mask[ii, jj], 'in_offsides': o_mask[ii, jj],
            'cur_count': C[ii, jj], 'y_count': C[ii, jj] >= N, 'y_evans': evans[ii, jj]})
        df['test_A'] = ~df.in_sider; df['test_B'] = ~df.in_sider & ~df.in_offsides
        df.attrs.update(Y=Y, N=N)
        return df
