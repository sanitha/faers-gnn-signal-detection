"""R-GCN link-scoring model (PyTorch Geometric), seen-before setting, N=3, label y_count.
For test year Y: train on folds Y' in [2016, Y-1] (graph from years < Y', labels from year Y'), evaluate on fold Y (graph from years < Y).
Graph: drug, event, report nodes; relations: drug<->report, event<->report, SIDER drug<->event, OFFSIDES drug<->event (per run A-D).
Decoder: MLP on [h_drug * h_event, log-count features]. Fixed hyperparameters, fixed epochs (no tuning on test folds)."""
import pandas as pd, numpy as np, torch, torch.nn as nn, os, pickle, time, sys
from torch_geometric.nn import RGCNConv
from sklearn.metrics import average_precision_score, roc_auc_score
from harness import Harness, HERE
torch.manual_seed(0); np.random.seed(0)
N = 3; DEV = torch.device('cpu')
SETTING = os.environ.get('RG_SETTING', 'seen'); LABEL = os.environ.get('RG_LABEL', 'y_count')   # seen|cold ; y_count|y_evans
assert SETTING in ('seen', 'cold') and LABEL in ('y_count', 'y_evans')
SUF = '' if (SETTING, LABEL) == ('seen', 'y_count') else f'_{SETTING}_{LABEL}'
NEG_CAP = 200000   # cold setting: training negatives subsampled (all positives kept), as in the XGBoost runs
H = Harness()
ev2id = {e: i for i, e in enumerate(H.fe.event.unique())}; NE = len(ev2id); ND = len(H.drugs); dix = H.dix
cache = HERE + f'folds_{SETTING}.pkl'
if os.path.exists(cache): FOLDS = pickle.load(open(cache, 'rb'))
else:
    FOLDS = {}
    for Y in range(2016, 2026):
        f = H.fold(Y, N); f = (f[f.pair_prior >= 1] if SETTING == 'seen' else f[f.pair_prior == 0]).copy(); f['share'] = f.pair_prior / f.drug_prior_reports.clip(lower=1); FOLDS[Y] = f
    pickle.dump(FOLDS, open(cache, 'wb'))
TAB = ['pair_prior', 'drug_prior_reports', 'event_prior_reports', 'event_n_drugs', 'drug_n_events', 'share']
# relation ids: 0 d->r, 1 r->d, 2 e->r, 3 r->e, 4 d->e(sider), 5 e->d(sider), 6 d->e(offs), 7 e->d(offs)
sider_e = [(dix[d], ev2id[e]) for d, e in H.sider if d in dix and e in ev2id]
offs_e = [(dix[d], ev2id[e]) for d, e in H.offs if d in dix and e in ev2id]

def build_graph(Yp, run):
    """Graph from reports with year < Yp. Node order: drugs | events | reports."""
    r = H.x[H.x.year < Yp]
    rid = {k: i for i, k in enumerate(r.report_id.unique())}; NR = len(rid)
    base_e, base_r = ND, ND + NE
    rr = r.drop_duplicates('report_id')
    d_src = torch.tensor([dix[d] for d in rr.dn]); d_rep = torch.tensor([rid[k] for k in rr.report_id]) + base_r
    ok = r.event.isin(ev2id)
    e_src = torch.tensor([ev2id[e] for e in r.event[ok]]) + base_e; e_rep = torch.tensor([rid[k] for k in r.report_id[ok]]) + base_r
    ei, et = [torch.stack([d_src, d_rep]), torch.stack([d_rep, d_src]), torch.stack([e_src, e_rep]), torch.stack([e_rep, e_src])], [0, 1, 2, 3]
    if 'sider' in run or run == 'D':
        s = torch.tensor(sider_e).T; ei += [torch.stack([s[0], s[1] + base_e]), torch.stack([s[1] + base_e, s[0]])]; et += [4, 5]
    if 'offs' in run or run == 'D':
        o = torch.tensor(offs_e).T; ei += [torch.stack([o[0], o[1] + base_e]), torch.stack([o[1] + base_e, o[0]])]; et += [6, 7]
    edge_index = torch.cat(ei, 1); edge_type = torch.cat([torch.full((e.shape[1],), t) for e, t in zip(ei, et)])
    # node features: log1p of volume (drug), prior event freq (event), num_drugs (report)
    fd = H.fold  # not used; features taken from the fold table below
    nd = torch.tensor(np.log1p(rr.num_drugs.clip(upper=200).values), dtype=torch.float)
    return dict(edge_index=edge_index, edge_type=edge_type, NR=NR, rep_feat=nd)

class Net(nn.Module):
    def __init__(self, d=64, nt=len(TAB), use_graph=True):
        super().__init__(); self.use_graph = use_graph
        self.ed, self.ee = nn.Embedding(ND, d), nn.Embedding(NE, d)
        self.fd, self.fe_, self.fr = nn.Linear(1, d), nn.Linear(1, d), nn.Linear(1, d)
        self.c1, self.c2 = RGCNConv(d, d, 8), RGCNConv(d, d, 8)
        self.drop = nn.Dropout(0.2)
        self.dec = nn.Sequential(nn.Linear(d + nt, 64), nn.ReLU(), nn.Linear(64, 1))
    def encode(self, g, dfeat, efeat):
        h = torch.cat([self.ed.weight + self.fd(dfeat), self.ee.weight + self.fe_(efeat), self.fr(g['rep_feat'][:, None])], 0)
        if not self.use_graph:   # graph-free control: same embeddings, features and decoder, no message passing
            return self.drop(h)
        h = self.drop(torch.relu(self.c1(h, g['edge_index'], g['edge_type'])))
        return self.c2(h, g['edge_index'], g['edge_type'])
    def score(self, h, di, ei, tab):
        return self.dec(torch.cat([h[di] * h[ND + ei], tab], 1)).squeeze(1)

def prep(Yp, run, train=False):
    g = build_graph(Yp, run); f = FOLDS[Yp]
    if train and SETTING == 'cold':
        neg = f[~f[LABEL]]; f = pd.concat([f[f[LABEL]], neg.sample(min(NEG_CAP, len(neg)), random_state=Yp)])
    dv = f.groupby('drug').drug_prior_reports.first().reindex(H.drugs).fillna(0).values
    ev = H.fe[H.fe.year < Yp].groupby('event').report_id.nunique()
    efeat = torch.zeros(NE, 1)
    for e, c in ev.items(): efeat[ev2id[e], 0] = float(np.log1p(c))
    dfeat = torch.tensor(np.log1p(dv), dtype=torch.float)[:, None]
    ok = f.event.isin(ev2id).values; f = f[ok]
    di = torch.tensor([dix[d] for d in f.drug]); ei = torch.tensor([ev2id[e] for e in f.event])
    tab = torch.tensor(np.log1p(f[TAB].values.astype(float)), dtype=torch.float)
    return dict(g=g, dfeat=dfeat, efeat=efeat, di=di, ei=ei, tab=tab, y=torch.tensor(f[LABEL].values.astype(float), dtype=torch.float), f=f)

def run_fold(Y, run, epochs=40, verbose=True, seed=0):
    torch.manual_seed(seed); np.random.seed(seed); t0 = time.time(); tr = [prep(y, run, train=True) for y in range(2016, Y)]; te = prep(Y, run)
    m = Net(use_graph=(run != 'nograph')); opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
    pw = torch.tensor(20.0); lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    for ep in range(epochs):
        m.train(); tot = 0
        for p in tr:
            opt.zero_grad(); h = m.encode(p['g'], p['dfeat'], p['efeat']); s = m.score(h, p['di'], p['ei'], p['tab'])
            l = lossf(s, p['y']); l.backward(); opt.step(); tot += l.item()
        if verbose and ep % 10 == 0: print(f'  ep {ep} loss {tot/len(tr):.4f} ({time.time()-t0:.0f}s)', flush=True)
    m.eval()
    with torch.no_grad(): s = m.score(m.encode(te['g'], te['dfeat'], te['efeat']), te['di'], te['ei'], te['tab']).numpy()
    os.makedirs(HERE + 'rgcn_scores', exist_ok=True); np.save(HERE + f'rgcn_scores/Y{Y}_{run}_s{seed}{SUF}.npy', s)
    f = te['f']; out = []
    for ts in ('test_A', 'test_B'):
        mk = f[ts].values; y = f[LABEL].values[mk].astype(int)
        if y.sum() == 0: continue
        sc = s[mk]; idx = np.argsort(-sc, kind='stable')[:50]
        out.append(dict(setting=SETTING, label=LABEL, Y=Y, run=run, seed=seed, test=ts, pos=int(y.sum()), base=y.mean(), AUPRC=average_precision_score(y, sc), AUROC=roc_auc_score(y, sc), P50=y[idx].sum() / 50))
    return out

if __name__ == '__main__':
    Y = int(sys.argv[1]) if len(sys.argv) > 1 else 2022
    for run in (sys.argv[2:] or ['A']):
        for o in run_fold(Y, run): print(o)
