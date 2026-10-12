"""Where does the strain come from?  (A) teacher-forced: predict from the true interpolant at level t and measure
MMFF strain of the predicted geometry;  (B) free-running cmx sampler: strain of the prediction at every round."""
import sys
from pathlib import Path
import dgl, torch, numpy as np
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import mmff_metrics  # noqa: E402
from flowmol.dbtm.model import load_dbtm, probs_of  # noqa: E402
from flowmol.dbtm.teacher_data import TeacherDataset  # noqa: E402
from flowmol.data_processing.utils import get_upper_edge_mask, get_batch_idxs  # noqa: E402

dev = 'cuda:0'
ckpt = sys.argv[1]
d = load_dbtm('flowmol/trained_models/flowmol3/checkpoints/last.ckpt', ckpt, device=dev).eval()
sig = d.cfg['sigma']
ds = TeacherDataset(['data/teacher/shard3.pt'], keep='valid')
torch.manual_seed(0)
g = dgl.batch([ds[i] for i in range(400)]).to(dev)
nb, eb = get_batch_idxs(g)
upper = get_upper_edge_mask(g)
counts = g.batch_num_nodes().float()
ctx = dict(x1=g.ndata['x_1_true'], x0=g.ndata['x_0'], a1=g.ndata['a_1_true'], c1=g.ndata['c_1_true'], e1=g.edata['e_1_true'][upper])
for k in 'ace':
    ctx['n' + k] = torch.randn_like(ctx[k + '1']) * sig


def rep(tag, mols, extra=''):
    m = mmff_metrics(mols, workers=8)
    print(f'{tag} ok={m["mmff_ok_frac"]:.2f} strain={m["mmff_strain_median"]:.1f} rmsd={m["mmff_relax_rmsd_median"]:.2f} {extra}', flush=True)


true = dict(x=ctx['x1'], a=ctx['a1'], c=ctx['c1'], e=ctx['e1'])
rep('[teacher itself]', d.build_mols(g, true, upper))
with torch.no_grad():
    for t in [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 1.0]:
        tn = torch.full((g.num_nodes(),), t, device=dev)
        te = torch.full((int(upper.sum()),), t, device=dev)
        P = probs_of(d._forward(g, d._state(ctx, tn, te, counts, nb), nb, upper))
        rmse = (P['x'] - ctx['x1']).pow(2).sum(-1).mean().sqrt().item()
        acc = (P['a'].argmax(-1) == ctx['a1'].argmax(-1)).float().mean().item()
        # true topology + predicted coordinates isolates the geometry error
        rep(f'[A teacher-forced t={t}]', d.build_mols(g, dict(true, x=P['x']), upper), f'x_rmse={rmse:.3f} acc_a={acc:.3f}')

for K in (8, 16):
    torch.manual_seed(0)
    taus = [(r / K) ** 0.5 for r in range(1, K)]
    tr = d.sample(d.sample_n_atoms(400, dev), K=K, device=dev, mode='cmx', taus=taus, trace=[])
    for r, mols in enumerate(tr):
        rep(f'[B free-running K={K} round {r + 1} input_tau={([0.0] + taus)[r]:.3f}]', mols)
