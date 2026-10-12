import sys, torch, dgl
from torch.utils.data import DataLoader
sys.path.insert(0, '.')
from flowmol.dbtm.model import load_dbtm, probs_of, recenter
from flowmol.model_utils.load import read_config_file, data_module_from_config
from flowmol.data_processing.adaptive_sampler import AdaptiveEdgeSampler
from flowmol.data_processing.utils import get_batch_idxs, get_upper_edge_mask
from pathlib import Path
ck = sys.argv[1]
m = load_dbtm('flowmol/trained_models/flowmol3/checkpoints/last.ckpt', ck).eval()
cfg = read_config_file(Path('flowmol/trained_models/flowmol3/config.yaml'))
cfg['training']['trainer_args']['devices'] = 1; cfg['training']['num_workers'] = 0
dm = data_module_from_config(cfg); dm.setup('fit')
loader = DataLoader(dm.val_dataset, batch_sampler=AdaptiveEdgeSampler(dm.val_dataset, 30000), collate_fn=dgl.batch)
g = next(iter(loader)).to('cuda:0')
nb, eb = get_batch_idxs(g); up = get_upper_edge_mask(g); counts = g.batch_num_nodes().float().cuda()
x1, x0 = g.ndata['x_1_true'], g.ndata['x_0']
a1, c1, e1 = g.ndata['a_1_true'], g.ndata['c_1_true'], g.edata['e_1_true'][up]
print('data radius of gyration', x1.pow(2).sum(-1).mean().sqrt().item())
x0u = recenter(torch.randn_like(x0), nb, counts)
with torch.no_grad():
  for aligned in [True, False]:
    print('aligned x0' if aligned else 'UNALIGNED x0')
    xx0 = x0 if aligned else x0u
    for t in [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]:
        st = dict(a=(1-t)*torch.randn_like(a1)+t*a1, c=(1-t)*torch.randn_like(c1)+t*c1, e=(1-t)*torch.randn_like(e1)+t*e1,
                  x=recenter((1-t)*xx0+t*x1, nb, counts))
        o = probs_of(m.net(g, st['a'], st['c'], st['e'], st['x'], nb, up))
        print(f"t={t:.2f} x_rmse={((o['x']-x1).pow(2).sum(-1).mean().sqrt()):.2f}  pred_radius={o['x'].pow(2).sum(-1).mean().sqrt():.2f} "
              f"acc_a={(o['a'].argmax(-1)==a1.argmax(-1)).float().mean():.3f} acc_c={(o['c'].argmax(-1)==c1.argmax(-1)).float().mean():.3f} "
              f"acc_e={(o['e'].argmax(-1)==e1.argmax(-1)).float().mean():.3f} bondacc={((o['e'].argmax(-1)>0)&(e1.argmax(-1)>0)).sum().item()/max(1,(e1.argmax(-1)>0).sum().item()):.3f} "
              f"maxp_a={o['a'].max(-1).values.mean():.2f}")
