"""Distribution metrics (FCD etc., scripts/dist_metrics.py) at matched NFE.  Generated molecules are saved to
results/mols/<tag>.pt so further metrics do not need regeneration.
Specs: ref:val | ref:teacher | flowmol3:steps:eta:warp | dbtm:name:ckpt:K:tau_max"""
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np, torch
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import wandb
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import timed_sample  # noqa: E402
from dist_metrics import dist_metrics, load_reference, DIST_KEYS  # noqa: E402

BASE = 'flowmol/trained_models/flowmol3/checkpoints/last.ckpt'
p = argparse.ArgumentParser()
p.add_argument('--spec', nargs='+', required=True)
p.add_argument('--n_mols', type=int, default=5000)
p.add_argument('--batch_size', type=int, default=500)
p.add_argument('--out', default='results/eval_dist.jsonl')
p.add_argument('--run_name', required=True)
args = p.parse_args()
dev = 'cuda:0'
ref = load_reference()
run = wandb.init(project='flowmol3-dbtm', name=args.run_name, job_type='eval', config=vars(args))
Path('results/mols').mkdir(parents=True, exist_ok=True)
blk = lambda mols: [m.rdkit_mol.ToBinary() if m.rdkit_mol is not None else None for m in mols]

for spec in args.spec:
    s = spec.split(':')
    rec = dict(method=s[0], spec=spec, n_mols=args.n_mols)
    if s[0] == 'ref' and s[1] == 'val':
        raw = pickle.load(open('data/geom_raw/val_data.pickle', 'rb'))
        idx = np.random.RandomState(0).choice(len(raw), args.n_mols, replace=False)
        tag, blocks = 'ref_val_set', [raw[i][1][0].ToBinary() for i in idx]
        del raw
    elif s[0] == 'ref':
        tag, blocks = 'ref_teacher_100steps', [r['mol'] for r in torch.load('data/teacher/shard3.pt')['recs'][:args.n_mols]]
    elif s[0] == 'flowmol3':
        from flowmol.models.flowmol import FlowMol
        steps, eta, warp = int(s[1]), float(s[2]), s[3]
        fm = FlowMol.load_from_checkpoint(BASE, map_location=dev).to(dev).eval()
        ts = torch.linspace(0, 1, steps + 1, device=dev) ** (0.5 if warp == 'sqrt' else 1.0)
        mols, sec = timed_sample(lambda bs: fm.sample_random_sizes(bs, device=dev, n_timesteps=steps + 1, stochasticity=eta,
                                                                   high_confidence_threshold=0.9, tspan=ts), args.n_mols, args.batch_size)
        tag, blocks = f'flowmol3_s{steps}_eta{eta:g}_{warp}', blk(mols)
        rec.update(nfe=steps + 1, ms_per_mol=sec / args.n_mols * 1e3)
        del fm
    else:
        from flowmol.dbtm.model import load_dbtm
        name, ckpt, K, tmax = s[1], s[2], int(s[3]), float(s[4])
        d = load_dbtm(BASE, ckpt, device=dev).eval()
        taus = [tmax * r / (K - 1) for r in range(1, K)]
        mols, sec = timed_sample(lambda bs: d.sample(d.sample_n_atoms(bs, dev), K=K, device=dev, mode='cmx', taus=taus),
                                 args.n_mols, args.batch_size)
        tag, blocks = f'dbtm_{name}_K{K}_cap{tmax:g}', blk(mols)
        rec.update(nfe=K, ms_per_mol=sec / args.n_mols * 1e3)
        del d
    torch.cuda.empty_cache()
    torch.save(blocks, f'results/mols/{tag}.pt')
    m = dist_metrics(blocks, ref=ref)
    rec.update(tag=tag, **m)
    print(f'[{tag}] ' + ' '.join(f'{k}={m[k]:.4f}' for k in DIST_KEYS if k in m), flush=True)
    with open(args.out, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    wandb.log({f'{tag}/{k}': v for k, v in rec.items() if isinstance(v, (int, float))})
run.finish()
