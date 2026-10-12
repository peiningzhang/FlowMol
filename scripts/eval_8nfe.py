"""Matched-NFE comparison (target: 8 network evaluations). FlowMol3 baselines and DBTM checkpoints,
n_mols molecules split into n_subsets for CIs, PoseBusters + MMFF strain, logged to jsonl + wandb."""
import argparse, json, sys
from pathlib import Path
import torch
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import wandb
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import timed_sample, compute_metrics, summarize  # noqa: E402
from flowmol.models.flowmol import FlowMol  # noqa: E402
from flowmol.dbtm.model import load_dbtm  # noqa: E402

BASE = 'flowmol/trained_models/flowmol3/checkpoints/last.ckpt'
WARPS = {'lin': lambda u: u, 'sqrt': lambda u: u ** 0.5}
p = argparse.ArgumentParser()
p.add_argument('--baseline', nargs='*', default=[], help='steps:eta:warp, e.g. 7:30:lin')
p.add_argument('--dbtm', nargs='*', default=[], help='ckpt:K:tau_pow[:mode]')
p.add_argument('--n_mols', type=int, default=5000)
p.add_argument('--n_subsets', type=int, default=5)
p.add_argument('--batch_size', type=int, default=500)
p.add_argument('--out', default='results/eval_8nfe.jsonl')
p.add_argument('--run_name', required=True)
args = p.parse_args()

run = wandb.init(project='flowmol3-dbtm', name=args.run_name, job_type='eval', config=vars(args))
dev = 'cuda:0'


def record(tag, rec, sec, m):
    rec.update(tag=tag, ms_per_mol=sec / args.n_mols * 1e3, n_mols=args.n_mols, **m)
    print(f'[{tag}] {rec["ms_per_mol"]:.1f} ms/mol {summarize(m)} pb_ci={m.get("pb_valid_ci95", 0):.3f}', flush=True)
    with open(args.out, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    wandb.log({f'{tag}/{k}': v for k, v in rec.items() if isinstance(v, (int, float))})


if args.baseline:
    fm = FlowMol.load_from_checkpoint(BASE, map_location=dev).to(dev).eval()
    for spec in args.baseline:
        steps, eta, warp = spec.split(':'); steps, eta = int(steps), float(eta)
        ts = WARPS[warp](torch.linspace(0, 1, steps + 1, device=dev))
        fn = lambda bs: fm.sample_random_sizes(bs, device=dev, n_timesteps=steps + 1, stochasticity=eta,
                                               high_confidence_threshold=0.9, tspan=ts)
        mols, sec = timed_sample(fn, args.n_mols, args.batch_size)
        m = compute_metrics(mols, n_subsets=args.n_subsets, pb_workers=24)
        record(f'flowmol3_s{steps}_eta{eta:g}_{warp}', dict(method='flowmol3', steps=steps, nfe=steps + 1, eta=eta, warp=warp), sec, m)
    del fm

for spec in args.dbtm:
    parts = spec.split(':'); ckpt, K, pw = parts[0], int(parts[1]), float(parts[2])
    smode = parts[3] if len(parts) > 3 else 'cm'
    d = load_dbtm(BASE, ckpt, device=dev).eval()
    taus = [(r / K) ** pw for r in range(K)]
    fn = lambda bs: d.sample(d.sample_n_atoms(bs, dev), K=K, kappa=0.9, device=dev, mode=smode, taus=taus)
    mols, sec = timed_sample(fn, args.n_mols, args.batch_size)
    m = compute_metrics(mols, n_subsets=args.n_subsets, pb_workers=24)
    record(f'dbtm_{Path(ckpt).stem}_{smode}_K{K}_pow{pw:g}', dict(method='dbtm', ckpt=ckpt, K=K, nfe=K, tau_pow=pw, mode=smode), sec, m)
run.finish()
