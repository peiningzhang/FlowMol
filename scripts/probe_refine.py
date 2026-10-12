"""Probe tau schedules / low-noise refinement rounds for the cm-family samplers.
Spec: name:mode:argmax_above:tau1,tau2,...  (K = number of taus + 1 network evaluations)."""
import argparse, json, sys
from pathlib import Path
import torch
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import wandb
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import timed_sample, compute_metrics, summarize  # noqa: E402
from flowmol.dbtm.model import load_dbtm  # noqa: E402

BASE = 'flowmol/trained_models/flowmol3/checkpoints/last.ckpt'
p = argparse.ArgumentParser()
p.add_argument('--ckpt', required=True)
p.add_argument('--spec', nargs='+', required=True)
p.add_argument('--n_mols', type=int, default=1000)
p.add_argument('--n_subsets', type=int, default=2)
p.add_argument('--batch_size', type=int, default=500)
p.add_argument('--out', default='results/probe_refine.jsonl')
p.add_argument('--run_name', required=True)
args = p.parse_args()

run = wandb.init(project='flowmol3-dbtm', name=args.run_name, job_type='eval', config=vars(args))
dev = 'cuda:0'
d = load_dbtm(BASE, args.ckpt, device=dev).eval()
for spec in args.spec:
    name, mode, amax, ts = spec.split(':')
    taus = [float(t) for t in ts.split(',')]
    K = len(taus) + 1
    fn = lambda bs: d.sample(d.sample_n_atoms(bs, dev), K=K, device=dev, mode=mode, taus=taus, argmax_above=float(amax))
    mols, sec = timed_sample(fn, args.n_mols, args.batch_size)
    m = compute_metrics(mols, n_subsets=args.n_subsets, pb_workers=24)
    rec = dict(tag=name, mode=mode, argmax_above=float(amax), taus=taus, K=K, ms_per_mol=sec / args.n_mols * 1e3, n_mols=args.n_mols, **m)
    print(f'[{name} K={K}] {rec["ms_per_mol"]:.1f} ms/mol {summarize(m)} pb_ci={m.get("pb_valid_ci95", 0):.3f}', flush=True)
    with open(args.out, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    wandb.log({f'{name}/{k}': v for k, v in rec.items() if isinstance(v, (int, float))})
run.finish()
