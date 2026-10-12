"""Evaluate pretrained FlowMol3 at reduced numbers of Euler steps (NFE sweep)."""
import argparse
import json
import sys
from pathlib import Path

import torch
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
import wandb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import timed_sample, compute_metrics, summarize  # noqa: E402
from flowmol.models.flowmol import FlowMol  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--ckpt', default='flowmol/trained_models/flowmol3/checkpoints/last.ckpt')
p.add_argument('--steps', type=int, nargs='+', default=[1, 2, 4, 8, 16, 50, 100, 250])
p.add_argument('--etas', type=float, nargs='+', default=[30.0])
p.add_argument('--hc_thresh', type=float, default=0.9)
p.add_argument('--n_mols', type=int, default=1000)
p.add_argument('--n_subsets', type=int, default=1)
p.add_argument('--batch_size', type=int, default=500)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--out', default='results/baseline_nfe.jsonl')
p.add_argument('--wandb_project', default='flowmol3-dbtm')
p.add_argument('--run_name', default='baseline_flowmol3_nfe')
args = p.parse_args()

model = FlowMol.load_from_checkpoint(args.ckpt, map_location='cuda:0').cuda().eval()
run = wandb.init(project=args.wandb_project, name=args.run_name, job_type='eval', config=vars(args))
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
table = wandb.Table(columns=['method', 'steps', 'nfe', 'eta', 'sec_per_mol', 'metrics_json'])
for eta in args.etas:
    for steps in args.steps:
        # FlowMol integrates on linspace(0,1,n_timesteps): steps = n_timesteps-1 Euler steps.
        # With self-conditioning, step 1 costs one extra forward pass.
        n_timesteps = steps + 1
        nfe = steps + 1
        fn = lambda bs: model.sample_random_sizes(bs, device='cuda:0', n_timesteps=n_timesteps,
                                                  stochasticity=eta, high_confidence_threshold=args.hc_thresh)
        mols, sec = timed_sample(fn, args.n_mols, args.batch_size, seed=args.seed)
        m = compute_metrics(mols, n_subsets=args.n_subsets)
        rec = dict(method='flowmol3', steps=steps, nfe=nfe, eta=eta, sec_per_mol=sec / args.n_mols,
                   n_mols=args.n_mols, **m)
        print(f'[eta={eta} steps={steps} nfe={nfe}] {sec/args.n_mols*1e3:.2f} ms/mol  {summarize(m)}', flush=True)
        with open(args.out, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        wandb.log({f'eta{eta:g}/{k}': v for k, v in rec.items() if isinstance(v, (int, float))} | {'nfe': nfe})
        table.add_data('flowmol3', steps, nfe, eta, sec / args.n_mols, json.dumps(m))
wandb.log({'results': table})
run.finish()
