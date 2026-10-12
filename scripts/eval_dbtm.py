"""Evaluate a DBTM checkpoint over (K, kappa, commit-rule) grid; logs to wandb."""
import argparse, json, sys
from pathlib import Path
import torch, wandb
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import timed_sample, compute_metrics, summarize
from flowmol.dbtm.model import load_dbtm

p = argparse.ArgumentParser()
p.add_argument('--ckpt', required=True)
p.add_argument('--base_ckpt', default='flowmol/trained_models/flowmol3/checkpoints/last.ckpt')
p.add_argument('--K', type=int, nargs='+', default=[1, 2, 4, 8, 16])
p.add_argument('--kappa', type=float, nargs='+', default=[0.9])
p.add_argument('--commit', nargs='+', default=['confidence'])
p.add_argument('--mode', default='commit')
p.add_argument('--tau_pow', type=float, nargs='+', default=[1.0], help='taus = (r/K)**pow')
p.add_argument('--n_mols', type=int, default=1000)
p.add_argument('--n_subsets', type=int, default=1)
p.add_argument('--batch_size', type=int, default=500)
p.add_argument('--out', default='results/dbtm.jsonl')
p.add_argument('--run_name', required=True)
p.add_argument('--wandb_project', default='flowmol3-dbtm')
args = p.parse_args()
model = load_dbtm(args.base_ckpt, args.ckpt).eval()
run = wandb.init(project=args.wandb_project, name=args.run_name, job_type='eval', config=vars(args))
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
for commit in args.commit:
    for kappa in args.kappa:
      for tp in args.tau_pow:
        for K in args.K:
            taus = [(r / K) ** tp for r in range(1, K)]
            fn = lambda bs: model.sample_random_sizes(bs, K=K, kappa=kappa, commit=commit, device='cuda:0', mode=args.mode, taus=taus)
            mols, sec = timed_sample(fn, args.n_mols, args.batch_size)
            m = compute_metrics(mols, n_subsets=args.n_subsets)
            rec = dict(ckpt=args.ckpt, mode=args.mode, tau_pow=tp, K=K, kappa=kappa, commit=commit, ms_per_mol=sec / args.n_mols * 1e3, **m)
            print(f'[{args.run_name} pow={tp} commit={commit} kappa={kappa} K={K}] {rec["ms_per_mol"]:.1f} ms/mol {summarize(m)}', flush=True)
            open(args.out, 'a').write(json.dumps(rec) + '\n')
            wandb.log({f'{commit}/k{kappa:g}/{k}': v for k, v in rec.items() if isinstance(v, (int, float))} | {'K': K})
run.finish()
