"""Fine-tune pretrained FlowMol3 into a few-step DBTM map. Logs to wandb."""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import wandb
import dgl
from rdkit import RDLogger
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flowmol.dbtm.model import load_dbtm, DEFAULT_CFG  # noqa: E402
from flowmol.model_utils.load import read_config_file, data_module_from_config  # noqa: E402
from flowmol.data_processing.adaptive_sampler import AdaptiveEdgeSampler  # noqa: E402
from eval_utils import compute_metrics, summarize  # noqa: E402

RDLogger.DisableLog('rdApp.*')

p = argparse.ArgumentParser()
p.add_argument('--base_ckpt', default='flowmol/trained_models/flowmol3/checkpoints/last.ckpt')
p.add_argument('--base_config', default='flowmol/trained_models/flowmol3/config.yaml')
p.add_argument('--init', default=None, help='resume/initialise from a DBTM checkpoint')
p.add_argument('--out_dir', required=True)
p.add_argument('--run_name', required=True)
p.add_argument('--wandb_project', default='flowmol3-dbtm')
p.add_argument('--steps', type=int, default=20000)
p.add_argument('--lr', type=float, default=5e-5)
p.add_argument('--warmup', type=int, default=300)
p.add_argument('--clip', type=float, default=1.0)
p.add_argument('--max_num_edges', type=int, default=30000)
p.add_argument('--accum', type=int, default=1)
p.add_argument('--num_workers', type=int, default=6)
p.add_argument('--ema', type=float, default=0.999)
p.add_argument('--eval_every', type=int, default=2000)
p.add_argument('--eval_mols', type=int, default=300)
p.add_argument('--eval_mode', default='cm')
p.add_argument('--eval_tau_pow', type=float, default=0.5)
p.add_argument('--eval_tau_max', type=float, default=None, help='taus = tau_max * (r/(K-1))**pow instead of (r/K)**pow')
p.add_argument('--eval_K', type=int, nargs='+', default=[1, 2, 4])
p.add_argument('--save_every', type=int, default=1000)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--teacher_data', nargs='*', default=None, help='teacher pair shards (gen_teacher.py); replaces GEOM')
p.add_argument('--teacher_keep', default='pb', help='pb | valid | all')
for k, v in DEFAULT_CFG.items():
    if isinstance(v, (int, float)):
        p.add_argument(f'--{k}', type=type(v), default=v)
args = p.parse_args()
torch.manual_seed(args.seed)
dev = 'cuda:0'

cfg = {k: getattr(args, k) for k, v in DEFAULT_CFG.items() if isinstance(v, (int, float))}
model = load_dbtm(args.base_ckpt, args.init, cfg=cfg, device=dev)
model.cfg.update(cfg)   # command line wins over a loaded checkpoint
model.train()

config = read_config_file(Path(args.base_config))
config['training']['trainer_args']['devices'] = 1
config['training']['num_workers'] = args.num_workers
config['training']['max_num_edges'] = args.max_num_edges
if args.teacher_data:
    from flowmol.dbtm.teacher_data import TeacherDataset
    train_ds = TeacherDataset(args.teacher_data, keep=args.teacher_keep)
    print(f'teacher pairs: {len(train_ds)}', flush=True)
else:
    dm = data_module_from_config(config)
    dm.setup('fit')
    train_ds = dm.train_dataset
sampler = AdaptiveEdgeSampler(train_ds, args.max_num_edges)
loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=dgl.batch, num_workers=args.num_workers,
                    persistent_workers=args.num_workers > 0, prefetch_factor=4 if args.num_workers > 0 else None)

opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-12)
ema = ExponentialMovingAverage(model.parameters(), decay=args.ema)
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
run = wandb.init(project=args.wandb_project, name=args.run_name, config={**vars(args), 'dbtm_cfg': model.cfg},
                 dir=str(out_dir))


def save(tag):
    with ema.average_parameters():
        torch.save({'model': model.state_dict(), 'cfg': model.cfg}, out_dir / f'{tag}_ema.pt')
    torch.save({'model': model.state_dict(), 'cfg': model.cfg}, out_dir / f'{tag}.pt')


def eval_taus(K):
    if args.eval_tau_max is None:
        return [(r / K) ** args.eval_tau_pow for r in range(1, K)]
    return [args.eval_tau_max * (r / (K - 1)) ** args.eval_tau_pow for r in range(1, K)]


def evaluate(step):
    model.eval()
    res = {}
    with ema.average_parameters():
        for K in args.eval_K:
            torch.manual_seed(1234)
            t0 = time.time()
            mols = []
            while len(mols) < args.eval_mols:
                mols += model.sample_random_sizes(min(150, args.eval_mols - len(mols)), K=K, device=dev, mode=args.eval_mode,
                                                  taus=eval_taus(K))
            torch.cuda.synchronize()
            sec = time.time() - t0
            m = compute_metrics(mols, n_subsets=1, posebusters=True)
            print(f'[eval step {step} K={K}] {sec/len(mols)*1e3:.1f} ms/mol {summarize(m)}', flush=True)
            for k_, v_ in m.items():
                res[f'eval/K{K}/{k_}'] = v_
            res[f'eval/K{K}/ms_per_mol'] = sec / len(mols) * 1e3
    model.train()
    wandb.log(res, step=step)


it = iter(loader)
step, t0 = 0, time.time()
agg = {}
while step < args.steps:
    lr = args.lr * min(1.0, (step + 1) / args.warmup)
    for pg in opt.param_groups:
        pg['lr'] = lr
    opt.zero_grad(set_to_none=True)
    for _ in range(args.accum):
        try:
            g = next(it)
        except StopIteration:
            it = iter(loader)
            g = next(it)
        g = g.to(dev)
        loss, logs = model.training_losses(g)
        (loss / args.accum).backward()
        for k, v in logs.items():
            agg[k] = agg.get(k, 0.0) + float(v) / args.accum
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    if torch.isfinite(gn):
        opt.step()
        ema.update()
    else:
        print(f'step {step}: non-finite grad norm, skipped', flush=True)
    step += 1
    if step % 20 == 0:
        log = {f'train/{k}': v / 20 for k, v in agg.items()}
        log.update({'train/lr': lr, 'train/grad_norm': float(gn), 'train/it_per_s': step / (time.time() - t0),
                    'train/mem_gb': torch.cuda.max_memory_allocated() / 2**30})
        wandb.log(log, step=step)
        if step % 100 == 0:
            print(f'step {step} loss {log["train/total"]:.3f} tr_a {log["train/tr_a"]:.4f} anc_a {log["train/anc_a"]:.3f} '
                  f'bnd_a {log["train/bnd_a"]:.3f} bnd_x {log["train/bnd_x"]:.3f} {log["train/it_per_s"]:.2f} it/s '
                  f'mem {log["train/mem_gb"]:.1f}GB', flush=True)
        agg = {}
    if step % args.save_every == 0:
        save('last')
    if step % args.eval_every == 0 or step == args.steps:
        evaluate(step)
save('final')
run.finish()
