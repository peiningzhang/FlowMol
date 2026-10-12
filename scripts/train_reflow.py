"""Reflow baseline: fine-tune pretrained FlowMol3 (its own loss and Euler/CTMC sampler) on teacher pairs,
with the teacher's x0 as the position prior instead of an OT-aligned independent sample. Logs to wandb."""
import argparse, sys, time
from pathlib import Path
import dgl
import torch
import wandb
from rdkit import RDLogger
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage
sys.path.insert(0, str(Path(__file__).resolve().parent))
from flowmol.models.flowmol import FlowMol  # noqa: E402
from flowmol.dbtm.teacher_data import TeacherDataset  # noqa: E402
from flowmol.data_processing.adaptive_sampler import AdaptiveEdgeSampler  # noqa: E402
from flowmol.data_processing.utils import get_upper_edge_mask, get_batch_idxs  # noqa: E402
from eval_utils import compute_metrics, summarize  # noqa: E402
RDLogger.DisableLog('rdApp.*')

p = argparse.ArgumentParser()
p.add_argument('--base_ckpt', default='flowmol/trained_models/flowmol3/checkpoints/last.ckpt')
p.add_argument('--teacher_data', nargs='+', required=True)
p.add_argument('--teacher_keep', default='pb')
p.add_argument('--out_dir', required=True)
p.add_argument('--run_name', required=True)
p.add_argument('--steps', type=int, default=20000)
p.add_argument('--lr', type=float, default=5e-5)
p.add_argument('--warmup', type=int, default=300)
p.add_argument('--clip', type=float, default=1.0)
p.add_argument('--max_num_edges', type=int, default=40000)
p.add_argument('--num_workers', type=int, default=6)
p.add_argument('--ema', type=float, default=0.999)
p.add_argument('--eval_every', type=int, default=2000)
p.add_argument('--eval_mols', type=int, default=300)
p.add_argument('--eval_steps', type=int, nargs='+', default=[4, 8])
p.add_argument('--eval_eta', type=float, default=30.0)
p.add_argument('--seed', type=int, default=0)
args = p.parse_args()
torch.manual_seed(args.seed)
dev = 'cuda:0'
W = {'a': 0.4, 'c': 1.0, 'e': 2.0, 'x': 3.0}

model = FlowMol.load_from_checkpoint(args.base_ckpt, map_location=dev).to(dev).train()
ds = TeacherDataset(args.teacher_data, keep=args.teacher_keep)
print(f'teacher pairs: {len(ds)}', flush=True)
loader = DataLoader(ds, batch_sampler=AdaptiveEdgeSampler(ds, args.max_num_edges), collate_fn=dgl.batch,
                    num_workers=args.num_workers, persistent_workers=True, prefetch_factor=4)
opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-12)
ema = ExponentialMovingAverage(model.parameters(), decay=args.ema)
out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
run = wandb.init(project='flowmol3-dbtm', name=args.run_name, config=vars(args), dir=str(out_dir))


def save(tag):
    with ema.average_parameters():
        torch.save(model.state_dict(), out_dir / f'{tag}_ema.pt')


def evaluate(step):
    model.eval(); res = {}
    with ema.average_parameters(), torch.no_grad():
        for s in args.eval_steps:
            torch.manual_seed(1234); t0 = time.time(); mols = []
            while len(mols) < args.eval_mols:
                mols += model.sample_random_sizes(min(150, args.eval_mols - len(mols)), device=dev, n_timesteps=s + 1,
                                                  stochasticity=args.eval_eta, high_confidence_threshold=0.9)
            sec = time.time() - t0
            m = compute_metrics(mols, posebusters=True)
            print(f'[eval step {step} steps={s}] {sec/len(mols)*1e3:.1f} ms/mol {summarize(m)}', flush=True)
            res.update({f'eval/S{s}/{k}': v for k, v in m.items()})
    model.train(); wandb.log(res, step=step)


evaluate(0)
it, step, t0, agg = iter(loader), 0, time.time(), {}
while step < args.steps:
    for pg in opt.param_groups:
        pg['lr'] = args.lr * min(1.0, (step + 1) / args.warmup)
    try:
        g = next(it)
    except StopIteration:
        it = iter(loader); g = next(it)
    g = g.to(dev)
    upper = get_upper_edge_mask(g); nb, _ = get_batch_idxs(g)
    x0 = g.ndata['x_0'].clone()
    g = model.sample_prior(g, nb, upper)      # masked categorical priors
    g.ndata['x_0'] = x0                       # coupled teacher position prior
    losses = model(g)
    loss = sum(W[k] * v for k, v in losses.items())
    opt.zero_grad(set_to_none=True); loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    if torch.isfinite(gn):
        opt.step(); ema.update()
    step += 1
    for k, v in losses.items():
        agg[k] = agg.get(k, 0.0) + float(v)
    if step % 100 == 0:
        log = {f'train/{k}': v / 100 for k, v in agg.items()} | {'train/it_per_s': step / (time.time() - t0)}
        wandb.log(log, step=step); agg = {}
        print(f'step {step} ' + ' '.join(f'{k}={v:.4f}' for k, v in log.items()), flush=True)
    if step % 1000 == 0:
        save('last')
    if step % args.eval_every == 0:
        evaluate(step)
save('final')
run.finish()
