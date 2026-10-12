"""Pretrained FlowMol3 with warped time grids (training-free acceleration probe)."""
import json, sys
from pathlib import Path
import torch
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import timed_sample, compute_metrics, summarize
from flowmol.models.flowmol import FlowMol
model = FlowMol.load_from_checkpoint('flowmol/trained_models/flowmol3/checkpoints/last.ckpt', map_location='cuda:0').cuda().eval()
warps = {'lin': lambda u: u, 'early_dense(u^2)': lambda u: u**2, 'late_dense(1-(1-u)^2)': lambda u: 1-(1-u)**2,
         'late_dense3': lambda u: 1-(1-u)**3, 'sqrt': lambda u: u**0.5}
for name in sys.argv[1].split(','):
    for steps in [8, 16]:
        u = torch.linspace(0, 1, steps + 1, device='cuda:0')
        ts = warps[name](u)
        fn = lambda bs: model.sample_random_sizes(bs, device='cuda:0', n_timesteps=steps + 1, stochasticity=10.0,
                                                  high_confidence_threshold=0.9, tspan=ts)
        mols, sec = timed_sample(fn, 300, 150)
        m = compute_metrics(mols)
        rec = dict(warp=name, steps=steps, **m)
        print(f'[{name} steps={steps}] {summarize(m)}', flush=True)
        open('results/schedule_probe.jsonl', 'a').write(json.dumps(rec) + '\n')
