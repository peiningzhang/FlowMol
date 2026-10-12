import sys, torch, numpy as np
sys.path.insert(0, 'scripts')
from multiprocessing import Pool
from eval_utils import _mmff_one
recs = torch.load('data/teacher/shard2.pt')['recs'][:1500]
with Pool(8) as p:
    out = p.map(_mmff_one, [r['mol'] for r in recs], chunksize=32)
ok = [o for o in out if o is not None]
print('ok', len(ok) / len(out), 'strain median', np.median([o[0] - o[1] for o in ok]), 'rmsd median', np.median([o[2] for o in ok]))
