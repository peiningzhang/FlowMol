"""Fast RDKit-only filter: sanitizable and single fragment. Writes <shard>.keep.pt with key 'valid'."""
import sys
from multiprocessing import Pool
from pathlib import Path
import torch
from rdkit import Chem, RDLogger


def ok(mb):
    RDLogger.DisableLog('rdApp.*')
    if mb is None:
        return False
    try:
        m = Chem.Mol(mb)
        Chem.SanitizeMol(m)
        return len(Chem.GetMolFrags(m)) == 1
    except Exception:
        return False


if __name__ == '__main__':
    for f in sys.argv[1:]:
        recs = torch.load(f)['recs']
        with Pool(16) as p:
            valid = p.map(ok, [r['mol'] for r in recs], chunksize=256)
        torch.save(dict(valid=valid), Path(f).with_suffix('.keep.pt'))
        print(f'{f}: n={len(recs)} valid={sum(valid)/len(recs):.3f}', flush=True)
