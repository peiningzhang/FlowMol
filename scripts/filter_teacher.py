"""Mark which teacher molecules are RDKit-valid / PoseBusters-valid. Writes <shard>.keep.pt with boolean lists."""
import sys
from pathlib import Path
import posebusters as pb
import torch
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

for f in sys.argv[1:]:
    recs = torch.load(f)['recs']
    mols, valid = [], []
    for r in recs:
        m = Chem.Mol(r['mol']) if r['mol'] is not None else None
        ok = False
        if m is not None:
            try:
                mm = Chem.Mol(m); Chem.SanitizeMol(mm)
                ok = len(Chem.GetMolFrags(mm)) == 1
            except Exception:
                pass
        valid.append(ok)
        mols.append(m if m is not None else Chem.MolFromSmiles('C'))
    idx = [i for i, v in enumerate(valid) if v]
    df = pb.PoseBusters(config='mol', max_workers=32).bust([mols[i] for i in idx], None, None)
    passed = df.values.astype(bool).all(axis=1)
    keep_pb = [False] * len(recs)
    for i, ok in zip(idx, passed):
        keep_pb[i] = bool(ok)
    torch.save(dict(valid=valid, pb=keep_pb), Path(f).with_suffix('.keep.pt'))
    print(f'{f}: n={len(recs)} valid={sum(valid)/len(recs):.3f} pb={sum(keep_pb)/len(recs):.3f}', flush=True)
