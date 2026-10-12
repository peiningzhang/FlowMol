"""Why is the MMFF relaxation RMSD higher than the teacher's?  Per-molecule distribution (tail vs shift), which internal
coordinates move during relaxation (bonds / angles / ring torsions / rotatable torsions) and dependence on flexibility.
Compares teacher molecules, FlowMol3 at few steps and DBTM checkpoints."""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, torch
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import timed_sample  # noqa: E402


def one(mb):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, rdMolAlign, rdMolTransforms as T
    RDLogger.DisableLog('rdApp.*')
    try:
        m = Chem.Mol(mb)
        Chem.SanitizeMol(m)
        if len(Chem.GetMolFrags(m)) != 1:
            return None
        props = AllChem.MMFFGetMoleculeProperties(m)
        if props is None:
            return None
        ff = AllChem.MMFFGetMoleculeForceField(m, props)
        m0 = Chem.Mol(m)
        e0 = ff.CalcEnergy()
        ff.Minimize(maxIts=2000)
        e1 = ff.CalcEnergy()
        heavy = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() > 1]
        c0, c1 = m0.GetConformer(), m.GetConformer()
        hv = lambda a, ex: [n.GetIdx() for n in a.GetNeighbors() if n.GetAtomicNum() > 1 and n.GetIdx() != ex]
        db, da, tor = [], [], dict(ring=[], rot=[], rigid=[])
        for b in m.GetBonds():
            j, k = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if m.GetAtomWithIdx(j).GetAtomicNum() == 1 or m.GetAtomWithIdx(k).GetAtomicNum() == 1:
                continue
            db.append(abs(T.GetBondLength(c1, j, k) - T.GetBondLength(c0, j, k)))
            nj, nk = hv(b.GetBeginAtom(), k), hv(b.GetEndAtom(), j)
            if not nj or not nk or nj[0] == nk[0]:
                continue
            d = T.GetDihedralDeg(c1, nj[0], j, k, nk[0]) - T.GetDihedralDeg(c0, nj[0], j, k, nk[0])
            d = abs((d + 180) % 360 - 180)
            kind = 'ring' if b.IsInRing() else ('rot' if b.GetBondType() == Chem.BondType.SINGLE and not b.GetIsConjugated() else 'rigid')
            tor[kind].append(d)
        for a in m.GetAtoms():
            nb = hv(a, -1)
            for x in range(len(nb)):
                for y in range(x + 1, len(nb)):
                    da.append(abs(T.GetAngleDeg(c1, nb[x], a.GetIdx(), nb[y]) - T.GetAngleDeg(c0, nb[x], a.GetIdx(), nb[y])))
        rmsd = rdMolAlign.AlignMol(m, m0, atomMap=[(i, i) for i in heavy])
        mean = lambda v: float(np.mean(v)) if v else float('nan')
        return dict(strain=e0 - e1, rmsd=rmsd, n_heavy=len(heavy), n_rot=len(tor['rot']), n_ringbond=len(tor['ring']),
                    d_bond=mean(db), d_angle=mean(da), d_ring=mean(tor['ring']), d_rot=mean(tor['rot']), d_rigid=mean(tor['rigid']),
                    max_rot=max(tor['rot']) if tor['rot'] else float('nan'))
    except Exception:
        return None


def report(tag, blocks, out):
    with Pool(8) as p:
        res = p.map(one, blocks, chunksize=16)
    ok = [r for r in res if r is not None]
    A = lambda k: np.array([r[k] for r in ok], dtype=float)
    q = lambda v, ps=(25, 50, 75, 90): ' '.join(f'{x:.2f}' for x in np.nanpercentile(v, ps))
    r, nrot, nh = A('rmsd'), A('n_rot'), A('n_heavy')
    print(f'== {tag}: ok={len(ok) / len(blocks):.2f}  n_heavy med={np.median(nh):.0f}  n_rot med={np.median(nrot):.0f} mean={nrot.mean():.2f}')
    print(f'   rmsd q25/50/75/90: {q(r)}   frac>1.0A={np.mean(r > 1.0):.2f} frac>1.5A={np.mean(r > 1.5):.2f}   strain q25/50/75/90: {q(A("strain"))}')
    print(f'   median |change|: bond={np.nanmedian(A("d_bond")):.3f}A angle={np.nanmedian(A("d_angle")):.2f}deg ring_tor={np.nanmedian(A("d_ring")):.1f}deg '
          f'rigid_tor={np.nanmedian(A("d_rigid")):.1f}deg rot_tor={np.nanmedian(A("d_rot")):.1f}deg  max_rot_tor med={np.nanmedian(A("max_rot")):.0f}deg')
    bins = [(0, 2), (3, 5), (6, 8), (9, 99)]
    print('   rmsd median by n_rot ' + ' '.join(f'[{a}-{b}]: {np.median(r[(nrot >= a) & (nrot <= b)]):.2f} (n={int(((nrot >= a) & (nrot <= b)).sum())})'
                                              for a, b in bins if ((nrot >= a) & (nrot <= b)).any()), flush=True)
    with open(out, 'a') as f:
        f.write(json.dumps(dict(tag=tag, n=len(blocks), recs=ok)) + '\n')


if __name__ == '__main__':
    N, dev, out = 1000, 'cuda:0', 'results/diag_rmsd.jsonl'
    BASE = 'flowmol/trained_models/flowmol3/checkpoints/last.ckpt'
    blk = lambda mols: [m.rdkit_mol.ToBinary() for m in mols if m.rdkit_mol is not None]
    report('teacher (FlowMol3 100 steps)', [r['mol'] for r in torch.load('data/teacher/shard3.pt')['recs'][:N]], out)

    from flowmol.dbtm.model import load_dbtm
    for name, ckpt in [('A', 'runs/dbtm_teacher_r2/final_ema.pt'), ('B', 'runs/dbtm_teacher_rilx03/final_ema.pt')]:
        d = load_dbtm(BASE, ckpt, device=dev).eval()
        for K in (8, 16):
            taus = [0.7 * r / (K - 1) for r in range(1, K)]
            mols, _ = timed_sample(lambda bs: d.sample(d.sample_n_atoms(bs, dev), K=K, device=dev, mode='cmx', taus=taus), N, 500)
            report(f'ours {name} K={K}', blk(mols), out)
        del d

    from flowmol.models.flowmol import FlowMol
    fm = FlowMol.load_from_checkpoint(BASE, map_location=dev).to(dev).eval()
    for steps in (23, 50):
        ts = torch.linspace(0, 1, steps + 1, device=dev) ** 0.5
        mols, _ = timed_sample(lambda bs: fm.sample_random_sizes(bs, device=dev, n_timesteps=steps + 1, stochasticity=30.0,
                                                                 high_confidence_threshold=0.9, tspan=ts), N, 500)
        report(f'FlowMol3 {steps} steps', blk(mols), out)
