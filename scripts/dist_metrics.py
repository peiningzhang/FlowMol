"""Distribution metrics against the GEOM-Drugs test set, following VEDA-shortcut (util/metrics.py):
FCD (vs test SMILES), uniqueness, novelty (vs train SMILES) and Wasserstein distances of type-specific bond lengths /
bond angles, torsion circular-mean distance and PCA shape descriptors (vs test conformers).
Only valid (sanitizable, single-fragment) generated molecules enter, as in VEDA.  The reference is cached in data/dist_ref."""
import pickle
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
REF_DIR = ROOT / 'data' / 'dist_ref'
N_GEOM_REF = 5000


def _smiles(mb):
    """Canonical non-isomeric heavy-atom SMILES of a valid single-fragment molecule, else None."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    try:
        m = Chem.Mol(mb) if isinstance(mb, bytes) else Chem.MolFromSmiles(mb)
        if m is None:
            return None
        Chem.SanitizeMol(m)
        if len(Chem.GetMolFrags(m)) != 1:
            return None
        return Chem.MolToSmiles(Chem.RemoveHs(m), canonical=True, isomericSmiles=False)
    except Exception:
        return None


def _geom(mb):
    """Type-keyed bond lengths, angles, torsions and shape descriptors of one molecule (VEDA definitions)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolTransforms as T
    RDLogger.DisableLog('rdApp.*')
    try:
        m = Chem.Mol(mb)
        Chem.SanitizeMol(m)
        if len(Chem.GetMolFrags(m)) != 1 or m.GetNumConformers() == 0:
            return None
        conf = m.GetConformer()
        sym = [a.GetSymbol() for a in m.GetAtoms()]
        B, A, To = defaultdict(list), defaultdict(list), defaultdict(list)
        for b in m.GetBonds():
            j, k = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            s = sorted([sym[j], sym[k]])
            B[f'{s[0]}-{s[1]}_{b.GetBondType()}'].append(T.GetBondLength(conf, j, k))
            for i in (n.GetIdx() for n in m.GetAtomWithIdx(j).GetNeighbors() if n.GetIdx() != k):
                for l in (n.GetIdx() for n in m.GetAtomWithIdx(k).GetNeighbors() if n.GetIdx() != j):
                    if i == l:
                        continue
                    To[min(f'{sym[i]}-{sym[j]}-{sym[k]}-{sym[l]}', f'{sym[l]}-{sym[k]}-{sym[j]}-{sym[i]}')].append(
                        T.GetDihedralDeg(conf, i, j, k, l))
        for a in m.GetAtoms():
            nb = [n.GetIdx() for n in a.GetNeighbors()]
            for x in range(len(nb)):
                for y in range(x + 1, len(nb)):
                    s = sorted([sym[nb[x]], sym[nb[y]]])
                    A[f'{s[0]}-{sym[a.GetIdx()]}-{s[1]}'].append(T.GetAngleDeg(conf, nb[x], a.GetIdx(), nb[y]))
        pos = conf.GetPositions()
        ev = np.sort(np.linalg.eigvalsh(np.cov((pos - pos.mean(0)).T)))[::-1]
        asp, acy = ev[0] - 0.5 * (ev[1] + ev[2]), ev[1] - ev[2]
        return dict(B), dict(A), dict(To), (asp, acy, (asp ** 2 + 0.75 * acy ** 2) / ev.sum() ** 2)
    except Exception:
        return None


def _merge(res):
    B, A, To, S = defaultdict(list), defaultdict(list), defaultdict(list), []
    for r in res:
        if r is None:
            continue
        for dst, src in ((B, r[0]), (A, r[1]), (To, r[2])):
            for k, v in src.items():
                dst[k].extend(v)
        S.append(r[3])
    f = lambda d: {k: np.asarray(v, dtype=np.float32) for k, v in d.items()}
    return dict(bond=f(B), angle=f(A), torsion=f(To), shape=np.asarray(S, dtype=np.float32))


def _fcd():
    from fcd_torch import FCD
    return FCD(device='cuda:0' if torch.cuda.is_available() else 'cpu', n_jobs=4)


def load_reference(workers=8):
    f = REF_DIR / 'geom_test_ref.pt'
    if f.exists():
        return torch.load(f)
    REF_DIR.mkdir(parents=True, exist_ok=True)
    with Pool(workers) as p:
        train = set(p.map(_smiles, pickle.load(open(ROOT / 'data/geom_full_kekulized/train_data_smiles.pkl', 'rb')), chunksize=2000))
        test = [s for s in p.map(_smiles, pickle.load(open(ROOT / 'data/geom_full_kekulized/test_data_smiles.pkl', 'rb')), chunksize=2000) if s]
        raw = pickle.load(open(ROOT / 'data/geom_raw/test_data.pickle', 'rb'))
        idx = np.random.RandomState(0).choice(len(raw), N_GEOM_REF, replace=False)
        geom = _merge(p.map(_geom, [raw[i][1][0].ToBinary() for i in idx], chunksize=64))
    train.discard(None)
    ref = dict(train_smiles=train, fcd=_fcd().precalc(test), n_test=len(test), geom=geom)
    torch.save(ref, f)
    return ref


def _w1(gen, ref, min_n=10):
    """VEDA: unweighted mean over reference types of W1(gen, ref).  Also a count-weighted version restricted to
    types with at least min_n samples on both sides (less sensitive to rare types)."""
    d, w = [], []
    for k, r in ref.items():
        g = gen.get(k)
        if g is None or len(g) == 0:
            continue
        d.append(stats.wasserstein_distance(g, r))
        w.append(len(r) if min(len(g), len(r)) >= min_n else 0)
    if not d:
        return float('nan'), float('nan')
    d, w = np.array(d), np.array(w, dtype=float)
    return float(d.mean()), float((d * w).sum() / max(w.sum(), 1))


def _circ(gen, ref):
    d = []
    for k, r in ref.items():
        g = gen.get(k)
        if g is None or len(g) == 0:
            continue
        m = lambda v: np.angle(np.exp(1j * np.radians(v)).mean(), deg=True)
        x = abs(m(g) - m(r))
        d.append(min(x, 360 - x))
    return float(np.mean(d)) if d else float('nan')


def dist_metrics(blocks, workers=8, ref=None):
    """blocks: RDKit binary molecules (None for failed ones) with a conformer and explicit hydrogens."""
    ref = ref or load_reference(workers)
    todo = [b for b in blocks if b is not None]
    with Pool(workers) as p:
        smi = [s for s in p.map(_smiles, todo, chunksize=64) if s]
        geom = _merge(p.map(_geom, todo, chunksize=64))
    out = dict(dist_n_valid=len(smi), dist_valid_frac=len(smi) / max(len(blocks), 1))
    if len(smi) < 2:
        return out
    out['fcd'] = float(_fcd()(gen=smi, pref=ref['fcd']))
    out['uniqueness'] = len(set(smi)) / len(smi)
    out['novelty'] = float(np.mean([s not in ref['train_smiles'] for s in smi]))
    out['bond_w1'], out['bond_w1_weighted'] = _w1(geom['bond'], ref['geom']['bond'])
    out['angle_w1'], out['angle_w1_weighted'] = _w1(geom['angle'], ref['geom']['angle'])
    out['torsion_circ'] = _circ(geom['torsion'], ref['geom']['torsion'])
    out['torsion_w1_weighted'] = _w1({k: np.abs(v) for k, v in geom['torsion'].items()},
                                     {k: np.abs(v) for k, v in ref['geom']['torsion'].items()})[1]
    gs, rs = geom['shape'], ref['geom']['shape']
    out['shape_w1'] = float(np.mean([stats.wasserstein_distance(gs[:, i], rs[:, i]) for i in range(3)]))
    return out


DIST_KEYS = ['dist_valid_frac', 'fcd', 'uniqueness', 'novelty', 'bond_w1', 'bond_w1_weighted', 'angle_w1',
             'angle_w1_weighted', 'torsion_circ', 'torsion_w1_weighted', 'shape_w1']
