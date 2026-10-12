"""Shared evaluation helpers: timed sampling + FlowMol3 metric suite + wandb logging."""
import math
import time
from pathlib import Path

import numpy as np
import torch

from flowmol.analysis.metrics import SampleAnalyzer

PROCESSED_DIR = Path(__file__).resolve().parent.parent / 'data' / 'geom_full_kekulized'


def timed_sample(sample_fn, n_mols, batch_size, seed=0):
    """sample_fn(batch_size) -> list[SampledMolecule]. Returns molecules and seconds (cuda-synced)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    mols = []
    torch.cuda.synchronize()
    t0 = time.time()
    while len(mols) < n_mols:
        bs = min(batch_size, n_mols - len(mols))
        mols.extend(sample_fn(bs))
    torch.cuda.synchronize()
    return mols, time.time() - t0


def compute_metrics(mols, n_subsets=1, pb_workers=8, posebusters=True, mmff=True):
    analyzer = SampleAnalyzer(processed_data_dir=PROCESSED_DIR, pb_energy=True, pb_workers=pb_workers)
    per = len(mols) / n_subsets
    subset_metrics = []
    for i in range(n_subsets):
        sub = mols[int(i * per):int((i + 1) * per)]
        subset_metrics.append(analyzer.analyze(sub, energy_div=False, posebusters=posebusters,
                                               functional_validity=True))
    out = {}
    for k in subset_metrics[0]:
        vals = np.array([d[k] for d in subset_metrics], dtype=float)
        out[k] = float(vals.mean())
        if n_subsets > 1:
            out[f'{k}_ci95'] = float(1.96 * vals.std() / math.sqrt(n_subsets))
    if mmff:
        out.update(mmff_metrics(mols))
    return out


KEY_METRICS = ['frac_valid_mols', 'pb_valid', 'frac_mols_stable_valence', 'frac_atoms_stable',
               'avg_frag_frac', 'ood_rate', 'flag_rate', 'cum_reos_dev',
               'mmff_ok_frac', 'mmff_strain_median', 'mmff_strain_per_atom_median', 'mmff_relax_rmsd_median']


def summarize(m):
    keys = [k for k in KEY_METRICS if k in m] + [k for k in m if 'reos' in k and k not in KEY_METRICS]
    return ' '.join(f'{k}={m[k]:.4f}' for k in keys if isinstance(m.get(k), float))


def _mmff_one(mb):
    """Return (E_gen, E_relaxed, heavy-atom RMSD, n_atoms) in kcal/mol for one molecule, or None."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, rdMolAlign
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
        rmsd = rdMolAlign.AlignMol(m, m0, atomMap=[(i, i) for i in heavy])
        return e0, e1, rmsd, m.GetNumAtoms()
    except Exception:
        return None


def mmff_metrics(mols, workers=16):
    """MMFF94 strain of generated conformers. Denominator for mmff_ok_frac is all molecules."""
    from multiprocessing import Pool
    blocks = [m.rdkit_mol.ToBinary() if m.rdkit_mol is not None else None for m in mols]
    todo = [b for b in blocks if b is not None]
    with Pool(workers) as p:
        res = [r for r in p.map(_mmff_one, todo, chunksize=16) if r is not None]
    if not res:
        return {'mmff_ok_frac': 0.0}
    e0, e1, rmsd, n = map(np.array, zip(*res))
    strain = e0 - e1
    return {'mmff_ok_frac': len(res) / len(mols),
            'mmff_strain_median': float(np.median(strain)),
            'mmff_strain_mean': float(np.mean(np.clip(strain, None, 1e4))),
            'mmff_strain_per_atom_median': float(np.median(strain / n)),
            'mmff_relax_rmsd_median': float(np.median(rmsd))}
