"""Generate FlowMol3 teacher pairs (prior x0 -> sampled molecule) for reflow-style coupling.
Stores the full graph state including fake atoms, plus RDKit binaries for later PoseBusters filtering."""
import argparse, sys, time
from pathlib import Path
import dgl
import torch
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from flowmol.models.flowmol import FlowMol
from flowmol.analysis.molecule_builder import SampledMolecule
from flowmol.data_processing.utils import build_edge_idxs, get_upper_edge_mask, get_batch_idxs

p = argparse.ArgumentParser()
p.add_argument('--ckpt', default='flowmol/trained_models/flowmol3/checkpoints/last.ckpt')
p.add_argument('--n_mols', type=int, default=50000)
p.add_argument('--batch_size', type=int, default=500)
p.add_argument('--steps', type=int, default=100)
p.add_argument('--eta', type=float, default=10.0)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--out', required=True)
args = p.parse_args()
torch.manual_seed(args.seed)
dev = 'cuda:0'
fm = FlowMol.load_from_checkpoint(args.ckpt, map_location=dev).to(dev).eval()
recs, t0 = [], time.time()
with torch.no_grad():
    while len(recs) < args.n_mols:
        n_atoms = fm.sample_n_atoms(args.batch_size).to(dev)
        g = dgl.batch([dgl.graph(tuple(build_edge_idxs(int(n))), num_nodes=int(n), device=dev) for n in n_atoms])
        upper = get_upper_edge_mask(g)
        nb, _ = get_batch_idxs(g)
        g = fm.sample_prior(g, nb, upper)
        g = fm.vector_field.integrate(g, nb, upper_edge_mask=upper, n_timesteps=args.steps + 1,
                                      stochasticity=args.eta, high_confidence_threshold=0.9)
        g.edata['ue_mask'] = upper
        g = g.to('cpu')
        for gi in dgl.unbatch(g):
            um = gi.edata['ue_mask']
            a, c, e = gi.ndata['a_1'].argmax(-1), gi.ndata['c_1'].argmax(-1), gi.edata['e_1'][um].argmax(-1)
            try:
                sm = SampledMolecule(gi, fm.atom_type_map, ctmc_mol=True, fake_atoms=fm.fake_atoms,
                                     exclude_charges=fm.exclude_charges, explicit_aromaticity=fm.explicit_aromaticity)
                mb = sm.rdkit_mol.ToBinary() if sm.rdkit_mol is not None else None
            except Exception:
                mb = None
            recs.append(dict(x0=gi.ndata['x_0'].half(), x1=gi.ndata['x_1'].float(), a=a.byte(), c=c.byte(), e=e.byte(), mol=mb))
        print(f'{len(recs)} mols {time.time() - t0:.0f}s', flush=True)
torch.save(dict(recs=recs, args=vars(args), dims=dict(a=fm.vector_field.n_atom_types if hasattr(fm.vector_field, 'n_atom_types') else None)), args.out)
