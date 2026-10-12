"""Dataset of FlowMol3 teacher pairs (x0 prior positions -> generated molecule), written by scripts/gen_teacher.py.
Graphs carry the same keys as MoleculeDataset (x_1_true, a_1_true, c_1_true, e_1_true, x_0), so DBTM.training_losses
uses the teacher's x0 as the position prior (coupled pair) when x0_random=0."""
import functools
from pathlib import Path
from typing import List

import dgl
import torch
from torch.nn.functional import one_hot

N_A, N_C, N_E = 11, 6, 4


class TeacherDataset(torch.utils.data.Dataset):
    def __init__(self, files: List[str], keep: str = 'pb'):
        self.recs = []
        for f in files:
            recs = torch.load(f)['recs']
            if keep != 'all':
                mask = torch.load(Path(f).with_suffix('.keep.pt'))[keep]
                recs = [r for r, k in zip(recs, mask) if k]
            self.recs += recs
        self.edge_cache = {}

    def __len__(self):
        return len(self.recs)

    @functools.cached_property
    def n_edges_per_graph(self):
        return torch.tensor([r['x1'].shape[0] ** 2 for r in self.recs])

    def _edges(self, n):
        if n not in self.edge_cache:
            up = torch.triu_indices(n, n, offset=1)
            self.edge_cache[n] = (up, torch.cat((up, up.flip(0)), dim=1))
        return self.edge_cache[n]

    def __getitem__(self, idx):
        r = self.recs[idx]
        n = r['x1'].shape[0]
        up, edges = self._edges(n)
        g = dgl.graph((edges[0], edges[1]), num_nodes=n)
        x1 = r['x1'].float()
        x0 = r['x0'].float()
        g.ndata['x_1_true'] = x1 - x1.mean(0, keepdim=True)
        g.ndata['x_0'] = x0 - x0.mean(0, keepdim=True)
        g.ndata['a_1_true'] = one_hot(r['a'].long(), N_A).float()
        g.ndata['c_1_true'] = one_hot(r['c'].long(), N_C).float()
        e = one_hot(r['e'].long(), N_E).float()
        g.edata['e_1_true'] = torch.cat((e, e))
        return g
