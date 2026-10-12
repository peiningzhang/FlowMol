"""FlowMol3-DBTM: few-step FlowMol3 via a time-free autonomous transport map on the simplex.

The pretrained FlowMol3 GVP network is reused as the map T. Differences from FlowMol3:
  * categorical features (a, c, e) live on continuous Gaussian->one-hot interpolants instead of masked CTMC states
    (the token embeddings become Linear layers initialised from the pretrained embedding rows);
  * the network gets a constant time input (autonomous map);
  * training uses a transport loss (finite-difference JVP), an anchor loss and a boundary loss, with
    per-atom partial-context interpolants;
  * sampling is K rounds of "predict, commit confident atoms, renoise the rest".
"""
import math
from typing import Dict, List

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from flowmol.models.ctmc_vector_field import CTMCVectorField
from flowmol.models.flowmol import FlowMol
from flowmol.data_processing.utils import build_edge_idxs, get_upper_edge_mask, get_batch_idxs
from flowmol.data_processing.priors import centered_normal_prior_batched_graph
from flowmol.analysis.molecule_builder import SampledMolecule
from flowmol.utils.embedding import get_time_embedding

DEFAULT_CFG = dict(
    sigma=1.0,              # std of the Gaussian prior on simplex embeddings
    h=1e-2,                 # finite-difference step for the JVP in the transport loss
    ada_c=1.0, ada_r=0.5,   # adaptive weight (||delta||^2 + c)^(-r)
    lam_transport=1.0, lam_anchor=1.0, lam_bnd=1.0,
    modality_w=dict(x=3.0, a=0.4, c=1.0, e=2.0),
    t_anchor_x=0.5,         # anchor threshold for positions (categoricals use the phase-transition formula)
    pc_prob=0.5,            # fraction of molecules trained with a partial-context interpolant
    scprop=0.5,             # prob. of feeding self-conditioning input during training
    lam_ril=0.0,            # refinement-in-loop: train on states built from the model's own (sampled) predictions
    lam_ldist=0.0,          # weight (relative to ril_x) of a local-distance term on the refinement-in-loop prediction:
    ldist_cut=3.0,          # squared error of pair distances for pairs closer than ldist_cut A in the data (bonds, 1-3, some 1-4)
    ril_x_only=0,           # 1: the refinement-in-loop loss is applied to positions only
    ril_tb_max=0.97,        # upper end of the level the rolled-out prediction is re-noised to
    ril_tmin=0.35, ril_tmax=0.85,  # starting level of the rollout is U(ril_tmin, ril_tmax); high enough that predictions stay near the data
    x0_random=0,            # 1: independent Gaussian position prior in training (no OT alignment), matching inference
)


def phase_transition_t(V: int, sigma: float = 1.0, a: float = 1.0):
    return 1.0 - (1.0 + sigma * math.sqrt(2.0 * math.log(V))) ** (-1.0 / a)


class SoftVF(CTMCVectorField):
    """Same weights as the pretrained CTMCVectorField but takes continuous simplex inputs and no time."""

    def forward(self, g, a_t, c_t, e_t_upper, x_t, node_batch_idx, upper_edge_mask, prev=None):
        a_emb = self.token_embeddings['a'](a_t)
        c_emb = self.token_embeddings['c'](c_t)
        t_emb = get_time_embedding(torch.zeros(g.batch_size, device=g.device), embedding_dim=self.time_embedding_dim)
        t_emb = t_emb[node_batch_idx]
        s = self.scalar_embedding(torch.cat([a_emb, c_emb, t_emb], dim=-1))

        e_full = torch.empty(g.num_edges(), e_t_upper.shape[-1], device=g.device, dtype=e_t_upper.dtype)
        e_full[upper_edge_mask] = e_t_upper
        e_full[~upper_edge_mask] = e_t_upper
        e_feat = self.edge_embedding(self.token_embeddings['e'](e_full))

        vec = torch.zeros((g.num_nodes(), self.n_vec_channels, 3), device=g.device)
        g.ndata['x_t'] = x_t
        if prev is not None:
            s, x_t, vec, e_feat = self.self_conditioning_residual_layer(
                g, s, x_t, vec, e_feat, prev, node_batch_idx, upper_edge_mask)
        return self.denoise_graph(g, s, vec, x_t, e_feat, node_batch_idx, upper_edge_mask,
                                  apply_softmax=False, remove_com=True)


def convert_to_soft(vf: CTMCVectorField):
    for feat in ['a', 'c', 'e']:
        emb = vf.token_embeddings[feat]
        V = vf.n_cat_feats[feat]
        lin = nn.Linear(V, emb.weight.shape[1], bias=False)
        lin.weight.data = emb.weight.data[:V].T.clone()   # drop the mask-token row
        vf.token_embeddings[feat] = lin
    vf.__class__ = SoftVF
    return vf


def recenter(x, node_batch_idx, counts):
    sums = torch.zeros(counts.shape[0], 3, device=x.device, dtype=x.dtype).index_add_(0, node_batch_idx, x)
    return x - (sums / counts.unsqueeze(-1))[node_batch_idx]


def probs_of(out):
    return dict(x=out['x'], a=F.softmax(out['a'], -1), c=F.softmax(out['c'], -1), e=F.softmax(out['e'], -1))


class DBTM(nn.Module):
    def __init__(self, base: FlowMol, cfg: dict = None):
        super().__init__()
        self.cfg = {**DEFAULT_CFG, **(cfg or {})}
        self.net: SoftVF = convert_to_soft(base.vector_field)
        self.atom_type_map = base.atom_type_map
        self.fake_atoms = base.fake_atoms
        self.explicit_aromaticity = base.explicit_aromaticity
        self.dims = dict(a=base.n_atom_types, c=base.n_atom_charges, e=base.n_bond_types)
        self.register_buffer('n_atoms_vals', base.n_atoms_map.clone())
        self.register_buffer('n_atoms_probs', base.n_atoms_dist.probs.clone())
        self.t_star = {k: phase_transition_t(v, self.cfg['sigma']) for k, v in self.dims.items()}

    # ------------------------------------------------------------------ helpers
    def _forward(self, g, st, nb, upper, prev=None):
        return self.net(g, st['a'], st['c'], st['e'], st['x'], nb, upper, prev=prev)

    def _state(self, ctx, t_n, t_e, counts, nb):
        """Interpolant I at per-token times (t=1 -> clean data)."""
        tn, te = t_n.unsqueeze(-1), t_e.unsqueeze(-1)
        x = (1 - tn) * ctx['x0'] + tn * ctx['x1']
        return dict(a=(1 - tn) * ctx['na'] + tn * ctx['a1'], c=(1 - tn) * ctx['nc'] + tn * ctx['c1'],
                    e=(1 - te) * ctx['ne'] + te * ctx['e1'], x=recenter(x, nb, counts))

    # ------------------------------------------------------------------ training
    def training_losses(self, g: dgl.DGLGraph):
        cfg, dev = self.cfg, g.device
        nb, eb = get_batch_idxs(g)
        upper = get_upper_edge_mask(g)
        B, N = g.batch_size, g.num_nodes()
        counts = g.batch_num_nodes().to(dev).float()
        src, dst = g.edges()
        src_u, dst_u, eb_u = src[upper], dst[upper], eb[upper]
        sig = cfg['sigma']

        ctx = dict(x1=g.ndata['x_1_true'], x0=g.ndata['x_0'], a1=g.ndata['a_1_true'], c1=g.ndata['c_1_true'],
                   e1=g.edata['e_1_true'][upper])
        if cfg['x0_random']:
            ctx['x0'] = recenter(torch.randn_like(ctx['x1']), nb, counts)
        ctx['na'] = torch.randn_like(ctx['a1']) * sig
        ctx['nc'] = torch.randn_like(ctx['c1']) * sig
        ctx['ne'] = torch.randn_like(ctx['e1']) * sig
        a_idx, c_idx, e_idx = ctx['a1'].argmax(-1), ctx['c1'].argmax(-1), ctx['e1'].argmax(-1)

        # times and partial context
        t = torch.rand(B, device=dev)
        use_pc = torch.rand(B, device=dev) < cfg['pc_prob']
        f = torch.rand(B, device=dev)
        clean = (torch.rand(N, device=dev) < f[nb]) & use_pc[nb]
        clean_e = clean[src_u] & clean[dst_u]
        t_n = torch.where(clean, torch.ones_like(t[nb]), t[nb])
        t_e = torch.where(clean_e, torch.ones_like(t[eb_u]), t[eb_u])
        m_n, m_e = (~clean).float(), (~clean_e).float()

        I = self._state(ctx, t_n, t_e, counts, nb)

        # self-conditioning input: prediction from an earlier (noisier) state of the same trajectory
        prev = None
        if torch.rand(1).item() < cfg['scprop']:
            u = torch.rand(B, device=dev)
            tp_n = torch.where(clean, torch.ones_like(t_n), (t * u)[nb])
            tp_e = torch.where(clean_e, torch.ones_like(t_e), (t * u)[eb_u])
            with torch.no_grad():
                prev = probs_of(self._forward(g, self._state(ctx, tp_n, tp_e, counts, nb), nb, upper))

        out = self._forward(g, I, nb, upper, prev=prev)
        P = probs_of(out)

        # ---- transport loss (finite-difference JVP along the interpolant velocity)
        h = cfg['h']
        with torch.no_grad():
            vel = dict(a=(ctx['a1'] - ctx['na']) * m_n[:, None], c=(ctx['c1'] - ctx['nc']) * m_n[:, None],
                       e=(ctx['e1'] - ctx['ne']) * m_e[:, None], x=(ctx['x1'] - ctx['x0']) * m_n[:, None])
            I2 = {k: I[k] + h * vel[k] for k in I}
            I2['x'] = recenter(I2['x'], nb, counts)
            P2 = probs_of(self._forward(g, I2, nb, upper, prev=prev))

        L, logs = {}, {}
        for k, m in [('x', m_n), ('a', m_n), ('c', m_n), ('e', m_e)]:
            delta = (P2[k] - P[k].detach()) / h
            sq = delta.pow(2).sum(-1) if k != 'x' else delta.pow(2).mean(-1)
            w = (sq + cfg['ada_c']).pow(-cfg['ada_r'])
            tgt = (P[k].detach() + delta)
            err = (P[k] - tgt).pow(2)
            err = err.sum(-1) if k != 'x' else err.mean(-1)
            L[f'tr_{k}'] = (w * err * m).sum() / m.sum().clamp(min=1)
            logs[f'dnorm_{k}'] = (sq * m).sum().detach() / m.sum().clamp(min=1)

        # ---- anchor loss (late phase of the trajectory)
        for k, idx, tt, m, logits in [('a', a_idx, t_n, m_n, out['a']), ('c', c_idx, t_n, m_n, out['c']),
                                      ('e', e_idx, t_e, m_e, out['e'])]:
            sel = m * (tt > self.t_star[k]).float()
            L[f'anc_{k}'] = (F.cross_entropy(logits, idx, reduction='none') * sel).sum() / sel.sum().clamp(min=1)
        sel = m_n * (t_n > cfg['t_anchor_x']).float()
        L['anc_x'] = ((out['x'] - ctx['x1']).pow(2).mean(-1) * sel).sum() / sel.sum().clamp(min=1)

        # ---- boundary loss: clean data maps to itself
        clean_state = dict(a=ctx['a1'], c=ctx['c1'], e=ctx['e1'], x=ctx['x1'])
        ob = self._forward(g, clean_state, nb, upper)
        L['bnd_a'] = F.cross_entropy(ob['a'], a_idx)
        L['bnd_c'] = F.cross_entropy(ob['c'], c_idx)
        L['bnd_e'] = F.cross_entropy(ob['e'], e_idx)
        L['bnd_x'] = (ob['x'] - ctx['x1']).pow(2).mean()

        if cfg['lam_ril'] > 0:
            with torch.no_grad():
                ta = cfg['ril_tmin'] + torch.rand(B, device=dev) * (cfg['ril_tmax'] - cfg['ril_tmin'])
                Ia = self._state(ctx, ta[nb], ta[eb_u], counts, nb)
                Pa = probs_of(self._forward(g, Ia, nb, upper))
                tb = ta + (cfg['ril_tb_max'] - ta) * torch.rand(B, device=dev)
                tbn, tbe = tb[nb].unsqueeze(-1), tb[eb_u].unsqueeze(-1)
                pick = lambda p: F.one_hot(torch.multinomial(p, 1).squeeze(-1), p.shape[-1]).float()
                Ib = dict(a=(1 - tbn) * torch.randn_like(Pa['a']) * sig + tbn * pick(Pa['a']),
                          c=(1 - tbn) * torch.randn_like(Pa['c']) * sig + tbn * pick(Pa['c']),
                          e=(1 - tbe) * torch.randn_like(Pa['e']) * sig + tbe * pick(Pa['e']),
                          x=recenter((1 - tbn) * (torch.randn_like(Pa['x']) if cfg['x0_random'] else ctx['x0'])
                                     + tbn * Pa['x'], nb, counts))  # coupled prior: same x0, as in the 'cmx' sampler
            ob_r = self._forward(g, Ib, nb, upper, prev=Pa if torch.rand(1).item() < cfg['scprop'] else None)
            L['ril_a'] = F.cross_entropy(ob_r['a'], a_idx)
            L['ril_c'] = F.cross_entropy(ob_r['c'], c_idx)
            L['ril_e'] = F.cross_entropy(ob_r['e'], e_idx)
            L['ril_x'] = (ob_r['x'] - ctx['x1']).pow(2).mean()
            if cfg['lam_ldist'] > 0:
                d_true = (ctx['x1'][src_u] - ctx['x1'][dst_u]).norm(dim=-1)
                d_pred = (ob_r['x'][src_u] - ob_r['x'][dst_u]).norm(dim=-1)
                near = (d_true < cfg['ldist_cut']).float()
                L['ril_ldist'] = ((d_pred - d_true).pow(2) * near).sum() / near.sum().clamp(min=1)
        total = 0.0
        for k in ['x', 'a', 'c', 'e']:
            if cfg['lam_ril'] > 0 and (k == 'x' or not cfg['ril_x_only']):
                total = total + cfg['modality_w'][k] * cfg['lam_ril'] * L[f'ril_{k}']
            total = total + cfg['modality_w'][k] * (cfg['lam_transport'] * L[f'tr_{k}'] + cfg['lam_anchor'] * L[f'anc_{k}']
                                                    + cfg['lam_bnd'] * L[f'bnd_{k}'])
        if cfg['lam_ril'] > 0 and cfg['lam_ldist'] > 0:
            total = total + cfg['modality_w']['x'] * cfg['lam_ril'] * cfg['lam_ldist'] * L['ril_ldist']
        with torch.no_grad():  # diagnostics: accuracy of the map on noisy tokens at late times
            late = (m_n * (t_n > 0.5).float()).bool()
            logs['acc_a_late'] = (out['a'].argmax(-1) == a_idx)[late].float().mean() if late.any() else torch.tensor(0.)
        logs.update({k: v.detach() for k, v in L.items()})
        logs['total'] = total.detach()
        return total, logs

    # ------------------------------------------------------------------ sampling
    def sample_n_atoms(self, n, device):
        idx = torch.multinomial(self.n_atoms_probs, n, replacement=True)
        return self.n_atoms_vals[idx].to(device)

    @torch.no_grad()
    def sample(self, n_atoms: torch.Tensor, K: int = 1, kappa: float = 0.9, device='cuda:0',
               commit: str = 'confidence', return_graph: bool = False, mode: str = 'commit',
               taus=None, sample_cats: bool = True, argmax_above: float = 2.0, trace=None) -> List[SampledMolecule]:
        # trace: pass [] to get the molecules predicted at every round (list of lists)
        """K rounds (= K network evaluations) of predict / commit / renoise."""
        sig = self.cfg['sigma']
        edge_idxs = {int(n): build_edge_idxs(n) for n in torch.unique(n_atoms)}
        g = dgl.batch([dgl.graph((edge_idxs[int(n)][0], edge_idxs[int(n)][1]), num_nodes=int(n), device=device)
                       for n in n_atoms])
        upper = get_upper_edge_mask(g)
        nb, eb = get_batch_idxs(g)
        counts = g.batch_num_nodes().to(device).float()
        src, dst = g.edges()
        src_u, dst_u = src[upper], dst[upper]
        N, Eu = g.num_nodes(), int(upper.sum())
        A, C, E = self.dims['a'], self.dims['c'], self.dims['e']

        a = torch.randn(N, A, device=device) * sig
        c = torch.randn(N, C, device=device) * sig
        e = torch.randn(Eu, E, device=device) * sig
        x = centered_normal_prior_batched_graph(g, nb, std=1.0)
        # 'cmx': re-noise positions with the round-0 prior sample (keeps the x0->x1 coupling of teacher-pair training);
        # 'cmall': also reuse the round-0 categorical noise
        n0 = dict(a=a.clone(), c=c.clone(), e=e.clone(), x=x.clone())
        fix_x, fix_cat = mode in ('cmx', 'cmall'), mode == 'cmall'
        if mode in ('cmx', 'cmall'):
            mode = 'cm'
        committed = torch.zeros(N, dtype=torch.bool, device=device)
        committed_e = torch.zeros(Eu, dtype=torch.bool, device=device)
        prev = None

        if mode == 'cm':
            # consistency-style multistep: re-noise the full prediction to level tau_r with fresh noise
            taus = taus if taus is not None else [r / K for r in range(1, K)]
        for r in range(1, K + 1):
            P = probs_of(self.net(g, a, c, e, x, nb, upper, prev=prev))
            if trace is not None:
                trace.append(P)
            if r == K:
                break
            if mode == 'cm':
                tau = taus[r - 1]
                pick = (lambda p: torch.multinomial(p, 1).squeeze(-1)) if sample_cats and tau < argmax_above \
                    else (lambda p: p.argmax(-1))
                na, nc, ne = (n0['a'], n0['c'], n0['e']) if fix_cat else \
                    (torch.randn_like(a) * sig, torch.randn_like(c) * sig, torch.randn_like(e) * sig)
                nx = n0['x'] if fix_x else torch.randn_like(x)
                a = (1 - tau) * na + tau * F.one_hot(pick(P['a']), A).float()
                c = (1 - tau) * nc + tau * F.one_hot(pick(P['c']), C).float()
                e = (1 - tau) * ne + tau * F.one_hot(pick(P['e']), E).float()
                x = recenter((1 - tau) * nx + tau * P['x'], nb, counts)
                prev = P
                continue
            conf = torch.minimum(P['a'].max(-1).values, P['c'].max(-1).values) if commit == 'confidence' \
                else torch.rand(N, device=device)
            conf = torch.where(committed, torch.full_like(conf, -1.0), conf)
            R_b = torch.zeros(g.batch_size, device=device).index_add_(0, nb, (~committed).float())
            n_r = torch.ceil(R_b / (K - r + 1))
            key = nb.double() * 4 + (1 - conf.double())
            order = torch.argsort(key)
            starts = torch.cumsum(counts.long(), 0) - counts.long()
            rank = torch.empty(N, dtype=torch.long, device=device)
            rank[order] = torch.arange(N, device=device) - starts[nb[order]]
            newly = ((conf >= kappa) | (rank < n_r[nb])) & ~committed
            committed = committed | newly
            a = torch.where(newly[:, None], F.one_hot(P['a'].argmax(-1), A).float(), a)
            c = torch.where(newly[:, None], F.one_hot(P['c'].argmax(-1), C).float(), c)
            x = torch.where(newly[:, None], P['x'], x)
            ce = committed[src_u] & committed[dst_u]
            new_e = ce & ~committed_e
            e = torch.where(new_e[:, None], F.one_hot(P['e'].argmax(-1), E).float(), e)
            committed_e = ce
            # fresh noise for everything not yet committed
            un = ~committed
            a = torch.where(un[:, None], torch.randn_like(a) * sig, a)
            c = torch.where(un[:, None], torch.randn_like(c) * sig, c)
            e = torch.where((~committed_e)[:, None], torch.randn_like(e) * sig, e)
            x = torch.where(un[:, None], torch.randn_like(x), x)
            x = recenter(x, nb, counts)
            prev = P

        if trace is not None:
            return [self.build_mols(g, Pr, upper) for Pr in trace]
        return self.build_mols(g, P, upper)

    def build_mols(self, g, P, upper) -> List[SampledMolecule]:
        """Molecules from a prediction dict (x and a/c/e probabilities; e on upper edges)."""
        A, C, E = self.dims['a'], self.dims['c'], self.dims['e']
        e_idx = P['e'].argmax(-1)
        e_full = torch.empty(g.num_edges(), dtype=torch.long, device=e_idx.device)
        e_full[upper] = e_idx
        e_full[~upper] = e_idx
        g.ndata['x_1'] = P['x']
        g.ndata['a_1'] = F.one_hot(P['a'].argmax(-1), A + 1).float()
        g.ndata['c_1'] = F.one_hot(P['c'].argmax(-1), C + 1).float()
        g.edata['e_1'] = F.one_hot(e_full, E + 1).float()
        g.edata['ue_mask'] = upper
        gc = g.to('cpu')
        return [SampledMolecule(gi, self.atom_type_map, ctmc_mol=True, fake_atoms=self.fake_atoms,
                                exclude_charges=False, explicit_aromaticity=self.explicit_aromaticity)
                for gi in dgl.unbatch(gc)]

    def sample_random_sizes(self, n, K=1, device='cuda:0', **kw):
        return self.sample(self.sample_n_atoms(n, device), K=K, device=device, **kw)


def load_dbtm(base_ckpt: str, dbtm_ckpt: str = None, cfg: dict = None, device='cuda:0'):
    base = FlowMol.load_from_checkpoint(base_ckpt, map_location='cpu')
    model = DBTM(base, cfg)
    if dbtm_ckpt is not None:
        sd = torch.load(dbtm_ckpt, map_location='cpu')
        model.load_state_dict(sd['model'])
        model.cfg.update(sd.get('cfg', {}))
    return model.to(device)
