"""
Test of hypothesis H3 (PLAN.md section 9): the learned commutation matrix
``C_rs`` must distinguish commuting relation pairs from non-commuting ones.

The graphs are Cayley graphs, where commutation is a *ground truth* of the
structure:

* ``Z^2`` with the two axis generators -- the generators commute, so every
  cell closes and the pair must be learned as commuting;
* the dihedral group ``D4`` (rotation + reflection) -- the generators do NOT
  commute, so no square witnesses the pair and it must not be learned as
  commuting.
"""

import random

import pytest
import torch

from relgat_projector.core.operator import GivensRelationOperator
from relgat_projector.core.loss import CurvatureLoss
from relgat_projector.core.sampler.quadruples import QuadrupleSampler


def _cayley(elements, gens):
    """Return (out_adj, in_adj, num_nodes) for a Cayley graph."""
    index = {g: i for i, g in enumerate(elements)}
    out = {i: [] for i in range(len(elements))}
    inn = {i: [] for i in range(len(elements))}
    for g, i in index.items():
        for r, gen in gens.items():
            h = gen(g)
            j = index[h]
            out[i].append((j, r))
            inn[j].append((i, r))
    return out, inn, len(elements)


def _grid_cayley(n=4):
    """Z_n x Z_n: generators a (shift x) and b (shift y) commute."""
    elems = [(x, y) for x in range(n) for y in range(n)]

    def a(g):
        return ((g[0] + 1) % n, g[1])

    def b(g):
        return (g[0], (g[1] + 1) % n)

    return _cayley(elems, {0: a, 1: b})


def _dihedral_cayley(n=4):
    """D_n as right multiplication on {r^k, s r^k}; the generators do not commute.

    Group law: r^n = e, s^2 = e, s r = r^{-1} s.  Right multiplication:
    ``r^k . r = r^{k+1}``, ``s r^k . r = s r^{k+1}``,
    ``r^k . s = s r^{-k}``, ``s r^k . s = r^{-k}``.
    """
    elems = [("rot", k) for k in range(n)] + [("ref", k) for k in range(n)]

    def rot(g):
        # right multiplication by r
        return (g[0], (g[1] + 1) % n)

    def refl(g):
        # right multiplication by s
        if g[0] == "rot":
            return ("ref", (-g[1]) % n)
        return ("rot", (-g[1]) % n)

    return _cayley(elems, {0: rot, 1: refl})


def _train_commutation(adj, num_rel, num_nodes, steps=400, seed=0):
    out_adj, in_adj = adj
    quad = QuadrupleSampler(out_adj, in_adj, num_nodes=num_nodes, num_rel=num_rel,
                            rng=random.Random(seed + 1))
    loss = CurvatureLoss(num_rel, comm_weight=1.0, anti_weight=0.5, margin=0.3)
    opt = torch.optim.Adam(loss.parameters(), lr=5e-2)
    torch.manual_seed(seed)
    angles = torch.randn(32, num_nodes, requires_grad=False)
    for _ in range(steps):
        pos, neg = quad.sample_relation_pairs(16, max_neg=16)
        if not pos:
            break
        obj = loss.supervision(pos, neg)
        opt.zero_grad(set_to_none=True)
        obj.backward()
        opt.step()
    return loss.commutation().detach()


def test_grid_generators_commute():
    out, inn, n = _grid_cayley(4)
    quad = QuadrupleSampler(out, inn, num_nodes=n, num_rel=2, rng=random.Random(3))
    counts = quad.witness_counts(600)
    assert len(counts.get((0, 1), set())) >= 8, counts
    c = _train_commutation((out, inn), 2, n)
    # (0,1) is witnessed by every cell of the grid -> must be high
    assert c[0, 1].item() > 0.5, c


def test_dihedral_generators_do_not_commute():
    out, inn, n = _dihedral_cayley(4)
    quad = QuadrupleSampler(out, inn, num_nodes=n, num_rel=2, rng=random.Random(0))
    # a non-abelian Cayley graph has *no* witness of r . s = s . r
    assert quad.witness_counts(2000) == {}
    pos, _ = quad.sample_relation_pairs(16, max_neg=0, n_diamonds=2000)
    assert pos == []
    c = _train_commutation((out, inn), 2, n)
    # never supervised -> must not be learned as commuting
    assert c[0, 1].item() <= 0.5 + 1e-6, c


def test_commutation_is_symmetric():
    loss = CurvatureLoss(4)
    torch.nn.init.normal_(loss.comm_logit, 0.0, 1.0)
    c = loss.commutation()
    assert torch.allclose(c, c.t(), atol=1e-6)
