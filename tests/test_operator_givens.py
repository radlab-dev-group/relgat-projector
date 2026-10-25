"""
M1 property tests for the WIĄZKA operator (PLAN.md, milestone M1).

Verified properties:
  * T^{-1} T = I                     (exact inverse)
  * ||T x|| / ||x|| = s_r            (similarity: controlled scale only)
  * cos(u, v) == cos(Tu, Tv)         (isometry: cosine geometry preserved)
  * T_r o T_s != T_s o T_r           (non-commutativity for D >= 4)
  * T_r(0) = identity                (dial: lambda=0)
  * compiled chain == applied chain  (path compilation correctness)
  * gradient flows through apply()   (trainable)
"""

import pytest
import torch

from relgat_projector.core.operator.givens import GivensRelationOperator

D = 32
NUM_REL = 5
K = 16


@pytest.fixture
def op():
    torch.manual_seed(0)
    return GivensRelationOperator(D, NUM_REL, num_generators=K, use_scale=True)


def test_inverse_roundtrip(op):
    x = torch.randn(64, D)
    rel = torch.randint(0, NUM_REL, (64,))
    y = op.apply(x, rel)
    back = op.inverse(y, rel)
    assert torch.allclose(back, x, atol=1e-5), (back - x).abs().max().item()


def test_scale_and_isometry(op):
    u = torch.randn(128, D)
    v = torch.randn(128, D)
    rel = torch.ones(128, dtype=torch.long)
    tu = op.apply(u, rel)
    tv = op.apply(v, rel)
    # norm ratio == s_r for the relation
    ratio = (tu.norm(dim=1) / u.norm(dim=1))
    expected_s = op.log_scale[1].exp()
    assert torch.allclose(ratio, expected_s.expand_as(ratio), atol=1e-5)
    # cosine preserved (similarity transform)
    cos_in = torch.nn.functional.cosine_similarity(u, v, dim=-1)
    cos_out = torch.nn.functional.cosine_similarity(tu, tv, dim=-1)
    assert torch.allclose(cos_in, cos_out, atol=1e-5)


def test_non_commutativity(op):
    x = torch.randn(32, D)
    r1, r2 = 0, 2
    t12 = op.apply(op.apply(x, torch.full((32,), r1)), torch.full((32,), r2))
    t21 = op.apply(op.apply(x, torch.full((32,), r2)), torch.full((32,), r1))
    assert not torch.allclose(t12, t21, atol=1e-6)


def test_dial_identity(op):
    x = torch.randn(16, D)
    rel = torch.randint(0, NUM_REL, (16,))
    y = op.dial(x, rel, 0.0)
    assert torch.allclose(y, x, atol=1e-5)
    # linear interpolation of angles -> small lambda moves the vector a bit
    y_half = op.dial(x, rel, 0.5)
    assert not torch.allclose(y_half, x, atol=1e-4)


def test_apply_chain_vs_compiled(op):
    x = torch.randn(8, D)
    chain = torch.randint(0, NUM_REL, (8, 3))
    direct = op.apply_chain(x, chain)
    compiled = op.compiled(chain)
    via = compiled.apply(x)
    assert torch.allclose(direct, via, atol=1e-4)


def test_apply_chain_vs_sequential(op):
    x = torch.randn(8, D)
    chain = torch.randint(0, NUM_REL, (8, 4))
    out = x
    for k in range(4):
        out = op.apply(out, chain[:, k])
    assert torch.allclose(out, op.apply_chain(x, chain), atol=1e-5)


def test_gradients_flow(op):
    x = torch.randn(16, D)
    rel = torch.randint(0, NUM_REL, (16,))
    loss = op.apply(x, rel).pow(2).mean()
    loss.backward()
    assert op.theta.grad is not None
    assert op.log_scale.grad is not None
    assert torch.isfinite(op.theta.grad).all()


def test_compiled_matches_chain(op):
    x = torch.randn(16, D)
    chain = torch.randint(0, NUM_REL, (16, 4))
    compiled = op.compiled(chain)
    via = compiled.apply(x)
    direct = op.apply_chain(x, chain)
    assert torch.allclose(via, direct, atol=1e-4)
    m, s = compiled.as_matrix()
    assert m.shape == (16, D, D) and s.shape == (16,)
    # similarity matrix: columns have equal norm
    colnorms = m.norm(dim=-2)
    assert torch.allclose(colnorms, colnorms[:, :1], atol=1e-3)


def test_apply_to_matrix(op):
    x = torch.randn(24, D)
    a = op.apply(x, torch.full((24,), 3))
    b = op.apply_to_matrix(x, 3)
    assert torch.allclose(a, b)


def test_basis_stays_skew_under_gradient_steps():
    """The generator basis must remain in so(D) no matter where gradients go.

    The stored tensor is raw; skewness is enforced by the ``basis`` property.
    Without the projection the learned basis drifts out of the Lie algebra and
    the operator silently stops being a rotation (measured ISOT ~ 4e-2).
    """
    import torch

    torch.manual_seed(0)
    from relgat_projector.core.operator import build_operator

    op = build_operator("givens", dim=16, num_rel=4, num_generators=6)
    opt = torch.optim.SGD(op.parameters(), lr=0.5)
    x = torch.randn(8, 16)
    rel = torch.randint(0, 4, (8,))
    for _ in range(50):
        loss = op.apply(x, rel).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    b = op.basis
    assert torch.allclose(b, -b.transpose(-1, -2), atol=1e-5), \
        "learned basis is no longer skew-symmetric"

    # and the operator must still be (near-)orthogonal: cosine preserving
    u = torch.randn(64, 16)
    v = torch.randn(64, 16)
    tu = op.apply(u, torch.full((64,), 1, dtype=torch.long))
    tv = op.apply(v, torch.full((64,), 1, dtype=torch.long))
    ci = torch.nn.functional.cosine_similarity(u, v, dim=-1)
    co = torch.nn.functional.cosine_similarity(tu, tv, dim=-1)
    assert (ci - co).abs().max() < 1e-4, "operator no longer preserves cosines"


def test_rotation_matrix_survives_a_broken_batched_solve():
    """The lstsq fallback must produce the same orthogonal rotation.

    Multi-threaded oneMKL breaks batched ``torch.linalg.solve`` (invalid LU
    pivots), which used to take down every CPU inference of a 256-d model; the
    operator falls back instead of failing.
    """
    from relgat_projector.core.operator.givens import GivensRelationOperator

    op = GivensRelationOperator(dim=32, num_rel=4, num_generators=4)
    rel = torch.tensor([0, 1, 2, 3])
    x = torch.randn(4, 32)
    reference = op.apply(x, rel)

    original = GivensRelationOperator._LU_SOLVE_OK
    try:
        GivensRelationOperator._LU_SOLVE_OK = False
        fallback = op.apply(x, rel)
    finally:
        GivensRelationOperator._LU_SOLVE_OK = original
    assert torch.allclose(reference, fallback, atol=1e-4)

    R = op.relation_matrices()
    eye = torch.eye(R.shape[-1])
    assert (R.transpose(-1, -2) @ R - eye).abs().max() < 1e-4
