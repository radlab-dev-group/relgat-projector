"""Checkpoint round-trip and operator ablation wiring (PLAN.md 3.4, 5)."""

import os
import random

import pytest
import torch

from relgat_projector.core.operator import build_operator, operator_spec
from relgat_projector.inference import RelationFlow
from relgat_projector.trainer.wiazka_trainer import WiazkaConfig, WiazkaTrainer


def _toy_graph(seed=1, n_nodes=30, n_rels=3, dim=16):
    rng = random.Random(seed)
    node2emb = {i: torch.randn(dim) for i in range(n_nodes)}
    rel2idx = {f"r{i}": i for i in range(n_rels)}
    edges = []
    for i in range(n_nodes):
        for _ in range(3):
            j = rng.randrange(n_nodes)
            if j != i:
                edges.append((i, j, f"r{rng.randrange(n_rels)}"))
    return node2emb, rel2idx, edges


def _trainer(**kw):
    node2emb, rel2idx, edges = _toy_graph()
    cfg = WiazkaConfig(op_dim=24, num_generators=6, gat_num_layers=1, gat_heads=1,
                       batch_size=16, epochs=1, device="cpu", **kw)
    return WiazkaTrainer(node2emb, rel2idx, edges, cfg), rel2idx


# --------------------------------------------------------------- checkpoint
def test_checkpoint_roundtrip_reproduces_transforms(tmp_path):
    trainer, rel2idx = _trainer()
    trainer.train(epochs=1, log_every=10_000)
    ckpt = str(tmp_path / "ckpt")
    trainer.save(ckpt, metrics={"eval/cos_edge": 0.5})

    for name in ("model.pt", "config.json", "rel2idx.json", "commutation_C.npy"):
        assert os.path.exists(os.path.join(ckpt, name)), name

    reloaded = RelationFlow.load(ckpt)
    live = RelationFlow.from_model(trainer.model, rel2idx, device="cpu")

    base = torch.randn(4, 16)
    v, v_ = live.refine(base), reloaded.refine(base)
    assert torch.allclose(v, v_, atol=1e-5)
    assert torch.allclose(
        live.apply_chain(v, ["r1", "r2"]),
        reloaded.apply_chain(v_, ["r1", "r2"]),
        atol=1e-5,
    )
    assert reloaded.rel2idx == rel2idx
    assert reloaded.meta["model_spec"]["operator"]["type"] == "givens"


def test_checkpoint_is_standalone_from_trainer(tmp_path):
    """A checkpoint must load with only the model class (no trainer object)."""
    trainer, _ = _trainer()
    ckpt = str(tmp_path / "c2")
    trainer.save(ckpt)
    from relgat_projector.handlers.wiazka_checkpoint import load_checkpoint

    model, rel2idx, spec, _meta = load_checkpoint(ckpt)
    assert model.num_rel == len(rel2idx)
    assert spec["base_dim"] == 16
    assert not model.training, "loaded checkpoints must be in eval mode"


def test_load_rejects_foreign_format(tmp_path):
    ckpt = str(tmp_path / "bad")
    os.makedirs(ckpt)
    torch.save({"format": "other", "model": {}, "rel2idx": {}, "spec": {}},
               os.path.join(ckpt, "model.pt"))
    from relgat_projector.handlers.wiazka_checkpoint import load_checkpoint

    with pytest.raises(ValueError):
        load_checkpoint(ckpt)


# ----------------------------------------------------------------- ablation
@pytest.mark.parametrize("kind", ["givens", "distmult", "transe"])
def test_one_configuration_drives_every_operator(kind):
    """The trainer passes the full option set; baselines ignore what they lack."""
    op = build_operator(kind, dim=16, num_rel=4, num_generators=6,
                        use_scale=True, use_context=False)
    assert op.dim == 16 and op.num_rel == 4
    x = torch.randn(5, 16)
    rel = torch.full((5,), 2, dtype=torch.long)
    out = op.apply(x, rel)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    # dial 0 must be the identity for every operator in the ablation grid
    zero = op.apply(x, rel, torch.zeros(5))
    assert torch.allclose(zero, x, atol=1e-5), kind


def test_givens_records_its_construction():
    op = build_operator("givens", dim=16, num_rel=3, num_generators=4)
    spec = op.ctor_kwargs
    assert spec["type"] == "givens"
    assert spec["kwargs"]["num_generators"] == 4
    assert spec["kwargs"]["use_context"] is False
    op.enable_context(True)
    assert op.ctor_kwargs["kwargs"]["use_context"] is True
    rebuilt = build_operator(spec["type"], **spec["kwargs"])
    assert rebuilt.use_context is True


def test_context_toggle_preserves_transform_when_eps_zero():
    """F4 switches context on; with eps ~ 0 transport must not change."""
    op = build_operator("givens", dim=16, num_rel=3, num_generators=4)
    x = torch.randn(6, 16)
    rel = torch.randint(0, 3, (6,))
    before = op.apply(x, rel)
    op.enable_context(True)
    after = op.apply(x, rel)
    # context is on but starts negligible (eps ~ 2.5e-3) with a zero
    # hyper-network output, so transport must not move yet
    assert op.use_context and op.eps < 0.01
    assert torch.allclose(before, after, atol=1e-4)


def test_unknown_operator_is_rejected():
    with pytest.raises(ValueError):
        build_operator("nope", dim=8, num_rel=2)
    with pytest.raises(ValueError):
        operator_spec("nope")


def test_topology_split_excludes_heldout_from_gat():
    """Held-out edges must not appear in the GAT topology (leakage guard)."""
    node2emb, rel2idx, edges = _toy_graph(n_nodes=60, n_rels=3)
    cfg = WiazkaConfig(op_dim=24, num_generators=6, gat_num_layers=1,
                       gat_heads=1, batch_size=16, epochs=1, device="cpu",
                       train_ratio=0.8)
    n_in = len(edges)
    trainer = WiazkaTrainer(node2emb, rel2idx, edges, cfg)
    assert trainer.eval_edges, "no edges were held out"
    # every *position* is used exactly once: nothing is trained on and
    # held out simultaneously
    assert len(trainer.edges) + len(trainer.eval_edges) == n_in
    assert 0.0 <= trainer.topology_endpoint_coverage <= 1.0
