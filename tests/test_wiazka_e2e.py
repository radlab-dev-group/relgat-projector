"""End-to-end smoke: tiny graph -> stage training -> inference API."""

import random

import pytest
import torch

from relgat_projector.trainer.wiazka_trainer import WiazkaTrainer, WiazkaConfig
from relgat_projector.inference import RelationFlow


def _toy_graph(seed=0, n_nodes=40, n_rels=3):
    rng = random.Random(seed)
    node2emb = {i: torch.randn(16) for i in range(n_nodes)}
    rel2idx = {f"r{i}": i for i in range(n_rels)}
    edges = []
    for i in range(n_nodes):
        for _ in range(3):
            j = rng.randrange(n_nodes)
            if j == i:
                continue
            edges.append((i, j, f"r{rng.randrange(n_rels)}"))
    return node2emb, rel2idx, edges


def test_trainer_smoke():
    node2emb, rel2idx, edges = _toy_graph()
    cfg = WiazkaConfig(
        op_dim=32, num_generators=8, gat_num_layers=1, gat_heads=1,
        batch_size=16, epochs=3, device="cpu",
    )
    trainer = WiazkaTrainer(node2emb, rel2idx, edges, cfg)
    history = trainer.train(epochs=2, log_every=1000)
    assert history and all(
        torch.isfinite(torch.tensor([v for v in h.values() if isinstance(v, float)])).all()
        for h in history
    )
    metrics = trainer.evaluate(n_probe=64)
    assert "eval/cos_edge" in metrics
    assert metrics["eval/cos_edge"] > -1.0


def test_inference_api_roundtrip():
    node2emb, rel2idx, edges = _toy_graph()
    cfg = WiazkaConfig(op_dim=32, num_generators=8, gat_num_layers=1, gat_heads=1,
                       batch_size=16, epochs=1, device="cpu")
    trainer = WiazkaTrainer(node2emb, rel2idx, edges, cfg)
    conn = RelationFlow.from_model(trainer.model, rel2idx, device="cpu")

    base = torch.randn(1, 16)
    v = conn.refine(base)
    assert v.shape == (1, 32)

    v2 = conn.apply(v, "r1")
    assert v2.shape == v.shape
    v3 = conn.apply_chain(v, ["r1", "r2"])
    assert v3.shape == v.shape

    # dial at 0 ~ identity
    v_same = conn.apply(v, "r1", lam=0.0)
    assert torch.allclose(v_same, v, atol=1e-5)

    # compiled chain == chain
    op = conn.compile(["r1", "r2"])
    via = op.apply(v)
    assert torch.allclose(via, v3, atol=1e-3)

    p = conn.applicability(v, "r0")
    assert (p >= 0).all() and (p <= 1).all()

    names, conf = conn.decode(v, v3)
    assert len(names) == 1 and 0.0 <= float(conf[0]) <= 1.0
