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


def test_loss_owned_parameters_are_optimised():
    """The commutation matrix lives in CurvatureLoss, not in the model.

    A trainer that optimises ``model.parameters()`` only freezes it silently,
    which is why the learned commutation map used to stay at 0.5 forever.
    """
    node2emb, rel2idx, edges = _toy_graph()
    cfg = WiazkaConfig(op_dim=24, num_generators=6, gat_num_layers=1,
                       gat_heads=1, batch_size=16, epochs=1, device="cpu")
    trainer = WiazkaTrainer(node2emb, rel2idx, edges, cfg)
    opt_ids = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
    for p in trainer.curvature_loss.parameters():
        assert id(p) in opt_ids, "C_rs is not optimised by the trainer"


def _past_min_steps(mgr, stage_index):
    """Put the manager in `stage_index` with its minimum steps already spent."""
    from relgat_projector.trainer.stages.stage_manager import STAGES

    mgr.current = stage_index
    mgr.steps_in_stage = STAGES[mgr.order[stage_index]].min_steps
    return mgr


def test_stage_manager_direction_of_better():
    from relgat_projector.trainer.stages import StageManager

    # F2 gate: the ranking lift over the raw-base baseline, higher is better
    mgr = _past_min_steps(StageManager(), 1)
    assert mgr.should_stop({"eval/mrr_lift": 0.05}) is True
    assert mgr.should_stop({"eval/mrr_lift": -0.05}) is False
    # F3 gate: the two-hop transport must beat the identity on its own target
    mgr3 = _past_min_steps(StageManager(), 2)
    assert mgr3.should_stop({"eval/path_lift": 0.05}) is True
    assert mgr3.should_stop({"eval/path_lift": -0.05}) is False
    # F1 gate: the anchor is higher-is-better
    mgr1 = _past_min_steps(StageManager(), 0)
    assert mgr1.should_stop({"eval/anchor": 0.99}) is True
    assert mgr1.should_stop({"eval/anchor": 0.1}) is False


def test_stage_gate_requires_every_metric():
    """One trivially-satisfied metric must not open the gate.

    ``eval/cce`` is ~0 everywhere for an exactly invertible operator, so an
    any-metric rule let F2 advance after a single evaluation.
    """
    from relgat_projector.trainer.stages import StageManager

    mgr = _past_min_steps(StageManager(), 1)
    assert mgr.should_stop({"eval/cce": 0.0}) is False, "cce alone opened F2"
    # metrics that are not part of the gate are ignored
    assert mgr.should_stop({"eval/cce": 0.0, "eval/isot": 0.01,
                            "eval/cos_edge": -1.0,
                            "eval/mrr_lift": 0.5}) is True
    # a metric that was never computed cannot open the gate either
    mgr2 = _past_min_steps(StageManager(), 1)
    assert mgr2.should_stop({"eval/isot": 0.0}) is False


def test_stage_cannot_be_left_before_min_steps():
    """A gate that is already true at initialisation must not rush the curriculum."""
    from relgat_projector.trainer.stages import StageManager

    # F1's anchor gate is satisfied by construction at step 0 (the refinement
    # starts as the exact identity of the projected base vectors)
    mgr = StageManager()
    assert mgr.should_stop({"eval/anchor": 1.0}) is False
    mgr.steps_in_stage = 1
    assert mgr.should_stop({"eval/anchor": 1.0}) is False
    _past_min_steps(mgr, 0)
    assert mgr.should_stop({"eval/anchor": 1.0}) is True
