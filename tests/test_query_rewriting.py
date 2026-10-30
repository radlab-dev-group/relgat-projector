"""QR-Recall: the metric that decides whether the transform is useful."""

import math
import random

import pytest
import torch

from relgat_projector.core.metrics.query_rewriting import (
    _normalize,
    applicability_split,
    gold_pairs,
    qr_recall,
    rank_of_target,
)
from relgat_projector.trainer.wiazka_trainer import WiazkaConfig, WiazkaTrainer


def _index():
    """4 texts: north, east, south, west (unit circle)."""
    angles = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]
    return torch.tensor([[math.cos(a), math.sin(a)] for a in angles])


def test_rank_of_target_basic():
    S = torch.tensor([[0.9, 0.5, 0.1], [0.1, 0.2, 0.8]])
    t = torch.tensor([0, 2])
    assert rank_of_target(S, t).tolist() == [1, 1]
    mask = torch.zeros(2, 3, dtype=torch.bool)
    mask[0, 0] = True          # the gold itself may not be masked away
    assert rank_of_target(S, t, mask)[1] == 1


def test_rank_of_target_counts_only_unmasked():
    S = torch.tensor([[0.9, 0.8, 0.1]])
    t = torch.tensor([2])
    assert rank_of_target(S, t).item() == 3
    mask = torch.tensor([[True, True, False]])
    assert rank_of_target(S, t, mask).item() == 1


def test_rank_tie_break_conventions():
    S = torch.tensor([[0.5, 0.5, 0.1]])
    t = torch.tensor([0])
    assert rank_of_target(S, t, tie_break="optimistic").item() == 1
    assert rank_of_target(S, t, tie_break="pessimistic").item() == 2


def test_identity_transform_has_zero_delta():
    idx = _index()
    pairs = [(0, 1), (1, 2)]
    res = qr_recall(pairs, idx, transform=lambda x: x, ks=(1,))
    assert res.delta(1) == 0.0
    assert res.before[1] == res.after[1]


def test_useful_transform_is_detected():
    """Gold answers ARE the rotated neighbours; the rotation finds them all."""
    idx = _index()
    rot = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    gold = {i: int(torch.argmax((_normalize(idx[i]) @ _normalize(idx).t() * 0
                                 + (idx @ rot.t() - idx[i]).norm(dim=-1).neg()))
                    if False else 0) for i in []}
    # gold for query i = the j whose vector is closest to rot(x_i)
    sim = _normalize(idx @ rot.t()) @ _normalize(idx).t()
    pairs = [(i, int(sim[i].argmax())) for i in range(4)]
    res = qr_recall(pairs, idx, transform=lambda x: x @ rot.t(), ks=(1, 2))
    assert res.after[1] == 1.0
    assert res.delta(1) > 0.0
    assert res.mrr_after > res.mrr_before


def test_harmful_transform_is_reported_as_such():
    idx = _index()
    pairs = [(0, 1), (1, 2)]
    # a reflection is a worse answer for an "east" gold than the identity
    reflect = torch.tensor([[-1.0, 0.0], [0.0, 1.0]])
    good = qr_recall(pairs, idx, transform=lambda x: x, ks=(1,))
    bad = qr_recall(pairs, idx, transform=lambda x: x @ reflect, ks=(1,))
    assert bad.after[1] < good.after[1]


def test_query_itself_is_masked_out():
    """The query's own vector must not count as a hit for its gold answer."""
    idx = _index()
    # the gold is the *closest other* vector: the query's own vector would
    # win the ranking without the mask
    sim = _normalize(idx) @ _normalize(idx).t()
    for i in range(4):
        sim[i, i] = -1.0
    pairs = [(i, int(sim[i].argmax())) for i in range(4)]
    no_mask = qr_recall(pairs, idx, transform=lambda x: x, ks=(1,),
                        exclude_self=False)
    masked = qr_recall(pairs, idx, transform=lambda x: x, ks=(1,),
                       exclude_self=True)
    # the unmasked ranking is dominated by the query's own vector: rank 1
    # would have to beat cos(x, x) = 1, impossible with unit vectors
    assert (no_mask.ranks_after == 2).all()
    # once the own vector is masked, the gold (closest other) is rank 1
    assert (masked.ranks_after == 1).all()


def test_other_golds_are_filtered():
    idx = torch.eye(3)
    # query 0 has two gold answers; with filtering the second one must not
    # count against it
    pairs = [(0, 1), (0, 2)]
    res = qr_recall(pairs, idx, transform=lambda x: x, ks=(1,),
                    filter_others=True)
    assert all(r == 1 for r in res.ranks_after.tolist())


def test_gold_pairs_selection():
    edges = [(0, 1, 0), (1, 2, 1), (2, 3, 0)]
    assert gold_pairs(edges, 0) == [(0, 1), (2, 3)]
    assert gold_pairs(edges, 1) == [(1, 2)]
    assert gold_pairs(edges, 5) == []
    assert len(gold_pairs(edges, 0, max_pairs=1)) == 1


def test_empty_relation_is_not_an_error():
    res = qr_recall([], torch.eye(3), transform=lambda x: x, ks=(1,))
    assert res.n_queries == 0
    assert math.isnan(res.delta(1))


def test_applicability_split_separates_groups():
    idx = _index()
    rot = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    sim = _normalize(idx @ rot.t()) @ _normalize(idx).t()
    pairs = [(i, int(sim[i].argmax())) for i in range(4)]
    scores = torch.tensor([0.9, 0.8, 0.1, 0.2])
    res = qr_recall(pairs, idx, transform=lambda x: x @ rot.t(), ks=(1,),
                    applicability=lambda x: scores)
    split = applicability_split(res, scores, threshold=0.5)
    assert split["share_applicable"] == 0.5
    assert split["recall_after@1_applicable"] == 1.0


def _trainer_with_structure(n_rels=2, seed=3):
    """A graph where relation 0 *is* a rotation of the node embeddings."""
    rng = random.Random(seed)
    n = 40
    base = torch.randn(n, 16)
    rot = torch.zeros(16, 16)
    rot[0, 1], rot[1, 0] = 1.0, -1.0     # rotation in one plane
    rotated = base @ rot.t()
    node2emb = {i: base[i] + 0.05 * torch.randn(16) for i in range(n)}
    rel2idx = {"rot": 0, "noise": 1}
    edges = [(i, (i + 1) % n, "rot") for i in range(n)]
    edges += [(i, rng.randrange(n), "noise") for i in range(n)]
    cfg = WiazkaConfig(op_dim=16, num_generators=4, gat_num_layers=1,
                       gat_heads=1, batch_size=32, epochs=1, device="cpu")
    return WiazkaTrainer(node2emb, rel2idx, edges, cfg), rotated


def test_qr_eval_reports_both_ends():
    trainer, _ = _trainer_with_structure()
    out = trainer.qr_eval(["rot"], ks=(1, 5), max_queries=32)
    assert "rot" in out
    row = out["rot"]
    assert row["n_queries"] > 0
    for key in ("recall_before@1", "recall_after@1", "delta_recall@1"):
        assert key in row


def test_qr_eval_detects_a_learned_rotation():
    """Feed the *oracle* rotation as the transform: recall must jump."""
    trainer, _ = _trainer_with_structure()
    index = trainer.node_emb
    n = index.shape[0]
    rot = torch.zeros(index.shape[1], index.shape[1])
    rot[0, 1], rot[1, 0] = 1.0, -1.0
    sim = _normalize(index @ rot.t()) @ _normalize(index).t()
    pairs = [(i, int(sim[i].argmax())) for i in range(n)]
    res = qr_recall(pairs, index, transform=lambda x: x @ rot.t(), ks=(1, 2),
                    relation="rot", chain=(0,))
    assert res.after[1] == 1.0
    assert res.after[1] > res.before[1]
