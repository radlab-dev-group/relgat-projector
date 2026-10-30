"""Sentence-level F4 supervision: it must fire, and it must carry gradient.

The reason F4 never improved anything is measured, not assumed: `inverse()` is
the exact analytic inverse of `apply()` at the same base point, so the cycle
loss is an identity and logs 0.0000 through an entire run.  `L_shadow_sent`
replaces that dead end with real supervision -- a graph edge whose endpoints
own usage examples trains the transport on the two sentences.  These tests pin
the properties that would have caught the original defect.
"""

import random

import numpy as np
import pytest
import torch

from relgat_projector.trainer.wiazka_trainer import WiazkaConfig, WiazkaTrainer

DIM = 16


def _toy_corpus(seed=0, n_nodes=40, n_rels=3, per_unit=3):
    """A graph whose every node owns a few "sentences" near its own vector."""
    rng = random.Random(seed)
    node2emb = {i: torch.randn(DIM) for i in range(n_nodes)}
    rel2idx = {f"r{i}": i for i in range(n_rels)}
    edges = []
    for i in range(n_nodes):
        for _ in range(3):
            j = rng.randrange(n_nodes)
            if j != i:
                edges.append((i, j, f"r{rng.randrange(n_rels)}"))

    texts, lus = [], []
    for node, base in node2emb.items():
        for _ in range(per_unit):
            jitter = torch.randn(DIM) * 0.05
            texts.append((base + jitter).numpy().astype(np.float32))
            lus.append(node)
    return (node2emb, rel2idx, edges,
            np.stack(texts), np.asarray(lus, dtype=np.int64))


def _trainer(**over):
    node2emb, rel2idx, edges, sents, lus = _toy_corpus()
    cfg = WiazkaConfig(
        op_dim=32, num_generators=8, gat_num_layers=1, gat_heads=1,
        batch_size=16, epochs=1, device="cpu",
        shadow_sent_weight=1.0, shadow_sent_batch=8, **over)
    return WiazkaTrainer(node2emb, rel2idx, edges, cfg,
                         shadow_sentences=sents, shadow_sentence_lus=lus)


def test_sentence_index_maps_sentences_to_their_own_node():
    trainer = _trainer()
    off, flat, lu = (trainer._lu_sent_offsets, trainer._lu_sent_flat,
                     trainer._sent_lu)
    # the CSR invariant, checked on every node segment: a cache position
    # stored under node n must be a sentence owned by n
    for node in range(trainer.num_nodes):
        seg = flat[off[node]:off[node + 1]]
        if seg.numel():
            assert bool((lu[seg] == node).all()), f"segment {node} not pure"

    # and a sentence the sampler draws for node `a` is one of node `a`'s
    s_src, s_dst, rel, src, dst = trainer._sample_shadow_pairs(16)
    assert s_src.shape == s_dst.shape
    for node, vec in zip(src.cpu(), s_src):
        seg = flat[off[node]:off[node + 1]]
        assert vec in trainer._sent_cache[seg], f"drawn {vec} not owned by {node}"
    for node, vec in zip(dst.cpu(), s_dst):
        seg = flat[off[node]:off[node + 1]]
        assert vec in trainer._sent_cache[seg], f"drawn {vec} not owned by {node}"


def test_pair_mask_never_offers_the_subject_as_an_answer():
    trainer = _trainer()
    s_src, s_dst, rel, src, dst = trainer._sample_shadow_pairs(12)
    mask = trainer._shadow_pair_mask(src, rel, dst)
    assert mask.shape == (12, 12)
    # a sentence of `a` is never the answer to `a -r->` (the Słowosieć has no
    # self-loops, so the subject is never among its own answers); every cell
    # where the subject of one edge meets the target of another must be masked
    same = (src.unsqueeze(1) == dst.unsqueeze(0))
    assert not bool((same & ~mask).any()), "a subject leaked in as its own answer"
    # and the positive (the target of its own row) is never masked away
    assert not bool(mask.diagonal().any())


def test_shadow_sent_term_is_nonzero_and_differentiable():
    """The whole point: this term has a gradient, the cycle term did not."""
    trainer = _trainer()
    trainer.stage_manager.current = len(trainer.stage_manager.order) - 1
    trainer._sync_stage_state()
    assert trainer.stage_manager.is_active("L_shadow")

    edges = trainer._sample_edges(16).to(trainer.cfg.device)
    paths = trainer._sample_paths(8)
    quads = trainer._sample_quadruples(4)
    refined = trainer.model.refine()
    loss, log = trainer._step(edges, paths, quads, refined)

    assert "L_shadow_sent" in log, log.keys()
    assert log["L_shadow_sent"] > 0.0
    assert torch.isfinite(loss)

    # the operator must receive gradient from the term alone: the cycle loss
    # could never give it one
    trainer.optimizer.zero_grad(set_to_none=True)
    pairs = trainer._sample_shadow_pairs(16)
    s_src, s_dst, rel_s, a_s, b_s = pairs
    query = trainer.model.production_query(s_src, rel_s)
    term = trainer.shadow_sent_loss(
        query, s_dst, torch.arange(s_dst.shape[0], device=s_dst.device),
        mask=trainer._shadow_pair_mask(a_s, rel_s, b_s))
    term.backward()
    theta = trainer.operator.theta
    assert theta.grad is not None, "the transport got no gradient from sentences"
    assert float(theta.grad.abs().sum()) > 0.0


def test_cycle_term_is_the_identity_the_fix_replaced():
    """Document the defect: T_r^-1(T_r z) == z, so the old F4 could not learn."""
    trainer = _trainer()
    z = trainer.model.base_to_op(trainer._sample_shadow(8))
    rel = torch.arange(8, device=z.device) % trainer.num_rel
    back = trainer.model.inverse(trainer.model.apply(z, rel), rel, base_point=z)
    # exact in fp32 only to the last few digits: the point is that it is a
    # round-off identity (~1e-8), not a measurable quantity (~0.01), and that
    # it carries no usable signal no matter how the operator changes
    assert float(trainer.shadow_loss.cycle(z, back)) == pytest.approx(0.0, abs=1e-6)


def test_weight_zero_disables_the_term():
    node2emb, rel2idx, edges, sents, lus = _toy_corpus()
    cfg = WiazkaConfig(op_dim=32, num_generators=8, gat_num_layers=1,
                       gat_heads=1, batch_size=16, epochs=1, device="cpu",
                       shadow_sent_weight=0.0)
    trainer = WiazkaTrainer(node2emb, rel2idx, edges, cfg,
                            shadow_sentences=sents, shadow_sentence_lus=lus)
    trainer.stage_manager.current = len(trainer.stage_manager.order) - 1
    trainer._sync_stage_state()
    edges_b = trainer._sample_edges(16).to(trainer.cfg.device)
    _, log = trainer._step(edges_b, None, None, trainer.model.refine())
    assert "L_shadow_sent" not in log


def test_sentences_of_a_foreign_lu_are_dropped_not_kept():
    """A sentence nobody can be asked about must not become a permanent negative."""
    node2emb, rel2idx, edges, sents, lus = _toy_corpus()
    extra = np.random.default_rng(0).standard_normal((5, DIM)).astype(np.float32)
    sents = np.concatenate([sents, extra])
    # LU ids that are not nodes of this graph
    lus = np.concatenate([lus, np.array([9001, 9002, 9003, 9004, 9005])])
    cfg = WiazkaConfig(op_dim=32, num_generators=8, gat_num_layers=1,
                       gat_heads=1, batch_size=16, epochs=1, device="cpu",
                       shadow_sent_weight=1.0)
    trainer = WiazkaTrainer(node2emb, rel2idx, edges, cfg,
                            shadow_sentences=sents, shadow_sentence_lus=lus)
    assert int(trainer._sent_cache.shape[0]) == len(lus) - 5
