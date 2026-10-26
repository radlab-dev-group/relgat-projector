"""Regression tests for the 2026-10 training-quality fixes.

Every test here pins a failure that was measured on the real Slowosiec run:
an unnormalised GAT aggregation, a projection that stopped being orthonormal
under AdamW (export -> QR-Recall 0.0), a probe dominated by one relation, and
metrics that compared two different indexes.
"""

import random

import torch

from relgat_projector.core.metrics.ranking import (
    edge_probe,
    stratified_edge_probe,
)
from relgat_projector.core.model.wiazka import WiazkaModel
from relgat_projector.core.operator import build_operator
from relgat_projector.trainer.stages import StageManager
from relgat_projector.inference import RelationFlow
from relgat_projector.trainer.wiazka_trainer import (
    ENSEMBLE_WEIGHT_GRID,
    PRIOR_STRENGTH_GRID,
    WiazkaConfig,
    WiazkaTrainer,
    principal_projection,
)


def _toy_graph(seed=0, n_nodes=40, n_rels=3, base_dim=16):
    rng = random.Random(seed)
    node2emb = {i: torch.randn(base_dim) for i in range(n_nodes)}
    rel2idx = {f"r{i}": i for i in range(n_rels)}
    edges = []
    for i in range(n_nodes):
        for k in range(3):
            j = rng.randrange(n_nodes)
            if j != i:
                edges.append((i, j, f"r{k % n_rels}"))
    return node2emb, rel2idx, edges


def _trainer(**kw):
    node2emb, rel2idx, edges = _toy_graph()
    cfg = WiazkaConfig(
        op_dim=8, num_generators=4, gat_num_layers=1, gat_heads=2,
        batch_size=16, epochs=1, device="cpu", **kw
    )
    return WiazkaTrainer(node2emb, rel2idx, edges, cfg)


# ------------------------------------------------------------------ chart
def test_principal_projection_is_orthonormal():
    X = torch.randn(200, 64)
    W = principal_projection(X, 16, seed=3)
    assert W.shape == (16, 64)
    assert torch.allclose(W @ W.t(), torch.eye(16), atol=1e-5)


def test_projection_stays_orthonormal_under_training():
    """AdamW must not be able to warp the base<->operator chart.

    The projection weights are ~1/sqrt(op_dim) in magnitude while an Adam step
    is ~lr, so a trainable projection lost orthonormality (||W W^T - I|| = 0.22)
    within a few hundred steps and every exported query left the retriever
    space: QR-Recall measured exactly 0.0 for all relations.
    """
    trainer = _trainer()
    W = trainer.model.base_to_op.weight
    assert trainer.model.frozen_projection is True
    assert not W.requires_grad
    opt_ids = {id(p) for g in trainer.optimizer.param_groups for p in g["params"]}
    assert id(W) not in opt_ids

    trainer.train(epochs=1, log_every=10_000)
    W_after = trainer.model.base_to_op.weight
    assert torch.allclose(W_after @ W_after.t(), torch.eye(W.shape[0]), atol=1e-5)


def test_export_is_the_adjoint_of_the_chart():
    trainer = _trainer()
    model = trainer.model
    x = torch.randn(5, model._op_dim)
    exported = model.to_retriever_space(x)
    assert torch.allclose(exported, x @ model.base_to_op.weight, atol=1e-6)
    # round trip through the chart is the orthogonal projection: idempotent
    back = model.base_to_op(exported)
    assert torch.allclose(model.to_retriever_space(back) @ exported.t(),
                          exported @ exported.t(), atol=1e-4)


def test_export_residual_can_learn():
    """A double zero (gate 0 * zero projection) has no gradient at all."""
    trainer = _trainer()
    model = trainer.model
    assert float(model.export_gate.detach()) == 0.0
    assert float(model.export_proj.weight.std()) > 0.0
    out = model.to_retriever_space(torch.randn(4, model._op_dim))
    out.pow(2).mean().backward()
    assert model.export_gate.grad is not None


# ------------------------------------------------------------- aggregation
def _tiny_model(attention=True, base_dim=12, op_dim=6):
    node_emb = torch.randn(7, base_dim)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [5, 5, 5, 6, 6]])
    edge_type = torch.tensor([0, 1, 2, 0, 1])
    op = build_operator("givens", dim=op_dim, num_rel=3, num_generators=3)
    return WiazkaModel(node_emb, edge_index, edge_type, 3, op,
                       gat_num_layers=1, gat_heads=2, attention=attention)


def _aggregated(model, attention):
    """Run one GAT layer with the gate open, dropout off, attention forced."""
    layer = model.layers[0]
    layer.attention = attention
    model.eval()
    with torch.no_grad():
        layer.gate.fill_(1.0)
        base = model.refine()
        dup_index = torch.cat([model.edge_index, model.edge_index], dim=1)
        dup_type = torch.cat([model.edge_type, model.edge_type])
        once = layer(model.node_emb_fixed, base, model.edge_index,
                     model.edge_type)
        twice = layer(model.node_emb_fixed, base, dup_index, dup_type)
    return once, twice


def test_attention_aggregation_is_invariant_to_edge_multiplicity():
    """A normalised aggregation cannot grow when an edge is listed twice."""
    once, twice = _aggregated(_tiny_model(attention=True), attention=True)
    assert torch.allclose(once, twice, atol=1e-5)


def test_mean_aggregation_is_invariant_but_relation_blind():
    """Degree normalisation is there; per-relation weighting is not."""
    once, twice = _aggregated(_tiny_model(attention=False), attention=False)
    assert torch.allclose(once, twice, atol=1e-5)
    model = _tiny_model(attention=False)
    assert not hasattr(model.layers[0], "att_rel") or model.layers[0].att_rel is None


def test_relation_attention_reweights_messages_of_one_neighbourhood():
    """Softmax over the incoming edges must renormalise, not rescale."""
    model = _tiny_model(attention=True)
    layer = model.layers[0]
    model.eval()
    with torch.no_grad():
        layer.gate.fill_(1.0)
        layer.att_q[0].weight.zero_()
        layer.att_q[1].weight.zero_()
        base = model.refine()
        msgs = layer._messages(base, model.edge_index, model.edge_type)
        for h in range(layer.att_rel.shape[0]):
            layer.att_rel[h, 0] = -30.0     # silence relation 0
        out = layer(model.node_emb_fixed, base, model.edge_index,
                    model.edge_type)
        # node 5 is fed by relations 0, 1, 2 (edges 0,1,2); with relation 0
        # silenced its update is the mean of the two remaining messages
        # logits are bounded by tanh, so relation 0 keeps a weight of
        # exp(-4) rather than exactly zero -- the expectation accounts for it
        bound = layer.LOGIT_BOUND
        import math

        w0 = math.exp(bound * math.tanh(-30.0 / bound))
        alpha0 = w0 / (w0 + 2.0)
        per_head = [
            alpha0 * msgs[h][0]
            + (1.0 - alpha0) * (msgs[h][1] + msgs[h][2]) / 2
            for h in range(layer.heads)
        ]
        expected_update = torch.nn.functional.gelu(
            torch.stack(per_head, dim=0).mean(dim=0)
        )
        got_update = (out[5] - base[5]) / float(torch.tanh(layer.gate))
        assert torch.allclose(got_update, expected_update, atol=1e-5)
        assert torch.isfinite(out).all()
        assert not torch.allclose(out[5], base[5], atol=1e-6)


# ------------------------------------------------------------------ probe
def test_stratified_probe_covers_rare_relations():
    edges = [(i, i + 1, r) for r, n in ((0, 10000), (1, 4), (2, 1)) for i in range(n)]
    probe = stratified_edge_probe(edges, 64, seed=1)
    seen = {r for _, _, r in probe}
    assert seen == {0, 1, 2}
    assert sum(1 for _, _, r in probe if r == 1) == 4
    assert stratified_edge_probe(edges, 64, seed=1) == probe


def test_evaluate_reports_every_protocol():
    trainer = _trainer(probe_size=32)
    m = trainer.evaluate(n_probe=32)
    for key in ("eval/mrr", "eval/id_mrr", "eval/raw_mrr", "eval/prod_mrr",
                "eval/mrr_lift", "eval/mrr_prod_lift", "eval/mrr_macro",
                "eval/prod_mrr_traffic", "eval/raw_mrr_traffic",
                "eval/mrr_prod_lift_traffic", "eval/n_probe"):
        assert key in m, key
    assert abs(m["eval/mrr_lift"] - (m["eval/mrr"] - m["eval/id_mrr"])) < 1e-9
    assert abs(m["eval/mrr_prod_lift_traffic"]
               - (m["eval/prod_mrr_traffic"] - m["eval/raw_mrr_traffic"])) < 1e-9
    # an untrained operator may not score better than the raw baseline by
    # accident of the index it is compared against
    assert m["eval/prod_mrr"] > 0.0


def test_per_relation_report_is_plain_floats():
    trainer = _trainer(probe_size=32)
    rows = trainer.per_relation_report(n_per_rel=4)
    assert rows
    for row in rows:
        for key, value in row.items():
            if key == "relation":
                assert isinstance(value, str)
            elif key == "n":
                assert isinstance(value, int)
            else:
                assert isinstance(value, float), (key, type(value))
    assert rows == sorted(rows, key=lambda r: -r["lift_over_raw"])


def test_sampled_path_batches_are_never_padded():
    """Every returned chain must be a real walk of the full row length.

    Zero-padding a ``[B, k]`` relation batch silently adds relation id 0 to the
    shorter chains, so ``L_path`` optimises the wrong target for those rows.
    """
    trainer = _trainer(probe_size=32)
    adjacency = trainer.path_sampler.adjacency
    seen_depths = set()
    for _ in range(8):
        batch = trainer._sample_paths(32)
        if batch is None:
            continue
        src, rels, dst = batch
        assert rels.dim() == 2
        k = rels.shape[1]
        seen_depths.add(k)
        for i in range(src.shape[0]):
            node = int(src[i])
            for step in range(k):          # the row must walk end to end
                wanted = int(rels[i, step])
                nxt = [d for d, r in adjacency[node] if r == wanted]
                assert nxt, f"row {i} step {step} is not a real edge"
                node = nxt[0]
            assert node == int(dst[i]), f"row {i} does not end at its target"
    assert seen_depths, "no paths sampled at all"


def test_pce_pool_has_a_k1_point():
    """The F3 gate is defined on pce_k2 / pce_k1, so k=1 has to exist.

    ``PathSampler`` walks k >= 2 by construction, so the k=1 pool used to be
    empty, ``eval/pce_ratio_k2`` was never emitted, and the stage manager
    treated a missing metric as "not demonstrated" -- F3 could only be left
    through the ``max_steps`` safety valve.
    """
    trainer = _trainer(probe_size=32)
    pool = trainer._pce_pool(1, size=16)
    assert pool is not None
    src, rels, dst = pool
    assert rels.shape[1] == 1 and src.shape[0] == dst.shape[0] == 16
    metrics = trainer.evaluate(n_probe=32)
    assert "eval/pce_k1" in metrics and "eval/pce_ratio_k2" in metrics
    assert metrics["eval/pce_ratio_k2"] > 0.0


def test_f2_gate_is_the_transport_not_the_index():
    manager = StageManager()
    manager.current = 1
    manager.steps_in_stage = 10_000
    assert "eval/mrr_lift" in manager.gates["F2_relations"]
    # identity transport (lift 0) must NOT open the gate on a strict threshold
    manager.steps_in_stage = 700
    assert manager.should_stop({"eval/mrr_lift": -0.01}) is False


def test_legacy_checkpoint_spec_keeps_legacy_architecture():
    """Specs written before the attention/tie options must still load."""
    model = _tiny_model(attention=False)
    model = WiazkaModel(
        model.node_emb_fixed, model.edge_index, model.edge_type, 3,
        build_operator("givens", dim=6, num_rel=3, num_generators=3),
        gat_num_layers=1, gat_heads=2, attention=False, tie_projections=False,
    )
    spec = model.spec()
    for key in ("attention", "tie_projections", "frozen_projection"):
        spec.pop(key, None)
    rebuilt = WiazkaModel.from_spec(
        spec,
        node_emb=model.node_emb_fixed,
        edge_index=model.edge_index,
        edge_type=model.edge_type,
    )
    rebuilt.load_state_dict(model.state_dict())
    assert rebuilt.layers[0].attention is False
    assert not hasattr(rebuilt.layers[0], "att_q") or rebuilt.layers[0].att_q is None


# ------------------------------------------------------- relation balancing
def _skewed_trainer(power, n_per_rel=(900, 60, 40), n_nodes=80):
    """Trainer on a graph where one relation holds ~90 % of the edges.

    Mirrors the Slowosiec skew (``synonimia`` alone is 49.7 % of the edges, 14
    relations have fewer than 100).  Triplets are unique -- the loader drops
    duplicates -- so the pairs are drawn without replacement.
    """
    node2emb = {i: torch.randn(8) for i in range(n_nodes)}
    names = ["big", "mid", "small"]
    rel2idx = {name: i for i, name in enumerate(names)}
    all_pairs = [(a, b) for a in range(n_nodes) for b in range(n_nodes) if a != b]
    edges = []
    for rel_id, count in enumerate(n_per_rel):
        for a, b in random.sample(all_pairs, count):
            edges.append((a, b, names[rel_id]))
    cfg = WiazkaConfig(
        op_dim=8, num_generators=4, gat_num_layers=1, gat_heads=2,
        batch_size=16, epochs=1, device="cpu", balance_power=power,
    )
    return WiazkaTrainer(node2emb, rel2idx, edges, cfg)


def test_relation_sampling_is_tempered_not_uniform():
    """Sampling must land between the two measured pathologies.

    Slowosiec skew: ``synonimia`` carries 178 772 of 359 818 edges (49.7 %) and
    14 relations have fewer than 100.  Uniform-over-relations gave the dominant
    relation 1.8 % of every batch and cost it 0.24 MRR, while the 543 edges of
    the tiny relations soaked up 25 % of all samples.  Tempering with
    ``p ~ count ** 0.5`` must sit strictly between uniform and raw for both the
    dominant and the rarest relation.
    """
    import collections
    import math

    # ONE trainer: constructing it reseeds torch, so a fresh trainer per draw
    # would replay the same 256 samples over and over
    trainer = _skewed_trainer(0.5)
    counts_seen = collections.Counter()
    for _ in range(150):
        for r in trainer._sample_edges(256)[:, 2].tolist():
            counts_seen[r] += 1
    total = sum(counts_seen.values())
    assert total == 150 * 256
    share = {r: counts_seen[r] / total for r in (0, 1, 2)}
    raw = {r: n / 1000.0 for r, n in enumerate((900, 60, 40))}
    uniform = {r: 1.0 / 3.0 for r in (0, 1, 2)}
    tempered = {
        r: math.sqrt(n) / sum(math.sqrt(x) for x in (900, 60, 40))
        for r, n in enumerate((900, 60, 40))
    }

    for r in (0, 1, 2):
        assert abs(share[r] - tempered[r]) < 0.04, (r, share, tempered[r])
        if raw[r] > uniform[r]:
            assert uniform[r] < share[r] < raw[r], (r, share)
        else:
            assert raw[r] < share[r] < uniform[r], (r, share)
    # tempering moves in the required direction for *both* tails; on the real
    # graph (56 relations) power 0.5 gives synonimia 16.3 % of the batch
    # instead of 1.8 % (uniform) or 49.7 % (raw), and the 14 relations with
    # <100 edges 3.1 % instead of 25 % (uniform) or 0.15 % (raw)
    assert share[0] > uniform[0] and share[2] < uniform[2]


def test_relation_sampling_power_one_reproduces_the_raw_distribution():
    import collections

    trainer = _skewed_trainer(1.0)
    counts = collections.Counter()
    for _ in range(150):
        for r in trainer._sample_edges(256)[:, 2].tolist():
            counts[r] += 1
    total = sum(counts.values())
    assert abs(counts[0] / total - 0.9) < 0.03, counts


def test_relation_sampling_never_draws_an_absent_relation():
    node2emb = {i: torch.randn(8) for i in range(20)}
    rel2idx = {"a": 0, "b": 1, "c": 2}
    edges = [(i, (i + 1) % 20, "a") for i in range(20)]
    cfg = WiazkaConfig(op_dim=8, num_generators=4, gat_num_layers=1,
                       batch_size=8, epochs=1, device="cpu", balance_power=0.75)
    trainer = WiazkaTrainer(node2emb, rel2idx, edges, cfg)
    for _ in range(20):
        batch = trainer._sample_edges(64)
        assert set(batch[:, 2].tolist()) == {0}


# ------------------------------------------------------- curvature margins
def test_anti_commutation_margin_scales_with_vector_length():
    """A fixed margin of 0.25 is unreachable, so the term is dead.

    ``gap_l2 = ||T_r T_s x - T_s T_r x|| / sqrt(op_dim)`` is bounded by
    ``2 ||x|| / sqrt(op_dim)``.  In the 256-d operator chart of the Slowosiec
    base embeddings the mean vector norm is 0.385, so the *largest possible*
    gap is 0.048 and the default margin 0.25 can never be exceeded: the
    anti-commutation half of the curvature loss contributed exactly zero for a
    whole training run.  Scaling the margin to the vector norm fixes it.
    """
    from relgat_projector.core.loss.curvature import CurvatureLoss

    torch.manual_seed(0)
    n_rows, dim = 64, 256
    x = torch.randn(n_rows, dim)
    x = x / x.norm(dim=-1, keepdim=True) * 0.385      # measured chart norm
    other = torch.randn(n_rows, dim)
    other = other / other.norm(dim=-1, keepdim=True) * 0.385

    r = torch.zeros(n_rows, dtype=torch.long)
    s = torch.ones(n_rows, dtype=torch.long)
    fixed = CurvatureLoss(4, comm_weight=0.0, anti_weight=1.0, margin=0.25)
    auto = CurvatureLoss(4, comm_weight=0.0, anti_weight=1.0, margin_auto=True)

    # even perfectly anti-commuting routes cannot violate the fixed margin
    dead = fixed(x, x, r, s, other, -other)
    assert float(dead) == 0.0, float(dead)

    alive = auto(x, x, r, s, other, -other)
    assert torch.isfinite(alive)
    assert float(alive) > 0.0
    # ... and the penalty is monotone in how far apart the routes are
    near = auto(x, x, r, s, other, other * 0.9)
    assert float(near) == 0.0 < float(alive)


# ------------------------------------------------------------ L_pull (F4)
def test_pull_loss_trains_the_operator_end_to_end():
    """The production term must move the transport, not only the export.

    ``L_pull`` is the only term that scores the exported query against the
    *base* index, i.e. the index a vector database actually holds.  It has to
    reach both the operator angles and the export correction, otherwise the
    deployed query-rewriting path is untrained (measured: ``eval/mrr`` 0.19
    against ``eval/prod_mrr`` 0.11).
    """
    trainer = _trainer()
    from relgat_projector.trainer.stages import StageManager

    trainer.stage_manager = StageManager()
    while trainer.stage_manager.stage().name != "F4_shadows":
        trainer.stage_manager.advance()
        trainer.stage_manager._sync_stage_state = None
        WiazkaTrainer._sync_stage_state(trainer)

    edges = trainer._sample_edges(32).to(trainer.cfg.device)
    refined = trainer.model.refine()
    loss, log = trainer._step(edges, None, None, refined)
    assert "L_pull" in log
    assert torch.isfinite(loss), log
    loss.backward()
    assert trainer.operator.theta.grad is not None
    assert float(trainer.operator.theta.grad.abs().sum()) > 0.0
    assert trainer.model.export_proj.weight.grad is not None


# --------------------------------------------- structure-preserving refinement
def test_knn_pairs_are_distinct_nodes_and_neighbours():
    trainer = _trainer(knn_weight=1.0, knn_k=4, knn_anchors=20)
    src, dst, cos_base = trainer._base_neighbour_pairs()
    assert src.numel() > 0
    assert not bool((src == dst).any()), "a node must not be its own neighbour"
    # neighbours are the *nearest* nodes, so their cosine must be well above
    # the cosine of two random nodes in the same space (≈0)
    assert float(cos_base.mean()) > 0.3, float(cos_base.mean())
    assert int(src.max()) < trainer.num_nodes and int(dst.max()) < trainer.num_nodes


def test_knn_loss_is_zero_for_the_identity_refinement_and_bites_otherwise():
    """The hinge must be silent when the neighbourhood survives.

    ``L_anchor`` cannot express this: a refinement may rotate every vector
    away from its base position (anchor violated) and still preserve the
    ranking the retriever uses, and it may keep every vector close to its base
    position while shuffling its neighbours (anchor satisfied, index ruined).
    """
    trainer = _trainer(knn_weight=1.0, knn_k=4, knn_anchors=24, knn_slack=0.02)
    base = trainer.model.node_emb_fixed
    chart = trainer.model.base_to_op(base)

    # identity refinement: cos is exactly the base cos -> no penalty
    assert float(trainer._knn_loss(chart)) < 1e-5

    # an isometry of the chart preserves every cosine -> still no penalty
    torch.manual_seed(0)
    q, _ = torch.linalg.qr(torch.randn(chart.shape[1], chart.shape[1]))
    assert float(trainer._knn_loss(chart @ q)) < 1e-4

    # a refinement that decorrelates every vector from its neighbours pays
    shuffled = torch.randn_like(chart)
    penalty = float(trainer._knn_loss(shuffled))
    assert penalty > 0.05, penalty

    # and the penalty reaches the refinement, not only the sampled rows
    refined = trainer.model.refine()
    refined.retain_grad()
    loss = trainer._knn_loss(refined)
    loss.backward()
    assert refined.grad is not None


# ------------------------------------------------- F4 connection (context)
def _f4_trainer(**kw):
    trainer = _trainer(**kw)
    trainer.stage_manager = StageManager()
    while trainer.stage_manager.stage().name != "F4_shadows":
        trainer.stage_manager.advance()
        trainer._sync_stage_state()
    return trainer


def test_message_passing_leaves_the_operator_connection_untouched():
    """The GAT may not switch the F4 connection off for the rest of the step.

    Message passing runs the operator without context on purpose (a
    row-dependent transport would need one matrix per edge), but it has to
    give the operator back as it found it.  Restoring the *layer's* setting
    instead of the operator's previous one turned the connection off after the
    first GAT layer -- and because ``refine()`` runs first, every per-sample
    loss then saw a context-free operator and the hyper-network got zero
    gradient for an entire run.
    """
    trainer = _f4_trainer()
    operator = trainer.operator
    assert operator.use_context and operator.ctx_P is not None

    edges = trainer._sample_edges(32)
    trainer.model.refine()          # <- this used to kill the connection
    assert operator.use_context is True

    x = torch.randn(16, operator.dim)
    rel = torch.randint(0, trainer.num_rel, (16,))
    assert torch.isfinite(operator.apply(x, rel)).all()
    modulation = operator.eps * operator.ctx_P(torch.relu(operator.ctx_W(x)))
    assert float(modulation.abs().sum()) > 0.0, "ctx_P must not start at zero"


def test_connection_parameters_receive_gradient_in_f4():
    """ctx_W / ctx_P / log_eps must be trainable, not merely registered."""
    trainer = _f4_trainer()
    operator = trainer.operator
    edges = trainer._sample_edges(48).to(trainer.cfg.device)
    refined = trainer.model.refine()
    loss, log = trainer._step(edges, None, None, refined)
    assert torch.isfinite(loss), log
    loss.backward()
    for name, param in (("ctx_W", operator.ctx_W.weight),
                        ("ctx_P", operator.ctx_P.weight),
                        ("log_eps", operator.log_eps)):
        assert param.grad is not None, f"{name} received no gradient"
        assert float(param.grad.abs().sum()) > 0.0, f"{name} gradient is zero"
    known = {id(p) for g in trainer.optimizer.param_groups for p in g["params"]}
    for name, param in (("ctx_W", operator.ctx_W.weight),
                        ("ctx_P", operator.ctx_P.weight),
                        ("log_eps", operator.log_eps)):
        assert id(param) in known, f"{name} is not in the optimizer"


def test_cycle_is_exact_with_the_connection_on():
    """``T_r^{-1}(T_r(x))`` must invert the *transport that was used*.

    The forward map at ``x`` uses the modulated coefficients, so the inverse
    has to as well: with a context-aware inverse the cycle error is exactly
    zero and the connection may grow; with the unmodulated inverse the cycle
    loss measured the size of the context and trained it back to zero.
    """
    trainer = _f4_trainer()
    operator = trainer.operator
    torch.nn.init.normal_(operator.ctx_P.weight, std=0.5)
    with torch.no_grad():
        operator.log_eps.fill_(0.0)      # eps = 1: context fully on

    x = torch.randn(24, operator.dim)
    rel = torch.randint(0, trainer.num_rel, (24,))
    forward = operator.apply(x, rel)
    round_trip = operator.inverse(forward, rel, base_point=x)
    assert float((round_trip - x).pow(2).mean().sqrt()) < 1e-4

    naive = operator.inverse(forward, rel)
    assert float((naive - x).pow(2).mean().sqrt()) > 1e-2, (
        "the unmodulated inverse must differ, otherwise this test proves nothing"
    )


# ------------------------------------- production query (per-relation dial)
def test_production_query_can_turn_rewriting_completely_off():
    """lam = 0 must return the incoming query unchanged.

    The transport is a large win on ~40 Slowosiec relations and a clear loss on
    the ones the base embeddings already encode.  A single global "always
    rewrite" answer is wrong for one side; with a learned per-relation strength
    the model can stop degrading ``synonimia`` (QR-Recall@1 fell 0.50 -> 0.38
    when rewriting was unconditional).
    """
    node2emb, rel2idx, edges = _toy_graph()
    from relgat_projector.core.model.wiazka import WiazkaModel

    trainer = _trainer()
    model = trainer.model
    x = trainer.model.node_emb_fixed[:12]
    rel = torch.arange(12) % trainer.num_rel

    with torch.no_grad():
        model.transport_dial.fill_(-20.0)
        off = model.production_query(x, rel)
    assert torch.allclose(off, x, atol=1e-5)

    with torch.no_grad():
        model.transport_dial.fill_(20.0)
        on = model.production_query(x, rel)
    assert not torch.allclose(on, x, atol=1e-3)
    assert float(model.transport_strength(rel).min()) > 0.99


def test_transport_dial_is_per_relation_and_trainable():
    trainer = _trainer()
    model = trainer.model
    n = trainer.num_rel
    assert tuple(model.transport_dial.shape) == (n,)

    trainer.stage_manager = StageManager()
    while trainer.stage_manager.stage().name != "F4_shadows":
        trainer.stage_manager.advance()
        trainer._sync_stage_state()

    edges = trainer._sample_edges(48).to(trainer.cfg.device)
    refined = trainer.model.refine()
    loss, log = trainer._step(edges, None, None, refined)
    assert "L_pull" in log
    loss.backward()
    grad = model.transport_dial.grad
    assert grad is not None and float(grad.abs().sum()) > 0.0

    known = {id(p) for g in trainer.optimizer.param_groups for p in g["params"]}
    assert id(model.transport_dial) in known, "the dial is not optimised"


def test_checkpoint_without_the_dial_still_loads_and_strictness_survives(tmp_path):
    """A new parameter may not make old checkpoints unreadable ..."""
    from relgat_projector.core.model.wiazka import WiazkaModel
    from relgat_projector.handlers.wiazka_checkpoint import (
        load_checkpoint, save_checkpoint,
    )

    trainer = _trainer()
    with torch.no_grad():
        trainer.model.transport_dial.fill_(0.7)
    trainer.save(str(tmp_path / "ck"))
    reloaded, _, _, _ = load_checkpoint(str(tmp_path / "ck"), "cpu")
    assert torch.allclose(reloaded.transport_dial,
                          torch.full_like(reloaded.transport_dial, 0.7), atol=1e-6)

    # ... but a truncated state dict must still be an error
    legacy = {k: v for k, v in trainer.model.state_dict().items()
              if not k.startswith("export_proj")}
    bundle_path = tmp_path / "ck" / "model.pt"
    saved = torch.load(bundle_path, weights_only=True)
    torch.save({**saved, "model": legacy}, bundle_path)
    try:
        load_checkpoint(str(tmp_path / "ck"), "cpu")
    except RuntimeError as exc:
        assert "export_proj" in str(exc)
    else:
        raise AssertionError("a missing module must not load silently")


def test_inference_api_serves_the_scored_query(tmp_path):
    """`RelationFlow.production_query` must be the function `prod_mrr` measures."""
    from relgat_projector.inference import RelationFlow

    trainer = _trainer()
    trainer.save(str(tmp_path / "flow"))
    flow = RelationFlow.load(str(tmp_path / "flow"), device="cpu")

    base = trainer.model.node_emb_fixed[:6]
    name = next(iter(trainer.rel2idx))
    served = flow.production_query(base, name)
    assert served.shape == base.shape

    rid = trainer.rel2idx[name]
    expected = trainer.model.production_query(
        base, torch.full((6,), rid, dtype=torch.long)
    )
    assert torch.allclose(served.cpu(), expected, atol=1e-6)
    assert 0.0 <= flow.transport_strength(name) <= 1.0


def _toy_model(num_rel=4, dim=16, ctx=False, n_nodes=60, base_dim=8):
    """Small operator + model with random graph data, for scorer tests."""
    op = build_operator("givens", dim=dim, num_rel=num_rel, num_generators=4,
                        use_context=ctx)
    emb = torch.randn(n_nodes, base_dim)
    edge_index = torch.randint(0, n_nodes, (2, 200))
    edge_type = torch.randint(0, num_rel, (200,))
    return WiazkaModel(emb, edge_index, edge_type, num_rel, op,
                       gat_num_layers=1, gat_heads=1)


def test_ensemble_weight_controls_the_mix():
    torch.manual_seed(0)
    m = _toy_model(ctx=False)
    m.eval()
    xa, xb = torch.randn(7, 16), torch.randn(7, 16)
    head = m.relation_logits(xa, xb, mode="decoder")
    transport = m.relation_logits(xa, xb, mode="transport")
    zh = type(m)._row_zscore(head)
    zt = type(m)._row_zscore(transport)
    assert torch.allclose(m.relation_logits(xa, xb, mode="ensemble",
                                            ensemble_weight=0.0), zh, atol=1e-5)
    mixed = m.relation_logits(xa, xb, mode="ensemble", ensemble_weight=0.3)
    assert torch.allclose(mixed, zh + 0.3 * zt, atol=1e-5)
    # the model attribute is the default when no explicit weight is given
    m.relation_ensemble_weight = 0.3
    assert torch.allclose(m.relation_logits(xa, xb, mode="ensemble"), mixed,
                          atol=1e-6)


def test_fit_ensemble_weight_follows_the_evidence():
    """The mix has to follow whichever evidence actually carries the signal."""
    from relgat_projector.trainer.wiazka_trainer import (
        fit_ensemble_weight, macro_relation_accuracy)

    torch.manual_seed(3)
    rel = torch.randint(0, 4, (400,))
    noise_head = torch.randn(400, 4)

    # the head says nothing, the transport is right: blending it at all would
    # only hurt, so the weight has to go up
    perfect_transport = torch.zeros(400, 4)
    perfect_transport[torch.arange(400), rel] = 1.0
    assert fit_ensemble_weight(noise_head, perfect_transport, rel, 4) > 0.0

    # the reverse: noise transport must be dropped entirely, not averaged in
    perfect_head = torch.zeros(400, 4)
    perfect_head[torch.arange(400), rel] = 1.0
    assert fit_ensemble_weight(perfect_head, torch.randn(400, 4), rel, 4) == 0.0

    # complementary evidences (each right on a different half): only a mix of
    # the two can be right everywhere, and the fit has to find that mix
    other = torch.zeros(400, 4)
    other[torch.arange(400), (rel + 1) % 4] = 1.0
    half = torch.arange(400) < 200
    head_mix = torch.where(half.unsqueeze(-1), perfect_head, other)
    transport_mix = torch.where(half.unsqueeze(-1), other, perfect_transport)
    alpha = fit_ensemble_weight(head_mix, transport_mix, rel, 4)
    assert alpha > 0.0, "complementary evidence must be blended"
    zh = WiazkaModel._row_zscore(head_mix)
    zt = WiazkaModel._row_zscore(transport_mix)
    assert macro_relation_accuracy((zh + alpha * zt).argmax(-1), rel, 4) > \
        macro_relation_accuracy(zh.argmax(-1), rel, 4)


def test_fit_ensemble_weight_is_the_grid_argmax():
    from relgat_projector.trainer.wiazka_trainer import (
        ENSEMBLE_WEIGHT_GRID, fit_ensemble_weight, macro_relation_accuracy)

    torch.manual_seed(5)
    rel = torch.randint(0, 3, (300,))
    signal = torch.zeros(300, 3)
    signal[torch.arange(300), rel] = 1.0
    head = 0.6 * signal + 0.4 * torch.randn(300, 3)
    transport = 0.2 * signal + 0.8 * torch.randn(300, 3)
    alpha = fit_ensemble_weight(head, transport, rel, 3)
    zh = WiazkaModel._row_zscore(head)
    zt = WiazkaModel._row_zscore(transport)
    accs = {a: macro_relation_accuracy((zh + a * zt).argmax(-1), rel, 3)
            for a in ENSEMBLE_WEIGHT_GRID}
    assert accs[alpha] == max(accs.values())
    # ties resolve to the smaller weight
    assert alpha == min(a for a in ENSEMBLE_WEIGHT_GRID if accs[a] == max(accs.values()))


def test_trainer_fits_and_reports_the_ensemble_weight():
    torch.manual_seed(0)
    trainer = _trainer()
    refined = trainer.model.refine()
    stats = trainer._transport_prior_stats(refined)
    alpha = trainer.fit_relation_ensemble_weight(refined, stats)
    assert alpha in ENSEMBLE_WEIGHT_GRID
    assert trainer.model.relation_ensemble_weight == alpha
    # evaluate() fits the prior first and then the mix, and reports both
    metrics = trainer.evaluate(n_probe=48)
    assert metrics["eval/rel_ensemble_alpha"] == \
        trainer.model.relation_ensemble_weight
    assert metrics["eval/rel_prior_strength"] == \
        trainer.model.relation_prior_strength
    assert metrics["eval/rel_prior_strength"] in PRIOR_STRENGTH_GRID


def test_prior_strength_scales_the_frequency_correction():
    torch.manual_seed(0)
    m = _toy_model(ctx=False)
    m.eval()
    xa, xb = torch.randn(6, 16), torch.randn(6, 16)
    counts = torch.tensor([100, 10, 1, 1])
    log_prior = torch.log(counts.float() / counts.sum().float())
    head = torch.log_softmax(m.relation_logits(xa, xb, mode="decoder"), -1)

    m.relation_prior_strength = 0.0
    none = m.relation_logits(xa, xb, mode="prior", train_counts=counts)
    assert torch.allclose(none, head, atol=1e-5)

    m.relation_prior_strength = 1.0
    full = m.relation_logits(xa, xb, mode="prior", train_counts=counts)
    assert torch.allclose(full, head - log_prior, atol=1e-5)

    # a balanced head needs no correction at all, and the ensemble must follow
    # that decision rather than the correction
    m.relation_prior_strength = 0.0
    assert torch.allclose(
        m.relation_logits(xa, xb, mode="prior", train_counts=counts), head,
        atol=1e-5)


def test_relation_calibration_travels_with_the_checkpoint(tmp_path):
    node2emb = {i: torch.randn(8) for i in range(30)}
    rel2idx = {"r0": 0, "r1": 1}
    edges = [(i, (i + 3) % 30, f"r{i % 2}") for i in range(30)]
    cfg = WiazkaConfig(op_dim=16, num_generators=4, gat_num_layers=1,
                       gat_heads=1, batch_size=16, epochs=1, device="cpu")
    trainer = WiazkaTrainer(node2emb, rel2idx, edges, cfg)
    trainer.model.relation_ensemble_weight = 0.25
    trainer.model.relation_prior_strength = -0.5
    trainer.save(str(tmp_path / "ckpt"))
    conn = RelationFlow.load(str(tmp_path / "ckpt"), device="cpu")
    assert conn.model.relation_ensemble_weight == 0.25
    assert conn.model.relation_prior_strength == -0.5


def test_ensemble_weight_travels_with_the_checkpoint(tmp_path):
    node2emb = {i: torch.randn(8) for i in range(30)}
    rel2idx = {"r0": 0, "r1": 1}
    edges = [(i, (i + 3) % 30, f"r{i % 2}") for i in range(30)]
    cfg = WiazkaConfig(op_dim=16, num_generators=4, gat_num_layers=1,
                       gat_heads=1, batch_size=16, epochs=1, device="cpu")
    trainer = WiazkaTrainer(node2emb, rel2idx, edges, cfg)
    trainer.model.relation_ensemble_weight = 0.25
    trainer.save(str(tmp_path / "ckpt"))
    conn = RelationFlow.load(str(tmp_path / "ckpt"), device="cpu")
    assert conn.model.relation_ensemble_weight == 0.25


def test_connection_ablation_switch_is_honoured():
    """``enable_connection=False`` must keep F4 position-independent."""
    trainer = _trainer(enable_connection=False)
    trainer.stage_manager.current = len(trainer.stage_manager.order) - 1
    trainer._sync_stage_state()
    assert trainer.stage_manager.stage().enable_context
    assert not trainer.model.operator.use_context

    enabled = _trainer()
    enabled.stage_manager.current = len(enabled.stage_manager.order) - 1
    enabled._sync_stage_state()
    assert enabled.model.operator.use_context


def test_spread_sample_covers_the_whole_relation():
    from relgat_projector.trainer.wiazka_trainer import spread_sample

    rows = [(i, f"group{i % 10}") for i in range(1000)]
    sample = spread_sample(rows, 60)
    assert len(sample) == 60
    groups = {g for _, g in sample}
    assert groups == {f"group{i}" for i in range(10)}, (
        "a head-of-list sample would only see the first groups, a fixed stride "
        "an aliased subset of them")
    assert spread_sample(rows, 60) == sample          # deterministic
    assert spread_sample(rows, 60, seed=7) != sample  # seeded, not global RNG
    assert spread_sample(rows, 5000) == rows          # n larger than the list
    assert spread_sample([], 10) == []
    # the sample is not systematically biased to one end of the list
    assert min(i for i, _ in sample) < 60 and max(i for i, _ in sample) > 900


def test_loaded_checkpoint_infers_a_missing_prior_strength(tmp_path):
    """Old checkpoints must not get the correction their head cannot use."""
    import json

    node2emb = {i: torch.randn(8) for i in range(30)}
    rel2idx = {"r0": 0, "r1": 1}
    edges = [(i, (i + 3) % 30, f"r{i % 2}") for i in range(30)]
    cfg = WiazkaConfig(op_dim=16, num_generators=4, gat_num_layers=1,
                       gat_heads=1, batch_size=16, epochs=1, device="cpu",
                       decoder_balance=0.5)
    trainer = WiazkaTrainer(node2emb, rel2idx, edges, cfg)
    ckpt = str(tmp_path / "ckpt")
    trainer.save(ckpt)
    cfg_path = f"{ckpt}/config.json"
    meta = json.load(open(cfg_path, encoding="utf-8"))
    # simulate a checkpoint written before the scalar existed
    meta["config"].pop("relation_prior_strength", None)
    json.dump(meta, open(cfg_path, "w", encoding="utf-8"), ensure_ascii=False)

    assert RelationFlow.load(ckpt, device="cpu").model.relation_prior_strength == 0.0

    meta["config"]["decoder_balance"] = 0.0
    meta["config"].pop("relation_prior_strength", None)
    json.dump(meta, open(cfg_path, "w", encoding="utf-8"), ensure_ascii=False)
    assert RelationFlow.load(ckpt, device="cpu").model.relation_prior_strength == 1.0


# ------------------------------------------- ranking protocol (the query row)
def test_query_row_can_never_be_counted_as_the_answer():
    """``never_valid`` masks a column that cannot be the answer for a probe.

    ``a -r-> b`` is never answered by ``a`` (the Słowosieć has no self-loops:
    0 of 359 818 edges) and a retrieval pipeline does not return the query
    document.  Left in the index the query row is the candidate the rewritten
    query resembles most, so it is counted above the gold at small ``lam`` and
    the dial is rewarded for rewriting more than the metric needs.
    """
    from relgat_projector.core.metrics.ranking import filtered_ranking

    # index rows: 0 = the query itself, 1 = the gold, 2 = a distractor
    index = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    query = index[[0]]                     # no rewriting: the query is x itself
    gold = torch.tensor([1])

    _, ranks = filtered_ranking(query, index, gold)
    assert int(ranks[0]) == 2, "the query row must count against the model"

    _, ranks = filtered_ranking(query, index, gold, never_valid=[[0]])
    assert int(ranks[0]) == 1, "the query row must be masked out"

    # the gold column is never masked, even if a caller lists it: masking it
    # would make it -inf and score a perfect rank 1 out of an unusable query
    _, ranks = filtered_ranking(query, index, gold, never_valid=[[1]])
    assert int(ranks[0]) == 2

    # other true targets keep their meaning, the two lists are independent
    _, ranks = filtered_ranking(query, index, gold, other_targets=[[2]],
                                never_valid=[[0]])
    assert int(ranks[0]) == 1


def test_rank_mask_excludes_the_query_row_for_every_protocol():
    """Metric, dial fit and report share one protocol definition."""
    trainer = _trainer()
    probe = trainer._probe_edges(12)
    others, never_valid = trainer._rank_mask(probe)
    assert len(others) == len(never_valid) == len(probe)
    assert never_valid == [[int(s)] for s, _d, _r in probe]
    # a masked column is never the gold of its own probe
    for (s, d, r), masked in zip(probe, others):
        assert d not in masked
        assert s not in trainer._targets_by_src_rel.get((s, r), ())


def test_lam_zero_is_scorred_as_no_rewriting_not_as_a_loss():
    """The dial can only choose "do not rewrite" if that option is measurable.

    At ``lam = 0`` the served query *is* the base embedding, so the fit has to
    see the raw baseline there.  With the query row left in the index the
    source is one phantom position above the gold, i.e. the score of "do not
    rewrite" is understated relative to every value of ``lam`` that moves
    ``x_a`` away -- a systematic push towards over-rewriting.
    """
    from relgat_projector.core.metrics.ranking import filtered_ranking

    trainer = _trainer()
    probe = trainer._probe_edges(24)
    e = torch.tensor(probe, dtype=torch.long)
    src, dst, rel = e[:, 0], e[:, 1], e[:, 2]
    base = trainer._node_emb_dev
    others, never_valid = trainer._rank_mask(probe)

    with torch.no_grad():
        trainer.model.transport_dial.fill_(-20.0)   # lam ~ 0 for every relation
        query = trainer.model.production_query(base[src], rel)
    assert torch.allclose(query, base[src], atol=1e-4)

    _, ranks = filtered_ranking(query, base, dst, other_targets=others,
                                never_valid=never_valid)
    _, raw = filtered_ranking(base[src], base, dst, other_targets=others,
                              never_valid=never_valid)
    assert torch.equal(ranks, raw), "lam=0 must reproduce the raw baseline"
    assert float(ranks.float().reciprocal().mean()) > 0.0

    # ... and the phantom competitor is what used to hide that
    _, broken = filtered_ranking(query, base, dst, other_targets=others)
    assert float(broken.float().reciprocal().mean()) < \
        float(ranks.float().reciprocal().mean())


def test_dial_fit_prefers_no_rewriting_for_an_untrained_transport():
    """A dial fitted by scanning must be able to settle near zero.

    The transport of a freshly initialised model is noise, so the best answer
    for most relations is "barely rewrite".  The gradient path (``L_pull``,
    ~256 sampled queries per step) never managed it -- every dial stayed at
    ~0.81 for all 56 relations -- and the old protocol could not even score
    ``lam = 0``.
    """
    trainer = _trainer()
    fit = trainer.calibrate_transport_dial(n_per_rel=24)
    assert set(fit) == set(range(trainer.num_rel))
    assert min(fit.values()) <= 0.2, (
        f"a noise transport must be switched off somewhere, got {fit}")
    # the fitted value is the value the model serves
    with torch.no_grad():
        for r, lam in fit.items():
            served = float(trainer.model.transport_strength(torch.tensor([r]))[0])
            assert abs(served - lam) < 0.06


def test_smooth_curve_kills_spikes_but_keeps_boundaries():
    """The dial scan is fitted on ~100 edges, so its argmax must be de-noised.

    Averaging must not, however, invent an interior optimum: a curve that only
    decreases (the honest shape when rewriting never helps) still has to peak at
    ``lam = 0``.
    """
    from relgat_projector.trainer.wiazka_trainer import smooth_curve

    decreasing = [0.30, 0.25, 0.20, 0.15, 0.10]
    assert smooth_curve(decreasing) == smooth_curve(decreasing)
    assert max(range(5), key=lambda i: smooth_curve(decreasing)[i]) == 0

    spiky = [0.20, 0.21, 0.60, 0.22, 0.23]          # one noisy probe's worth
    smoothed = smooth_curve(spiky)
    assert max(range(5), key=lambda i: smoothed[i]) != 2, "a spike is not an optimum"

    interior = [0.10, 0.20, 0.30, 0.20, 0.10]        # a real optimum survives
    assert max(range(5), key=lambda i: smooth_curve(interior)[i]) == 2

    assert smooth_curve([0.5]) == [0.5]
    assert smooth_curve([0.5, 0.4], window=1) == [0.5, 0.4]


def test_qr_recall_filters_against_the_graph_not_the_sample():
    """A correct answer outside the sample must not count as a miss.

    ``synonimia`` has subjects with dozens of correct answers; a probe that
    lists one of them used to filter none of the others, so QR-Recall scored a
    perfect rewrite as a miss and understated exactly the relations whose
    multiplicity is highest.
    """
    from relgat_projector.core.metrics.query_rewriting import qr_recall

    index = torch.tensor([[1.0, 0.0, 0.0],
                          [0.8, 0.2, 0.0],     # the gold answer
                          [0.9, 0.1, 0.0],     # another correct answer
                          [0.0, 0.0, 1.0]])
    pairs = [(0, 1)]                            # the sample lists one answer
    identity = lambda x: x

    sampled = qr_recall(pairs, index, identity, ks=(1,))
    graph = qr_recall(pairs, index, identity, ks=(1,), other_map={0: [1, 2]})

    assert int(sampled.ranks_after[0]) == 2, "the unlisted gold outranks it"
    assert int(graph.ranks_after[0]) == 1, "the graph table must filter it out"
    assert graph.after[1] == 1.0 and sampled.after[1] == 0.0
    assert graph.mrr_after > sampled.mrr_after


def test_load_checkpoint_restores_the_fitted_scalars(tmp_path):
    """The low-level loader must not hand back an uncalibrated head.

    Both calibration scalars are attributes rather than parameters, so they are
    not in the state dict.  Restoring them only in ``RelationFlow.load`` left
    ``load_checkpoint`` -- the loader the trainer and any programmatic user
    reaches for -- with the constructor defaults, and the default prior of an
    unweighted head applied to a class-balanced one scores macro 0.05 against
    0.47.
    """
    import json

    from relgat_projector.handlers.wiazka_checkpoint import load_checkpoint

    trainer = _trainer(decoder_balance=0.5)
    trainer.model.relation_prior_strength = 0.25
    trainer.model.relation_ensemble_weight = 1.75
    ckpt = str(tmp_path / "ck")
    trainer.save(ckpt)

    model, _rel2idx, _spec, _meta = load_checkpoint(ckpt, device="cpu")
    assert model.relation_prior_strength == 0.25
    assert model.relation_ensemble_weight == 1.75

    # a checkpoint predating the scalars keeps the value its head implies
    cfg_path = f"{ckpt}/config.json"
    meta = json.load(open(cfg_path, encoding="utf-8"))
    meta["config"].pop("relation_prior_strength", None)
    meta["config"].pop("relation_ensemble_weight", None)
    json.dump(meta, open(cfg_path, "w", encoding="utf-8"), ensure_ascii=False)
    model, *_ = load_checkpoint(ckpt, device="cpu")
    assert model.relation_prior_strength == 0.0        # balanced head: no prior
    assert model.relation_ensemble_weight == 1.0       # neutral, as before


def test_edge_probe_unique_by_caps_high_target_sources():
    """The ranking protocol scores one query per (source, relation).

    A uniform draw lets a source with 40 true targets contribute 40 rows,
    so the metric then describes the target multiplicity of the graph, not
    the model.  ``unique_by="src_rel"`` makes the probe measure the model.
    """
    edges = [(0, d, 0) for d in range(40)] + [(i, (i + 1) % 60, 1) for i in range(1, 60)]
    capped = edge_probe(edges, 40, seed=1, unique_by="src_rel")
    # the capped probe has at most one row per (src, rel) and fills its budget
    ckeys = [(s, r) for s, _d, r in capped]
    assert len(ckeys) == len(set(ckeys)) == 40
    # and the cap is what makes it so: the same draw without it duplicates
    # the high-target source (40 of 100 rows are (0, 0))
    raw_all = edge_probe(edges, 40, seed=1)
    rkeys = [(s, r) for s, _d, r in raw_all]
    assert len(set(rkeys)) < len(rkeys)


def test_selection_prefers_a_better_traffic_model():
    """The checkpoint criterion must track the shipped production quality."""
    base = {"eval/mrr": 0.2, "eval/mrr_macro": 0.2, "eval/prod_mrr": 0.2,
            "eval/rel_acc_macro_ensemble": 0.4}
    weak_traffic = dict(base, **{"eval/prod_mrr_traffic_bestlam": 0.30})
    strong_traffic = dict(base, **{"eval/prod_mrr_traffic_bestlam": 0.55})
    assert (WiazkaTrainer.selection_score(strong_traffic)
            > WiazkaTrainer.selection_score(weak_traffic))
