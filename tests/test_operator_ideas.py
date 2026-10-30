"""
Pins for the operator-level diagnostics of ideas 03, 05, 06 and 07.

Every test here is a claim that, if it broke, would turn one of the four
sections into a number that reads well and means nothing -- the failure mode
this repository has already paid for once (a cycle loss that was a tautology).
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from relgat_projector.core.diagnostics.augmentation import (
    dial_curve,
    isometry_error,
)
from relgat_projector.core.diagnostics.consistency import (
    AdditiveTransport,
    Transport,
)
from relgat_projector.core.diagnostics.fingerprint import (
    fingerprint_delta,
    theta_angle_matrix,
)
from relgat_projector.core.diagnostics.pseudonymization import (
    anonymity_radius,
    k_anonymity,
    linkage_auc,
    retrieval_mrr,
)
from relgat_projector.core.diagnostics.ripple import (
    adjacency,
    connection_edit,
    concentration_auc,
    incident_relations,
    local_hd,
    loop_pairs,
)
from relgat_projector.core.operator import build_operator


class _OperatorTransport(Transport):
    """A real Givens operator behind the diagnostics' transport protocol."""

    def __init__(self, op, names):
        self.op = op
        self.relations = tuple(names)
        self.rel2idx = {n: i for i, n in enumerate(names)}

    def apply(self, x, rel):
        return self.op.apply(x, torch.full((x.shape[0],), self.rel2idx[rel],
                                           device=x.device, dtype=torch.long))

    def apply_at(self, x, rel, lambdas):
        return self.op.apply(
            x,
            torch.full((x.shape[0],), self.rel2idx[rel], device=x.device,
                       dtype=torch.long),
            lambdas=torch.full((x.shape[0],), float(lambdas), device=x.device))

    def inverse(self, x, rel):
        return self.op.inverse(x, torch.full((x.shape[0],), self.rel2idx[rel],
                                             device=x.device, dtype=torch.long))


@pytest.fixture()
def transport():
    torch.manual_seed(7)
    names = ["synonimia", "antonimia właściwa", "podtyp", "derywacja"]
    op = build_operator("givens", dim=24, num_rel=len(names), num_generators=8)
    return _OperatorTransport(op, names)


# ------------------------------------------------------------------ idea 03
def test_augmentation_is_a_similarity_so_cosines_survive(transport):
    v = torch.randn(40, 24)
    assert isometry_error(transport, v, "synonimia") < 1e-5


def test_dial_starts_at_the_trivial_pair_and_moves(transport):
    v = torch.randn(40, 24)
    curve = dial_curve(transport, v, "synonimia", (0.0, 0.5, 1.0))
    assert curve["self_cos_0.0"] == pytest.approx(1.0, abs=1e-5)
    assert curve["dial_range"] > 1e-3, "a dial that does not move is not a dial"


def test_an_additive_baseline_is_still_commutative(transport):
    # the floor of 12b/03: a translation separates the orders by exactly zero,
    # so a non-zero number here would mean the baseline is not a translation
    base = torch.randn(8, 24)
    additive = AdditiveTransport(
        {n: torch.randn(24) for n in transport.relations}, transport.relations)
    # (a+b)+c against (a+c)+b: equal in the algebra, and equal to within the
    # float associativity error, which is what "commutative baseline" means
    u = additive.apply(additive.apply(base, "synonimia"), "antonimia właściwa")
    w = additive.apply(additive.apply(base, "antonimia właściwa"), "synonimia")
    assert float((1 - F.cosine_similarity(u, w, dim=-1)).max()) < 1e-6


# ------------------------------------------------------------------ idea 05
def _toy_graph():
    # 0 -syn-> 1, 0 -ant-> 2, 0 -syn-> 3, 4 isolated
    edge_index = torch.tensor([[0, 0, 0], [1, 2, 3]])
    edge_type = torch.tensor([0, 1, 0])
    return edge_index, edge_type


def test_loops_need_two_distinct_relations_at_the_vertex():
    edge_index, edge_type = _toy_graph()
    names = ["synonimia", "antonimia", "podtyp"]
    inc = incident_relations(edge_index, edge_type, names, [0, 1, 2, 4])
    assert len(loop_pairs(inc[0])) == 1      # syn x ant
    assert loop_pairs(inc[4]) == []          # isolated: no HD at all


def test_adjacency_is_undirected():
    edge_index, _ = _toy_graph()
    adj = adjacency(edge_index)
    assert 0 in adj[1] and 1 in adj[0]


def test_zero_edit_changes_nothing_and_restores_the_weights():
    torch.manual_seed(3)
    names = ["synonimia", "antonimia"]
    op = build_operator("givens", dim=16, num_rel=2, num_generators=6,
                        use_context=True)
    model = torch.nn.Module()
    model.operator = op
    before = op.ctx_P.weight.detach().clone()
    with connection_edit(model, torch.randn(1, 16), 0.0):
        assert torch.equal(op.ctx_P.weight.detach(), before)
    assert torch.equal(op.ctx_P.weight.detach(), before)

    with connection_edit(model, torch.randn(1, 16), 0.5):
        assert not torch.equal(op.ctx_P.weight.detach(), before)
    assert torch.allclose(op.ctx_P.weight.detach(), before, atol=1e-7)


def test_concentration_auc_of_identical_samples_is_the_floor():
    # an edit that does nothing must score 0.5, not 0 and not 1: this is the
    # floor every reported concentration number is read against
    assert concentration_auc([0.1, 0.1, 0.1], [0.1, 0.1]) == pytest.approx(0.5)


def test_local_hd_of_a_commutative_transport_is_zero():
    additive = AdditiveTransport({"a": torch.randn(12), "b": torch.randn(12)},
                                 ["a", "b"])
    x = torch.randn(1, 12)
    assert local_hd(additive, x, [("a", "b")])["hd"] == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------------------------ idea 06
def test_linkage_attack_is_blind_only_when_it_should_be():
    corpus = F.normalize(torch.randn(60, 16), dim=-1)
    assert linkage_auc(corpus, corpus, top_k=8) == pytest.approx(1.0)
    shuffled = corpus[torch.randperm(60)]
    assert linkage_auc(corpus, shuffled, top_k=8) < 1.0


def test_k_anonymity_counts_the_whole_corpus_at_radius_two():
    corpus = F.normalize(torch.randn(30, 8), dim=-1)
    row = k_anonymity(corpus, corpus, epsilon=2.0)
    assert row["k_anon_median"] == 30


def test_anonymity_radius_is_a_measured_quantile_not_a_constant():
    corpus = F.normalize(torch.randn(80, 16), dim=-1)
    eps = anonymity_radius(corpus, share=0.01)
    assert 0.0 < eps < 2.0
    # a tighter share must be a smaller ball
    assert anonymity_radius(corpus, share=0.1) > eps


def test_retrieval_on_the_unprotected_index_is_perfect():
    corpus = F.normalize(torch.randn(50, 16), dim=-1)
    assert retrieval_mrr(corpus, corpus, corpus, list(range(50)), k=10) == 1.0


# ------------------------------------------------------------------ idea 07
def test_theta_angle_matrix_is_invariant_to_a_change_of_basis():
    """The whole fingerprint rests on this: ``<QAQ-, QBQ-> = <A,B>``."""
    torch.manual_seed(11)
    rotations = torch.linalg.matrix_exp(
        torch.randn(5, 10, 10) - torch.randn(5, 10, 10).transpose(-1, -2))
    q, _ = torch.linalg.qr(torch.randn(10, 10))
    conjugated = q @ rotations @ q.t()
    a = theta_angle_matrix(rotations)
    b = theta_angle_matrix(conjugated)
    np.testing.assert_allclose(a, b, atol=1e-4)


def test_fingerprint_of_a_model_with_itself_is_zero():
    torch.manual_seed(5)
    rotations = torch.linalg.matrix_exp(
        torch.randn(4, 8, 8) - torch.randn(4, 8, 8).transpose(-1, -2))
    angles = theta_angle_matrix(rotations)
    fp = {"theta_angles": angles, "commutation": np.eye(4) * 0.5}
    delta = fingerprint_delta(fp, dict(fp))
    assert delta["d_theta_angles"] == pytest.approx(0.0)
    assert delta["d_commutation"] == pytest.approx(0.0)


def test_delta_names_the_relation_that_regressed():
    fp_old = {"theta_angles": np.zeros((3, 3))}
    fp_new = {"theta_angles": np.zeros((3, 3))}
    old_acc = {0: (40, 40), 1: (38, 40)}
    new_acc = {0: (40, 40), 1: (20, 40)}
    delta = fingerprint_delta(fp_old, fp_new, old_acc, new_acc,
                              min_probes=20, tol=0.05)
    assert [r for r, _, _ in delta["regressed_relations"]] == [1]


def test_a_relation_with_too_few_probes_is_no_data_not_ok():
    delta = fingerprint_delta({"theta_angles": np.zeros((3, 3))},
                              {"theta_angles": np.zeros((3, 3))},
                              {2: (5, 5)}, {2: (0, 5)}, min_probes=20)
    assert delta["regressed_relations"] == []
