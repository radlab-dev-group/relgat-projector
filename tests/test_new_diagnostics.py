"""Tests of the three diagnostics added to the 12/01 benchmark suite.

Each one is pinned on a construction whose answer is known without a model:
a permutation transport, a hand-typed pair set, and a synthetic embedding
space whose cosine threshold is set by hand.
"""

import numpy as np
import pytest
import torch

from relgat_projector.core.diagnostics import (
    ANSWER,
    OUT_OF_SCOPE,
    REFUSE,
    auto_cosine_threshold,
    cosine_dedup,
    decide,
    dedup_metrics,
    high_cosine_contradiction_share,
    identity_error,
    orbit_entropy,
    orbit_labels,
    roc_auc,
    synthesize,
    threshold_entropy,
    typed_dedup,
)


class _FakeTyper:
    """Types a fixed list of pairs with ids the test controls."""

    name = "fake"

    def __init__(self, ids, conf=None):
        self.ids = list(ids)
        self.conf = list(conf) if conf is not None else [1.0] * len(ids)

    def prepare(self, x):
        return x

    def type_pairs(self, a, b):
        n = a.shape[0]
        return (np.asarray(self.ids[:n]), np.asarray(self.conf[:n]),
                np.ones(n))


# ---------------------------------------------------------------- idea 02: dedup
def test_cosine_dedup_drops_the_later_neighbour():
    v = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    keep = cosine_dedup(v, 0.99)
    assert keep.tolist() == [True, False, True]


def test_typed_dedup_merges_only_the_syn_edge():
    # rows<cols pairs of a 3-item set are (0,1),(0,2),(1,2): type (0,1) syn
    v = np.eye(3, dtype=np.float32)
    keep, merges = typed_dedup(v, _FakeTyper([0, 1, 1]), [0], tau_conf=0.5,
                               candidate_top_k=2)
    assert keep.tolist() == [True, False, True]
    assert len(merges) == 1 and merges[0][:2] == (0, 1)


def test_dedup_metrics_reads_cost_and_benefit_together():
    keep = np.array([True, False, True, True])
    m = dedup_metrics(keep, [(0, 1), (2, 3)], [(0, 2)])
    assert m["reduction"] == pytest.approx(0.25)
    # the ant pair (0,1) lost a member, the surviving one did not
    assert m["conflict_recall"] == pytest.approx(0.5)
    assert m["duplicate_recall"] == pytest.approx(1.0)


def test_high_cosine_share_counts_what_the_threshold_would_delete():
    v = np.array([[1.0, 0.0], [0.999, 0.01], [0.0, 1.0]], dtype=np.float32)
    s = high_cosine_contradiction_share(v, [(0, 1)], [(0, 2)], threshold=0.9)
    assert s["pairs_above_threshold"] >= 1
    assert s["contradiction_share"] > 0.0


# ------------------------------------------------------- idea 04: orbit entropy
def test_orbit_entropy_is_zero_inside_one_orbit():
    assert orbit_entropy(np.array([2, 2, 2])) == pytest.approx(0.0)
    assert orbit_entropy(np.array([0, 0, 1, 1])) == pytest.approx(np.log(2))


def test_orbit_labels_are_transitive():
    # 0~1 and 1~2 collapse all three into one orbit even though 0~2 was not
    # typed syn at all
    # two syn edges, 0~1 and 1~2; no 0~2 edge at all
    labels = orbit_labels(3, np.array([0, 0]), np.array([1.0, 1.0]),
                          np.array([0, 1]), np.array([1, 2]), [0])
    assert len(set(labels.tolist())) == 1


def test_auto_threshold_is_not_the_quantile_of_a_pair():
    """A 2-item set has one cosine; its quantile is that number, so the
    baseline would always merge and score a flat 0.0."""
    pair = np.array([[1.0, 0.0], [0.7, 0.7]], dtype=np.float32)
    assert np.isnan(auto_cosine_threshold(pair))
    corpus = np.eye(6, dtype=np.float32)[:, :3]
    assert np.isfinite(auto_cosine_threshold(np.vstack([corpus, corpus * .5])))


def test_decide_reports_the_three_states_separately():
    assert decide({"alpha": 0.1, "kappa": 0.9, "H_orb": 0.9}) == OUT_OF_SCOPE
    assert decide({"alpha": 0.9, "kappa": 0.9, "H_orb": 0.9}) == REFUSE
    assert decide({"alpha": 0.9, "kappa": 0.0, "H_orb": 0.0}) == ANSWER


def test_threshold_entropy_uses_the_supplied_cut():
    v = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    assert threshold_entropy(v, 0.9) == pytest.approx(
        orbit_entropy(np.array([0, 0, 1])))


# --------------------------------------------- idea 11: constraint synthesis
class _PermutationTransport:
    """An exact similarity (a rotation), so the identity's gold is exactly 0."""

    relations = ("r", "s")

    def __init__(self, perm):
        self.p = np.asarray(perm)

    def prepare(self, x):
        return x

    def apply(self, x, rel):
        idx = self.p if rel == "r" else np.argsort(self.p)
        return x[..., torch.as_tensor(idx, dtype=torch.long)]

    def inverse(self, x, rel):
        idx = np.argsort(self.p) if rel == "r" else self.p
        return x[..., torch.as_tensor(idx, dtype=torch.long)]


def test_identity_error_is_zero_for_a_rotation():
    rng = np.random.default_rng(0)
    x = torch.as_tensor(rng.normal(size=(8, 5)).astype(np.float32))
    tr = _PermutationTransport([2, 0, 4, 1, 3])
    err = identity_error(tr, x, "r")
    assert err["identity_error_max"] < 1e-5


def test_synthesize_respects_an_active_constraint():
    rng = np.random.default_rng(1)
    x = torch.as_tensor(rng.normal(size=(6, 5)).astype(np.float32))
    tr = _PermutationTransport([2, 0, 4, 1, 3])
    goal, forbid = x[:1], x[1:2]
    sol = synthesize(tr, goal, forbid, "r", "s", tau=-1.0)   # always active
    # the returned direction is unit, and it is perpendicular to the forbidden
    # direction in the reduced space, which is what the closed form promises
    assert float(sol["v"].norm()) == pytest.approx(1.0, abs=1e-5)
    assert float((sol["v"] * sol["b_rot"]).sum()) < 1e-5


def test_roc_auc_matches_mann_whitney_under_heavy_ties():
    """Ties used to be written back to the wrong rows of the rank vector.

    The old code averaged a tied block into ``ranks[i:j+1]`` -- positions in
    the *sorted* order used as if they indexed the input -- so a tie group
    overwrote the ranks of unrelated rows. Perfect separation of 60 origins
    against 480 impostors printed 0.9988 instead of 1.0, an error of the same
    size as the lifts every AUC in this repository reports.
    """
    rng = np.random.default_rng(0)

    def brute(labels, scores):
        labels = np.asarray(labels, float)
        scores = np.asarray(scores, float)
        pos, neg = scores[labels > 0.5], scores[labels < 0.5]
        wins = int((pos[:, None] > neg[None, :]).sum())
        wins += 0.5 * int((pos[:, None] == neg[None, :]).sum())
        return wins / (len(pos) * len(neg))

    worst = 0.0
    for _ in range(60):
        n = int(rng.integers(12, 60))
        # integers only: every score is tied with several others by design
        scores = rng.integers(0, 5, n).astype(float)
        labels = (rng.random(n) > 0.5).astype(float)
        if 0 < labels.sum() < n:
            worst = max(worst, abs(roc_auc(labels, scores)
                                   - brute(labels, scores)))
    assert worst == 0.0, f"roc_auc drifts from Mann-Whitney by {worst:.2e}"


def test_roc_auc_of_a_perfect_separation_with_ties_is_one():
    assert roc_auc([1, 1, 1, 0, 0, 0], [1.0, 1.0, 1.0, 0.2, 0.1, 0.0]) == 1.0
    assert roc_auc([1, 1, 0, 0], [0.5, 0.5, 0.5, 0.5]) == 0.5
