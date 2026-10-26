"""Tests for the label-free consistency diagnostics (ideas/12).

The transport interface is the whole model-facing surface of the module, so
the tests run it with a tiny synthetic transport instead of a checkpoint:
exact values are known for a hand-built operator, and the fixtures and the
aggregation (AUC signs, the 0.5-prior bookkeeping of C_rs, the commutative
baseline) are checked directly.
"""

import json
import math
import os

import numpy as np
import pytest
import torch

from relgat_projector.core.diagnostics import (
    AdditiveTransport,
    OrderSeparationReport,
    Transport,
    document_scores,
    order_description_separation,
    order_separation,
    pair_contradiction_scores,
    relative_holonomy,
    roc_auc,
    spearman,
)
from relgat_projector_apps.diagnostics.consistency_report import (
    find_best_checkpoint,
    negate,
)
from relgat_projector_apps.diagnostics.make_consistency_fixture import (
    DOCUMENTS,
    LEXICON,
    ORDER_PAIRS,
    POSITIVE_LABELS,
)


class RotationTransport(Transport):
    """A 2-D rotation transport: T_r(x) = R(angle_r) x.

    Non-commutative in general, exactly invertible, flat (holonomy 0) --
    the reference against which the additive baseline and the metrics are
    checked.
    """

    def __init__(self, angles):
        self.angles = angles
        self.relations = tuple(angles)

    def _rot(self, angle):
        c, s = math.cos(angle), math.sin(angle)
        return torch.tensor([[c, -s], [s, c]], dtype=torch.float64)

    def apply(self, x, rel):
        return x @ self._rot(self.angles[rel]).t()

    def inverse(self, x, rel):
        return x @ self._rot(-self.angles[rel]).t()


def _transport():
    return RotationTransport({"a": 0.3, "b": 0.7})


# ------------------------------------------------------------------- statistics
def test_roc_auc_extremes_and_ties():
    assert roc_auc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(1.0)
    assert roc_auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(0.0)
    # every score tied: AUC is exactly 0.5, not an error
    assert roc_auc([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)
    # no positives / no negatives: defined as nan
    assert roc_auc([1, 1], [0.2, 0.1]) != roc_auc([1, 1], [0.2, 0.1])


def test_spearman():
    x = [1.0, 2.0, 3.0, 4.0]
    assert spearman(x, x) == pytest.approx(1.0)
    assert spearman(x, list(reversed(x))) == pytest.approx(-1.0)
    # a constant side carries no variance
    assert spearman(x, [1.0] * 4) != spearman(x, [1.0] * 4)


# ----------------------------------------------------------------- the metrics
def test_additive_transport_is_the_exact_floor():
    """The translation family must score separation AND holonomy of 0."""
    torch.manual_seed(0)
    x = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
    tr = AdditiveTransport(
        {"a": torch.randn(16), "b": torch.randn(16)}, ["a", "b"])
    c = np.full((2, 2), 0.5)
    report = order_separation(x, tr, c)
    assert report.sep_min == pytest.approx(0.0, abs=1e-6)
    assert report.sep_max == pytest.approx(0.0, abs=1e-6)
    assert relative_holonomy(tr, x[0], ["a", "b"]) == pytest.approx(0.0, abs=1e-6)
    assert relative_holonomy(tr, x[0:2], ["a", "b", "a", "b"]) == pytest.approx(0.0, abs=1e-6)


def test_rotation_transport_holonomy_is_flat():
    tr = _transport()
    x = torch.nn.functional.normalize(torch.randn(4, 2), dim=-1).double()
    # rotations in the same plane commute: the closed loop closes exactly
    assert relative_holonomy(tr, x[0], ["a", "b"]) == pytest.approx(0.0, abs=1e-12)
    # order separation of commuting rotations is 0 as well
    u = tr.apply_chain(x, ["a", "b"])
    w = tr.apply_chain(x, ["b", "a"])
    assert torch.nn.functional.cosine_similarity(u, w, dim=-1).mean() == pytest.approx(1.0, abs=1e-12)


def test_order_separation_against_c():
    tr = RotationTransport({"a": 0.3, "b": 0.7})
    # 2-D rotations commute, so use a 3-D transport where they do not
    def rot(axis, angle):
        if axis == 0:
            c, s = np.cos(angle), np.sin(angle)
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    class R3(Transport):
        def __init__(self):
            self.relations = ("a", "b")
            self.M = {"a": torch.tensor(rot(0, 0.3), dtype=torch.float64),
                      "b": torch.tensor(rot(1, 0.7), dtype=torch.float64)}

        def apply(self, x, rel):
            return x @ self.M[rel].t()

        def inverse(self, x, rel):
            return x @ self.M[rel]

    x = torch.nn.functional.normalize(torch.randn(16, 3), dim=-1).double()
    report = order_separation(x, R3(), np.array([[0.5, 0.1], [0.9, 0.5]]))
    assert isinstance(report, OrderSeparationReport)
    assert report.n_pairs == 1
    # rotations around different axes do not commute: the two routes end
    # apart; with these angles the gap is ~1.5e-2, above the exact-
    # commutation floor (pinned at 0 by the additive test) but small, which
    # is what keeps the metric honest instead of a binary "separated" flag
    assert 0.01 < report.sep_median < 0.03
    # C is a label the report buckets pairs with, not an input to the metric:
    # the same transport, re-run with a different C, must measure the same
    # separation and only change which bucket the pair falls into
    other = order_separation(x, R3(), np.array([[0.5, 0.9], [0.1, 0.5]]))
    assert other.sep_median == pytest.approx(report.sep_median)
    assert other.c_high == 1 and report.c_low == 1
    assert report.n_c_evidence == 1 and other.n_c_evidence == 1


def test_document_scores_separate_consistent_from_contradictory():
    """K must be higher for claims that are synonyms than for antonyms."""
    torch.manual_seed(0)
    dim = 32

    class Directional(Transport):
        # syn pushes toward a common direction, ant pushes away from it
        def __init__(self):
            self.relations = ("synonimia", "antonimia właściwa")
            self.v_syn = torch.nn.functional.normalize(torch.randn(dim, dtype=torch.float64), dim=-1)
            self.v_ant = torch.nn.functional.normalize(torch.randn(dim, dtype=torch.float64), dim=-1)

        def apply(self, x, rel):
            v = self.v_syn if rel == "synonimia" else self.v_ant
            return x + 0.5 * v.to(x.dtype)

        def inverse(self, x, rel):
            v = self.v_syn if rel == "synonimia" else self.v_ant
            return x - 0.5 * v.to(x.dtype)

    tr = Directional()
    good = torch.stack([
        torch.nn.functional.normalize(
            torch.randn(dim, dtype=torch.float64) + 0.3 * tr.v_syn, dim=-1)
        for _ in range(5)])
    bad = torch.stack([
        torch.nn.functional.normalize(
            torch.randn(dim, dtype=torch.float64) - 0.3 * tr.v_syn, dim=-1)
        for _ in range(5)])
    good_scores = document_scores(good, tr)
    bad_scores = document_scores(bad, tr)
    assert good_scores["coherence_k"] > bad_scores["coherence_k"]


def test_pair_contradiction_scores_rank_gold_pairs():
    torch.manual_seed(0)
    dim = 32

    class AntTransport(Transport):
        """T_ant(v) = -v: the exact antonym transport."""

        def __init__(self):
            self.relations = ("antonimia właściwa",)

        def apply(self, x, rel):
            return -x

        def inverse(self, x, rel):
            return -x

    # c0 and c1 are antonyms (gold), c2 is off to the side
    c0 = torch.nn.functional.normalize(torch.randn(dim), dim=-1)
    c1 = -c0
    c2 = torch.nn.functional.normalize(torch.randn(dim), dim=-1)
    claims = torch.stack([c0, c1, c2])
    result = pair_contradiction_scores(claims, AntTransport(), [[0, 1]])
    assert result["n_gold"] == 1
    assert result["n_pairs"] == 3
    # the antonym pair must score at 1 and the off-side pairs below
    assert result["mean_gold"] == pytest.approx(1.0, abs=1e-5)
    assert result["mean_other"] == pytest.approx(0.0)
    assert result["auc"] == pytest.approx(1.0)


def test_order_description_separation_requires_same_shape():
    with pytest.raises(ValueError):
        order_description_separation(torch.randn(4, 8), torch.randn(3, 8))
    out = order_description_separation(torch.randn(5, 8), torch.randn(5, 8))
    assert 0.0 <= out["desc_separation"] <= 2.0


# ----------------------------------------------------------------- app helpers
def test_negate_prefaces_the_claim():
    assert negate("Dom jest duży.", "pl").startswith("Nieprawdą jest, że dom")
    assert negate("The house is big.", "en").startswith("It is not the case that the")


def test_find_best_checkpoint_prefers_higher_score(tmp_path):
    for name, score in (("run_a", 0.30), ("run_b", 0.34), ("run_c", float("nan"))):
        run = tmp_path / name
        run.mkdir()
        (run / "report.json").write_text(
            json.dumps({"checkpoint_scores": {"best": score}}), encoding="utf-8")
    found = find_best_checkpoint(str(tmp_path))
    assert found is not None
    assert found.endswith("run_b" + os.sep + "best")
    assert find_best_checkpoint(str(tmp_path / "nowhere")) is None


def test_fixture_is_consistent_and_bilingual():
    """The generator's invariants: labels, gold pairs, task/lexicon coverage."""
    for doc in DOCUMENTS:
        assert doc["label"] in POSITIVE_LABELS + ("coherent",)
        assert len(doc["claims"]) >= 2
        n = len(doc["claims"])
        for a, b in doc["contradictions"]:
            assert 0 <= a < n and 0 <= b < n and a != b
        for claim in doc["claims"]:
            assert claim["pl"] and claim["en"]
    # every relation of every order task has a lexicon and a template in BOTH
    # languages, otherwise the baseline transport and the descriptions break
    from relgat_projector_apps.diagnostics.make_consistency_fixture import RELATION_TEMPLATES
    for pair in ORDER_PAIRS:
        for rel in pair:
            assert rel in LEXICON, f"{rel} missing from the lexicon"
            assert rel in RELATION_TEMPLATES, f"{rel} missing a description template"
            for lang in ("pl", "en"):
                assert LEXICON[rel][lang]
                assert "{inner}" in RELATION_TEMPLATES[rel][lang]
    # the generated JSONL files must be parseable and line-aligned
    import os
    fixture_dir = "data/consistency"
    if os.path.isdir(fixture_dir):
        docs = [json.loads(l) for l in open(f"{fixture_dir}/documents_pl_en.jsonl", encoding="utf-8")]
        tasks = [json.loads(l) for l in open(f"{fixture_dir}/order_tasks_pl_en.jsonl", encoding="utf-8")]
        assert len(docs) == len(DOCUMENTS)
        assert all(t["desc_a"]["pl"] and t["desc_b"]["en"] for t in tasks)
        # and the two order readings really differ, per task and per language
        assert all(t["desc_a"][lang] != t["desc_b"][lang] for t in tasks for lang in ("pl", "en"))
