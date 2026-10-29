"""Tests for the relational-clustering diagnostics (ideas/01).

The metrics here decide the verdict of a whole family of ideas, so the ones
numpy implements are pinned against values computed independently, and the
clustering method is pinned against partitions planted by hand: what it must
merge, what it must keep apart, and what it must do when the graph says
nothing.  No embedder and no checkpoint is needed for any of it -- the model
-facing surface is exercised with the embedder-free transports.
"""

import json
import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from relgat_projector.core.diagnostics import (
    AdditiveTransport,
    Transport,
    AntonymBaselineTyper,
    adjusted_rand_index,
    candidate_pairs,
    clustering_objective,
    contradiction_purity,
    cosine_kmeans,
    cosine_matrix,
    cosine_threshold_components,
    evaluate_partition,
    gold_pair_index,
    intra_cluster_holonomy,
    normalized_mutual_info,
    signed_greedy_clustering,
    signed_weights,
    typing_auc,
)
from relgat_projector_apps.diagnostics.clustering_report import ids_of, load_fixture
from relgat_projector_apps.diagnostics.make_clustering_fixture import (
    PAIR_ANTONYMOUS,
    PAIR_MIXED,
    PAIR_NEUTRAL,
    PAIR_SYNONYMOUS,
    gold_pairs,
    items,
)

REL_SYNONYM = "synonimia"
REL_ANTONYM = "antonimia właściwa"


def _items(spec):
    """Build fixture-shaped items from ``(topic, stance, mixed)`` triples."""
    return [{"id": f"i{k}", "topic": {"pl": t, "en": t}, "stance": s,
             "group": f"{t}|{s if not m else 'mixed'}", "mixed": m,
             "text": {"pl": f"tekst {k}", "en": f"text {k}"}}
            for k, (t, s, m) in enumerate(spec)]


class PlaneRotation(Transport):
    """Two rotations sharing one axis: non-commutative, exactly invertible.

    The transport end of :func:`intra_cluster_holonomy`.  Two rotations of the
    *same* plane would commute and the loop would close exactly, so this pairs
    a rotation of the 0-1 plane with one of the 0-2 plane: their commutator is
    a rotation of the 1-2 plane, and a loop of the two relations does not come
    back to where it started.
    """

    relations = (REL_SYNONYM, REL_ANTONYM)

    def __init__(self, angles=(0.9, 0.6)):
        self._m = {REL_SYNONYM: self._plane_rotation((0, 1), angles[0]),
                   REL_ANTONYM: self._plane_rotation((0, 2), angles[1])}

    @staticmethod
    def _plane_rotation(axes, angle):
        c, s = float(np.cos(angle)), float(np.sin(angle))
        matrix = np.eye(3)
        first, second = axes
        matrix[first, first] = matrix[second, second] = c
        matrix[first, second] = -s
        matrix[second, first] = s
        return matrix

    def apply(self, x, rel):
        return x @ torch.as_tensor(self._m[rel], dtype=x.dtype,
                                   device=x.device).t()

    def inverse(self, x, rel):
        return x @ torch.linalg.inv(
            torch.as_tensor(self._m[rel], dtype=x.dtype,
                            device=x.device))

def _antonym_transport():
    """An additive transport whose two relations point in different axes."""
    return AdditiveTransport(
        {REL_SYNONYM: torch.tensor([1.0, 0.0, 0.0]),
         REL_ANTONYM: torch.tensor([0.0, 1.0, 0.0])},
        [REL_SYNONYM, REL_ANTONYM], strength=1.0)


# ------------------------------------------------------------------- agreement
def test_ari_and_nmi_against_independent_values():
    gold = [0, 0, 1, 1, 2, 2]
    # identical partitions score 1.0, and so does any relabelling of one
    assert adjusted_rand_index(gold, gold) == pytest.approx(1.0)
    assert normalized_mutual_info(gold, gold) == pytest.approx(1.0)
    assert adjusted_rand_index([1, 1, 0, 0, 2, 2], gold) == pytest.approx(1.0)
    assert normalized_mutual_info([1, 1, 0, 0, 2, 2], gold) == pytest.approx(1.0)
    # one cluster: ARI is exactly 0 at chance, NMI is not defined
    assert adjusted_rand_index([0] * 6, gold) == pytest.approx(0.0)
    assert normalized_mutual_info([0] * 6, gold) != normalized_mutual_info(
        [0] * 6, gold)
    # the pair-counting value of ``[[2,0],[0,2]]`` against ``[0,0,1,1]``:
    # obs = 2 * C(2,2) = ... worked out by hand, 2 agreements out of 6 pairs
    assert adjusted_rand_index([0, 0, 1, 1], [0, 1, 0, 1]) == pytest.approx(-0.5)
    # gold labels of the fixture are strings, and the metric must not care
    assert adjusted_rand_index([0, 0, 1, 1], ["a", "a", "b", "b"]) \
        == pytest.approx(1.0)
    assert normalized_mutual_info([0, 0, 1, 1], ["a", "a", "b", "b"]) \
        == pytest.approx(1.0)


def test_nmi_rewards_not_clustering_where_ari_does_not():
    """The reason the report prints ARI *and* the singleton share.

    A partition into singletons knows nothing, yet NMI reads it as high
    agreement; ARI reads it as chance.  Pinned so a future normalisation
    change cannot quietly make the singleton degenerate look like a win.
    """
    gold = [0, 0, 1, 1, 2, 2]
    singletons = list(range(6))
    assert adjusted_rand_index(singletons, gold) == pytest.approx(0.0)
    assert normalized_mutual_info(singletons, gold) > 0.5


# ------------------------------------------------------- contradiction purity
def test_contradiction_purity_sees_what_ari_does_not():
    """A partition can score a fine ARI and still hide every contradiction.

    Two partitions of ``{topic} x {stance}`` items with the *same* number of
    clusters: one merges the two stances of each topic (the defect idea 01 is
    about), the other merges across topics.  Only CP separates them.
    """
    its = _items([("a", 1, False), ("a", -1, False),
                  ("b", 1, False), ("b", -1, False)])
    ant = gold_pair_index(its)["ant"]
    assert len(ant) == 2

    hides = [0, 0, 1, 1]        # {a+,a-} {b+,b-}: both clusters contradictory
    across = [0, 1, 1, 0]       # {a+,b+} {a-,b-}: merged topics, no contradiction
    assert adjusted_rand_index(hides, [its[i]["group"] for i in range(4)]) \
        == pytest.approx(adjusted_rand_index(across,
                                            [its[i]["group"] for i in range(4)]))
    cp_hides = contradiction_purity(hides, ant)
    cp_across = contradiction_purity(across, ant)
    assert cp_hides["purity"] == pytest.approx(0.0)
    assert cp_hides["n_ant_pairs_hidden"] == 2
    assert cp_across["purity"] == pytest.approx(1.0)
    assert cp_across["n_ant_pairs_hidden"] == 0


def test_contradiction_purity_edges_report_their_own_degnerates():
    its = _items([("a", 1, False), ("a", -1, False), ("b", 1, False)])
    ant = gold_pair_index(its)["ant"]
    # no contradictions anywhere: purity 1.0 is not evidence of anything
    assert contradiction_purity([0, 1, 2], [])["purity"] == pytest.approx(1.0)
    # a partition into singletons cannot hide anything either -- the report
    # has to read this together with the singleton share, which is 1.0 here
    merged_nothing = contradiction_purity(list(range(3)), ant)
    assert merged_nothing["purity"] == pytest.approx(1.0)
    assert merged_nothing["singleton_share"] == pytest.approx(1.0)


# ------------------------------------------------------------------ baselines
def test_cosine_kmeans_recovers_a_planted_partition():
    rng = np.random.default_rng(0)
    planted, gold = [], []
    for cluster in range(3):
        centre = rng.normal(0.0, 1.0, size=24)
        planted.append(centre[None] + rng.normal(0.0, 0.2, size=(5, 24)))
        gold += [cluster] * 5
    vectors = np.vstack(planted)
    assert adjusted_rand_index(cosine_kmeans(vectors, k=3, seed=0), gold) \
        == pytest.approx(1.0)
    # k is handed to the baseline on purpose: with k = 1 it must collapse to
    # chance, which is what makes the no-k method's win comparable
    assert adjusted_rand_index(cosine_kmeans(vectors, k=1), gold) \
        == pytest.approx(0.0)
    # labels are dense, so the cluster count of the baseline is comparable
    assert set(cosine_kmeans(vectors, k=3, seed=0).tolist()) == {0, 1, 2}


def test_cosine_threshold_components_merges_only_near_duplicates():
    vectors = np.array([[1.0, 0.0], [1.0, 1e-3], [0.0, 1.0], [0.0, 1.0]])
    merged = cosine_threshold_components(vectors, 0.999)
    assert adjusted_rand_index(merged, [0, 0, 1, 1]) == pytest.approx(1.0)
    # above every cosine in the set: nothing merges, and that is reported as
    # "merged nothing" by the app rather than as a perfect clustering
    assert len(set(cosine_threshold_components(vectors, 1.5))) == 4


# ------------------------------------------------------- the signed clustering
def test_signed_greedy_clustering_keeps_repelling_items_apart():
    # two attracting chains, and repulsion between every cross pair
    rows = np.array([0, 1, 2, 0, 2, 1])
    cols = np.array([1, 2, 3, 3, 3, 3])
    weights = np.array([2.0, 2.0, 2.0, -1.0, -1.0, -1.0])
    labels = signed_greedy_clustering(4, rows, cols, weights)
    # the strongest edge wants to merge the chains, but the *total* weight
    # across that cut is -2.0: the evidence says the chains stay apart, and a
    # single edge cannot outvote the rest
    assert sorted(labels.tolist()) == [0, 0, 0, 1]
    # the same graph with the cross repulsion removed: one cluster
    labels = signed_greedy_clustering(4, rows[:3], cols[:3], weights[:3])
    assert len(set(labels.tolist())) == 1


def test_signed_greedy_clustering_merges_attraction_only():
    rows = np.array([0, 0, 1])
    cols = np.array([1, 2, 2])
    labels = signed_greedy_clustering(3, rows, cols, np.array([1.0, 1.0, 1.0]))
    assert len(set(labels)) == 1


def test_signed_greedy_clustering_never_merges_on_repulsion_alone():
    rows = np.array([0, 1, 0])
    cols = np.array([1, 2, 2])
    labels = signed_greedy_clustering(3, rows, cols,
                                     np.array([-1.0, -2.0, -0.5]))
    assert sorted(labels.tolist()) == [0, 1, 2]


def test_signed_greedy_clustering_on_an_empty_graph():
    labels = signed_greedy_clustering(
        4, np.array([], dtype=int), np.array([], dtype=int),
        np.array([], dtype=float))
    assert sorted(labels.tolist()) == [0, 1, 2, 3]


def test_signed_weights_signs_and_quantile():
    rows = np.array([0, 1, 2, 3])
    cols = np.array([1, 2, 3, 4])
    rel_id = np.array([1, 0, 1, 0])          # ant, syn, ant, syn
    conf = np.array([0.9, 0.8, 0.1, 0.2])
    appl = np.ones(4)

    kept_rows, kept_cols, weight = signed_weights(
        rows, cols, rel_id, conf, appl, ant_ids=[1], syn_ids=[0],
        top_fraction=1.0)
    assert (weight == np.array([-0.9, 0.8, -0.1, 0.2])).all()
    # the quantile keeps the strongest |conf| only: same decision for every
    # typer, which is what makes the head and the transport comparable
    kept_rows, kept_cols, weight = signed_weights(
        rows, cols, rel_id, conf, appl, ant_ids=[1], syn_ids=[0],
        top_fraction=0.5)
    assert sorted(zip(kept_rows.tolist(), kept_cols.tolist())) == [(0, 1), (1, 2)]
    assert (weight == np.array([-0.9, 0.8])).all()
    # an absolute cut stays available for a caller who really means one
    kept_rows, _, weight = signed_weights(
        rows, cols, rel_id, conf, appl, ant_ids=[1], syn_ids=[0],
        top_fraction=1.0, tau_attr=0.85)
    assert kept_rows.tolist() == [0] and weight.tolist() == [-0.9]


def test_signed_weights_drops_relations_it_was_not_told_to_use():
    rows, cols = np.array([0, 1, 2]), np.array([1, 2, 3])
    _, _, weight = signed_weights(
        rows, cols, np.array([0, 1, 7]),           # 7 = hypernymy, neither pole
        np.array([0.9, 0.8, 0.7]), np.ones(3), ant_ids=[1], syn_ids=[0],
        top_fraction=1.0)
    assert weight.tolist() == [0.9, -0.8]


def test_signed_weights_on_an_empty_typed_set():
    """S4 types no edge at all; that is a measurement, not an IndexError."""
    empty = (np.array([], dtype=int), np.array([], dtype=int),
             np.array([], dtype=int), np.array([], dtype=float),
             np.array([], dtype=float))
    rows, cols, weight = signed_weights(*empty, ant_ids=[1], syn_ids=[0])
    assert rows.size == cols.size == weight.size == 0


def test_clustering_objective_reports_what_the_partition_swallows():
    rows = np.array([0, 1, 0, 2])
    cols = np.array([1, 2, 3, 3])
    weights = np.array([2.0, 2.0, -1.5, -1.5])
    # {0,1} {2,3}: the attractive edge 0-1 is kept, the attractive edge 2-3 is
    # cut, and both repulsions are swallowed inside the clusters
    part = clustering_objective(4, rows, cols, weights, [0, 0, 1, 1])
    assert part["agreement"] == pytest.approx(0.5)
    assert part["negative_inside"] == pytest.approx(1.5)
    assert part["positive_cut"] == pytest.approx(-2.0)
    # everything in one cluster: all four edges are inside, so the repulsion
    # it swallowed shows up as a negative_inside the purity metric cannot see
    all_one = clustering_objective(4, rows, cols, weights, [0, 0, 0, 0])
    assert all_one["agreement"] == pytest.approx(1.0)
    assert all_one["negative_inside"] == pytest.approx(3.0)
    assert all_one["positive_cut"] == pytest.approx(0.0)
    # a partition into singletons cuts every edge it was given
    alone = clustering_objective(4, rows, cols, weights, [0, 1, 2, 3])
    assert alone["agreement"] == pytest.approx(0.0)
    assert alone["negative_inside"] == pytest.approx(0.0)
    assert alone["positive_cut"] == pytest.approx(-4.0)
    # an empty graph explains nothing and swallows nothing
    empty = clustering_objective(4, np.array([], dtype=int),
                                 np.array([], dtype=int),
                                 np.array([], dtype=float), [0, 0, 1, 1])
    assert empty == {"agreement": 0.0, "negative_inside": 0.0,
                     "positive_cut": 0.0}


# ------------------------------------------------------------------- the typer
def test_antonym_baseline_typer_reads_the_two_directions():
    typer = AntonymBaselineTyper(_antonym_transport())
    a = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    # the first pair sits where the antonym step lands, the second where the
    # synonym step lands
    b = torch.tensor([[1.0, 1.0, 0.0], [2.0, 0.0, 0.0]])
    rel_id, conf, appl = typer.type_pairs(typer.prepare(a), typer.prepare(b))
    assert rel_id.tolist() == [1, 0]
    assert (conf >= 0.0).all() and (conf <= 1.0).all()
    assert (appl == 1.0).all()


def test_candidate_pairs_prunes_and_deduplicates():
    rng = np.random.default_rng(1)
    sim = cosine_matrix(rng.normal(0.0, 1.0, size=(12, 6)))
    rows, cols = candidate_pairs(sim, top_k=3)
    assert (rows < cols).all()                  # canonical, each pair once
    assert len(set(zip(rows.tolist(), cols.tolist()))) == rows.size
    assert rows.size <= 12 * 3
    # min_cos prunes, top_k larger than the corpus does not blow up
    strict_rows, _ = candidate_pairs(sim, top_k=3, min_cos=1.0)
    assert strict_rows.size == 0


def test_typing_auc_ranks_pairs_not_indices():
    pairs = [(0, 1), (2, 3), (4, 5)]
    assert typing_auc([0.9, 0.1, 0.2], pairs, [(0, 1)], [(2, 3), (4, 5)]) \
        == pytest.approx(1.0)
    assert typing_auc([0.1, 0.9, 0.8], pairs, [(0, 1)], [(2, 3), (4, 5)]) \
        == pytest.approx(0.0)
    assert typing_auc([0.5, 0.5, 0.5], pairs, [(0, 1)], [(2, 3), (4, 5)]) \
        == pytest.approx(0.5)
    # a class with no scored pair is "not measured", not 0.0
    missing = typing_auc([0.5, 0.5], [(0, 1), (2, 3)], [(0, 1)], [(8, 9)])
    assert missing != missing


# ------------------------------------------------------------------ holonomy
def test_intra_cluster_holonomy_is_flat_for_a_translation():
    torch.manual_seed(0)
    x = F.normalize(torch.randn(6, 3), dim=-1).numpy()
    scores = intra_cluster_holonomy(x, [0, 0, 1, 1, 2, 2], _antonym_transport())
    assert scores["mean"] == pytest.approx(0.0, abs=1e-6)
    assert scores["n_clusters"] == 3


def test_intra_cluster_holonomy_detects_a_non_commutive_loop():
    torch.manual_seed(0)
    x = F.normalize(torch.randn(6, 3).double(), dim=-1).numpy()
    scores = intra_cluster_holonomy(x, [0, 0, 1, 1, 2, 2], PlaneRotation())
    assert scores["mean"] > 1e-3
    # a cluster of one item has no loop to walk, and the mean is undefined
    alone = intra_cluster_holonomy(x, list(range(6)), PlaneRotation())
    assert alone["n_clusters"] == 0
    assert alone["mean"] != alone["mean"]


# ------------------------------------------------------------------- the app
def test_evaluate_partition_scores_against_the_gold_groups():
    its = _items([("a", 1, False), ("a", 1, False), ("a", -1, False),
                  ("b", 1, False)])
    row = evaluate_partition(its, [0, 0, 1, 2], "perfect").as_row()
    assert row["ari"] == pytest.approx(1.0)
    assert row["cp"] == pytest.approx(1.0)
    assert row["hidden"] == 0
    # a partition of the wrong length is a caller error, loudly
    with pytest.raises(ValueError, match="labels for"):
        evaluate_partition(its, [0, 1], "broken")


def test_ids_of_maps_relation_names_to_checkpoint_ids():
    rel2idx = {REL_SYNONYM: 3, REL_ANTONYM: 7, "hyperonimia": 1}
    assert ids_of([REL_ANTONYM, REL_SYNONYM], rel2idx) == [7, 3]
    assert ids_of(["nieistniejaca"], rel2idx) == []
    # the fixture's own typer numbers its two classes 0 and 1
    assert ids_of([REL_SYNONYM], None) == []


def test_fixture_gold_is_consistent_by_construction():
    rows = items()
    assert len({r["id"] for r in rows}) == len(rows)
    for row in rows:
        assert set(row["text"]) == {"pl", "en"}
        assert row["text"]["pl"] and row["text"]["en"]
        assert row["group"].startswith(row["topic"]["pl"] + "|")
        assert row["stance"] in (-1, 1)
    # every topic has both stances, so a contradiction pair exists for each
    by_topic = {}
    for row in rows:
        by_topic.setdefault(row["topic"]["pl"], set()).add(
            "mixed" if row["mixed"] else row["stance"])
    for topic, stances in by_topic.items():
        assert -1 in stances or "mixed" in stances, topic
        assert 1 in stances, topic

    counts = {}
    for pair in gold_pairs(rows):
        counts[pair["label"]] = counts.get(pair["label"], 0) + 1
    assert counts[PAIR_ANTONYMOUS] > counts[PAIR_SYNONYMOUS] > 0
    assert counts[PAIR_MIXED] > 0 and counts[PAIR_NEUTRAL] > 0
    # the two readings of the same gold must agree pair for pair
    index = gold_pair_index(rows)
    for label, name in ((PAIR_ANTONYMOUS, "ant"), (PAIR_SYNONYMOUS, "syn"),
                        (PAIR_MIXED, "mixed"), (PAIR_NEUTRAL, "neutral")):
        assert sorted(index[name]) == sorted(
            (p["a"], p["b"]) for p in gold_pairs(rows) if p["label"] == label)


def test_generated_fixture_on_disk_matches_the_generator(tmp_path):
    from relgat_projector_apps.diagnostics.make_clustering_fixture import main

    main(["--out-dir", str(tmp_path)])
    its, pairs = load_fixture(str(tmp_path))
    assert len(its) == len(items())
    assert [json.dumps(p, sort_keys=True) for p in pairs] == \
        [json.dumps(p, sort_keys=True) for p in gold_pairs(items())]
