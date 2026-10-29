"""
Relational-clustering diagnostics (docs/IDEAS/01-relational-clustering.md).

The claim under test: group by the **type of relation** between two vectors,
not by their distance.  Cosine puts "the battery lasts two days" and "the
battery dies in hours" in one cloud, because they share topic, register and
skeleton -- so every pipeline that clusters and then summarises merges
contradictions into one sentence.

Three things are measured, all without annotation beyond the fixture's
gold-by-construction labels:

* **the go/no-go test** (idea 01 section 8): can the relation decoder tell an
  ``ant`` pair from a ``syn`` pair at all -- the AUC of its confidence on the
  two classes.  Below ~0.6 the whole family of ideas (1, 2, 4, 12a) has to wait
  for sentence-level training, and that is worth knowing before building.
* **contradiction purity (CP)**: the fraction of clusters that contain *no*
  contradicting pair.  ARI alone cannot see the failure -- a method can score a
  fine ARI while hiding every antonymy inside a cluster.
* **the signed-graph clustering itself**: candidates -> typed edges (+attraction
  for ``syn``, -repulsion for ``ant``) -> greedy correlation clustering with **no
  ``k``**, which is what makes it comparable to k-means at all: the baseline is
  handed the gold number of clusters, this method is not.

The model-facing surface is :class:`EdgeTyper`, and it is abstract: the learned
decoder (:class:`DecoderTyper`), the head-free transport fidelity
(:class:`TransportTyper`) and the embedder-free antonym baseline
(:class:`AntonymBaselineTyper`) all implement it, so the same bilingual fixture
runs trained or untrained, on any embedder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

#: canonical relation names, as in the Słowosieć export
REL_SYN = "synonimia"
REL_ANT = "antonimia właściwa"


# --------------------------------------------------------------------- helpers
def _unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def cosine_matrix(x: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
    """Cosine similarity of ``x`` against ``y`` (default: against itself)."""
    a, b = _unit(np.asarray(x, dtype=np.float64)), _unit(
        np.asarray(x if y is None else y, dtype=np.float64))
    return a @ b.T


def candidate_pairs(similarity: np.ndarray, top_k: int = 50,
                    min_cos: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Candidate pairs from ANN-style pruning: the ``top_k`` nearest of each row.

    The idea document's cost argument: the graph is linear in the number of
    *edges*, not quadratic in N.  A pipeline that types every pair costs
    ``N^2`` decoder calls; this prunes to ``N * top_k`` and degrades gracefully
    (fewer edges = weaker signal, no systematic error).
    """
    sim = np.array(similarity, dtype=np.float64, copy=True)
    n = sim.shape[0]
    np.fill_diagonal(sim, -np.inf)
    k = int(min(max(1, top_k), n - 1))
    idx = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    cols = idx.reshape(-1)
    keep = sim[rows, cols] >= float(min_cos)
    rows, cols = rows[keep], cols[keep]
    # keep each unordered pair once, and keep it in a canonical (a < b) order
    keep = rows < cols
    return rows[keep], cols[keep]


# ---------------------------------------------------------------- the typer API
class EdgeTyper(Protocol):
    """Given paired vectors, say *which* relation holds and with what confidence."""

    name: str

    def type_pairs(self, a: torch.Tensor,
                   b: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(rel_id, confidence, applicability)`` for every pair."""
        ...


class DecoderTyper:
    """The learned relation head: ``model.decode`` plus ``applicability``.

    The head sees a pair of *operator-space* vectors, so :meth:`prepare` charts
    the base vectors once (``base_to_op``) -- feeding it raw embeddings is a
    different space and silently returns noise.
    """

    name = "decoder"

    def __init__(self, model, rel2idx: Dict[str, int], device: str = "cpu"):
        self.model = model
        self.rel2idx = {str(k): int(v) for k, v in rel2idx.items()}
        self.device = torch.device(device)

    @torch.no_grad()
    def prepare(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.base_to_op(x.to(self.device))

    @torch.no_grad()
    def type_pairs(self, a, b):
        a = a.to(self.device)
        b = b.to(self.device)
        ids, conf = self.model.decode(a, b)
        appl = torch.sigmoid(self.model.applicability_score(a, ids))
        return (ids.cpu().numpy(), conf.cpu().numpy(), appl.cpu().numpy())


class TransportTyper:
    """Head-free typing: the relation that best *transports* ``a`` onto ``b``.

    ``argmax_r cos(T_r a, b)`` is exactly the quantity the edge loss optimises,
    so it needs no extra head and no extra training.  Its "confidence" is the
    transport fidelity itself, which is why it is not comparable to the head's
    sigmoid -- the two are reported side by side, never mixed.
    """

    name = "transport"

    def __init__(self, model, rel2idx: Dict[str, int], device: str = "cpu"):
        self.model = model
        self.rel2idx = {str(k): int(v) for k, v in rel2idx.items()}
        self.device = torch.device(device)

    @torch.no_grad()
    def prepare(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.base_to_op(x.to(self.device))

    @torch.no_grad()
    def type_pairs(self, a, b):
        scores = self.model.transport_relation_scores(
            a.to(self.device), b.to(self.device))
        conf, ids = scores.max(dim=-1)
        ones = torch.ones_like(conf)
        return (ids.cpu().numpy(), conf.cpu().numpy(), ones.cpu().numpy())


class AntonymBaselineTyper:
    """Embedder-free typing of the ``syn`` / ``ant`` pair classes only.

    Scores a pair by how much better the antonym step than the synonym step
    explains it: ``+cos(T_syn a, b)`` attracts, ``-cos(T_ant a, b)`` repels.
    With the commutative :class:`AdditiveTransport` of the consistency module
    this is a pure translation baseline -- the floor the learned operator has to
    beat, and the reason this diagnostic runs without any checkpoint at all.
    """

    name = "antonym-baseline"

    def __init__(self, transport, rel_syn: str = REL_SYN, rel_ant: str = REL_ANT):
        self.transport = transport
        self.rel_syn = rel_syn
        self.rel_ant = rel_ant
        # ids of the two classes in the fixture's own numbering: 0 = syn, 1 = ant
        self.ids = {0: rel_syn, 1: rel_ant}

    def prepare(self, x: torch.Tensor) -> torch.Tensor:
        return self.transport.prepare(x)

    @torch.no_grad()
    def type_pairs(self, a, b):
        syn = self.transport.apply(a, self.rel_syn)
        ant = self.transport.apply(a, self.rel_ant)
        b_u = F.normalize(b.to(syn.device).to(syn.dtype), p=2, dim=-1)
        s = (F.normalize(syn, p=2, dim=-1) * b_u).sum(-1)
        d = (F.normalize(ant, p=2, dim=-1) * b_u).sum(-1)
        ant_wins = d > s
        ids = ant_wins.long()
        conf = (d - s).abs().clamp(0.0, 1.0)
        appl = torch.ones_like(conf)
        return (ids.cpu().numpy(), conf.cpu().numpy(), appl.cpu().numpy())


# ------------------------------------------------------------------ the metrics
def adjusted_rand_index(labels: Sequence[int],
                        gold: Sequence[int]) -> float:
    """ARI of two partitions, in numpy (sklearn is not a dependency of this repo).

    ``ARI = (obs - exp) / (max - exp)`` with the pair-counting convention; 1.0
    for identical partitions, 0.0 at chance, negative below it.
    """
    a, b = _contingency(labels, gold)
    n = float(a.sum())
    if n < 2.0:
        return float("nan")
    sum_comb = lambda v: float((v * (v - 1.0)).sum() / 2.0)
    obs = sum_comb(a)
    row, col = a.sum(axis=1), a.sum(axis=0)
    exp = sum_comb(row) * sum_comb(col) * 2.0 / (n * (n - 1.0))
    maxv = 0.5 * (sum_comb(row) + sum_comb(col))
    denom = maxv - exp
    return float("nan") if abs(denom) < 1e-12 else (obs - exp) / denom


def normalized_mutual_info(labels: Sequence[int], gold: Sequence[int]) -> float:
    """NMI with the geometric-mean normalisation, in numpy."""
    a, _ = _contingency(labels, gold)
    n = a.sum()
    if n <= 0:
        return float("nan")
    p = a / n
    row, col = p.sum(axis=1), p.sum(axis=0)
    nz = p > 0
    mi = float((p[nz] * np.log(p[nz] / np.outer(row, col)[nz])).sum())
    hy = -float((row[row > 0] * np.log(row[row > 0])).sum())
    hx = -float((col[col > 0] * np.log(col[col > 0])).sum())
    denom = np.sqrt(hx * hy)
    return float("nan") if denom <= 0 else mi / float(denom)


def _dense_codes(values: Sequence) -> np.ndarray:
    """Dense 0..k-1 codes of an arbitrary label sequence (ints or strings).

    Both partitions of every metric here are categorical: the gold clusters of
    the fixture are ``"topic|stance"`` strings, the predicted ones are integers,
    and neither ordering carries meaning, so both are recoded the same way.
    """
    arr = np.asarray(list(values))
    if arr.dtype.kind in "US":
        _, codes = np.unique(arr, return_inverse=True)
        return codes.astype(np.int64)
    return arr.astype(np.int64)


def _contingency(labels: Sequence[int], gold: Sequence[int]):
    lab = _dense_codes(labels)
    go = _dense_codes(gold)
    if lab.shape != go.shape:
        raise ValueError(f"label/gold shape mismatch: {lab.shape} vs {go.shape}")
    _, lab_c = np.unique(lab, return_inverse=True)
    _, go_c = np.unique(go, return_inverse=True)
    a = np.zeros((int(lab_c.max()) + 1 if lab_c.size else 0,
                  int(go_c.max()) + 1 if go_c.size else 0), dtype=np.float64)
    if lab_c.size:
        np.add.at(a, (lab_c, go_c), 1.0)
    return a, lab_c


def contradiction_purity(labels: Sequence[int], ant_pairs: Sequence[Tuple[int, int]]
                         ) -> Dict[str, float]:
    """Share of clusters that contain **no** contradicting pair.

    This is idea 01's target metric: a cluster that merges "the battery is
    great" with "the battery is terrible" is not a slightly wrong cluster, it is
    a wrong answer for every downstream summarise-then-report pipeline, and
    ARI/NMI do not see it at all.
    """
    lab = np.asarray(list(labels), dtype=np.int64)
    clusters = np.unique(lab)
    dirty = set()
    for i, j in ant_pairs:
        if lab[i] == lab[j]:
            dirty.add(int(lab[i]))
    n = int(clusters.size)
    sizes = np.bincount(lab - int(clusters.min())) if n else np.array([0])
    return {
        "purity": float(1.0 - len(dirty) / n) if n else float("nan"),
        "n_clusters": n,
        "n_contradicting": int(len(dirty)),
        "n_ant_pairs_hidden": int(sum(
            1 for i, j in ant_pairs if lab[i] == lab[j])),
        "singleton_share": float((sizes == 1).mean()) if n else float("nan"),
    }


def intra_cluster_holonomy(x: np.ndarray, labels: Sequence[int],
                           transport, rel_a: str = REL_SYN,
                           rel_b: str = REL_ANT,
                           max_per_cluster: int = 8) -> Dict[str, float]:
    """Defect of a closed ``a,b,a^-1,b^-1`` loop walked *inside* each cluster.

    The idea document's label-free quality check: if the transport around a
    cluster is path inconsistent, the cluster is apparent rather than real.
    Metric clustering cannot produce this number at all -- it needs the
    relation structure, which is exactly the point.
    """
    from .consistency import relative_holonomy

    lab = np.asarray(list(labels), dtype=np.int64)
    defects: List[float] = []
    for c in np.unique(lab):
        rows = np.flatnonzero(lab == c)[:max_per_cluster]
        if rows.size < 2:
            continue
        seg = torch.as_tensor(x[rows], dtype=torch.float32)
        doc = F.normalize(seg.mean(dim=0, keepdim=True), p=2, dim=-1).squeeze(0)
        defects.append(relative_holonomy(transport, doc, [rel_a, rel_b]))
    return {
        "mean": float(np.mean(defects)) if defects else float("nan"),
        "max": float(np.max(defects)) if defects else float("nan"),
        "n_clusters": len(defects),
    }


# ------------------------------------------------------------------ baselines
def cosine_kmeans(x: np.ndarray, k: int, seed: int = 0,
                  iterations: int = 50) -> np.ndarray:
    """k-means on cosine distance, with k-means++ seeding -- the baseline.

    Handed the gold ``k`` on purpose: the relational method never gets a
    ``k``, so if it wins anyway the win is not an artefact of the baseline
    being under-parameterised.
    """
    a = _unit(np.asarray(x, dtype=np.float64))
    n = a.shape[0]
    k = int(max(1, min(k, n)))
    rng = np.random.default_rng(seed)
    # k-means++: seed the first centre uniformly, the rest proportional to the
    # squared distance to the closest centre already chosen
    centres = a[rng.integers(n)][None]
    while centres.shape[0] < k:
        d = 1.0 - (a @ centres.T)
        d = np.min(d, axis=1)
        d = np.clip(d, 0.0, None)
        total = float(d.sum())
        p = (d / total) if total > 0 else np.full(n, 1.0 / n)
        centres = np.vstack([centres, a[rng.choice(n, p=p)]])
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(int(iterations)):
        sim = a @ _unit(centres).T
        new = np.argmax(sim, axis=1)
        if np.array_equal(new, labels) and _ > 0:
            labels = new
            break
        labels = new
        for c in range(k):
            rows = labels == c
            if rows.any():
                centres[c] = _unit(a[rows].mean(axis=0, keepdims=True))[0]
    # relabel densely so the cluster count is comparable with other methods
    _, dense = np.unique(labels, return_inverse=True)
    return dense.astype(np.int64)


def cosine_threshold_components(x: np.ndarray, threshold: float) -> np.ndarray:
    """Connected components of the ``cos >= threshold`` graph: SemDeDup-style dedup.

    The second baseline from the idea document, and the one a practitioner
    actually reaches for.  It has no ``k`` either, which makes it the fair
    comparison for a method that also has none.
    """
    sim = cosine_matrix(x)
    n = sim.shape[0]
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    roots = np.array([find(i) for i in range(n)], dtype=np.int64)
    _, dense = np.unique(roots, return_inverse=True)
    return dense.astype(np.int64)


# ------------------------------------------------- the relational method
def signed_weights(rows: np.ndarray, cols: np.ndarray, rel_id: np.ndarray,
                   conf: np.ndarray, appl: np.ndarray, ant_ids: Sequence[int],
                   syn_ids: Sequence[int], tau_attr: float = 0.0,
                   tau_appl: float = 0.0,
                   top_fraction: float = 0.2) -> Tuple[np.ndarray, np.ndarray,
                                                       np.ndarray]:
    """Typed candidate pairs -> a signed, sparse graph ``(rows, cols, weight)``.

    ``+conf`` for an attracting relation (``syn`` and friends), ``-conf`` for a
    repelling one (``ant``), and *no edge at all* where ``applicability`` says
    the relation is not defined: the "no edge" class is what stops the decoder
    hallucinating a relation between unrelated items.

    The graph keeps the strongest ``top_fraction`` of the typed candidates by
    ``|conf|`` rather than an absolute confidence threshold, because the
    confidence of a sigmoid head and the confidence of a transport score are not
    on one scale -- and on this space they cannot be: the raw index has a mean
    pairwise cosine of 0.946, so any fixed cut is either everything or nothing.
    A quantile is the same decision for every typer, which is what makes the
    methods comparable; ``tau_attr`` / ``tau_appl`` stay as optional extra
    filters (``0.0`` = off) for a caller who really means an absolute cut.
    """
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    conf = np.asarray(conf, dtype=np.float64)
    keep = (conf >= tau_attr) & (np.asarray(appl, dtype=np.float64) >= tau_appl)
    ant = np.isin(np.asarray(rel_id), list(ant_ids))
    syn = np.isin(np.asarray(rel_id), list(syn_ids))
    keep &= (ant | syn)
    # an empty typed set is a result, not an error: it says the typer named no
    # relation above the cut for any candidate, which the caller reports as
    # "no edges" (S4 does exactly this, and that is one of its measurements)
    if keep.any() and 0.0 < top_fraction < 1.0:
        magnitude = np.abs(conf)
        n_candidates = int(keep.sum())
        n_keep = int(np.clip(np.ceil(n_candidates * float(top_fraction)), 1,
                             n_candidates))
        limit = np.partition(magnitude[keep], n_candidates - n_keep)[
            n_candidates - n_keep]
        keep &= magnitude >= limit
    sign = np.where(np.isin(np.asarray(rel_id), list(ant_ids))[keep], -1.0, 1.0)
    return rows[keep], cols[keep], sign * conf[keep]


def signed_greedy_clustering(n: int, rows: np.ndarray, cols: np.ndarray,
                             weights: np.ndarray,
                             objective: Optional[Dict[str, float]] = None
                             ) -> np.ndarray:
    """Correlation clustering without ``k``: greedy merge on the signed graph.

    Start with every item in its own cluster and merge the most attractive pair
    of clusters while the total weight *between* them stays positive.  A
    negative edge therefore never splits a cluster -- it simply makes the merge
    unattractive -- which is the behaviour the idea document asks for: nodes the
    evidence does not support stay alone rather than being forced into the
    nearest cloud, and no ``k`` is ever supplied.

    The objective is the agreement correlation-clustering objective

        sum_{{i<j}} w_ij * [same cluster]

    so a merge that would trade purity for ARI is visible in
    :func:`clustering_objective` instead of silent.
    """
    parent = list(range(int(n)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def key(u: int, v: int) -> Tuple[int, int]:
        return (u, v) if u < v else (v, u)

    # cluster-level weights: the graph is sparse and stays sparse under
    # contraction, so a dict is both exact and cheap at fixture scale
    w: Dict[Tuple[int, int], float] = {}
    for r, c, v in zip(np.asarray(rows, dtype=np.int64),
                       np.asarray(cols, dtype=np.int64),
                       np.asarray(weights, dtype=np.float64)):
        if int(r) == int(c):
            continue
        k = key(int(r), int(c))
        w[k] = w.get(k, 0.0) + float(v)

    # the merge order is by the weight of the pair being examined; after a
    # contraction the cluster-level weights change, so the order is refreshed
    # from the current map at every step instead of being frozen at the start
    while w:
        (a, b), best = max(w.items(), key=lambda kv: kv[1])
        if best <= 0.0:
            # every remaining pair is non-positive: merging would only put
            # items together that the graph says should stay apart
            break
        ra, rb = find(a), find(b)
        if ra == rb:
            del w[(a, b)]
            continue
        # total weight between the two clusters, over all cross edges
        total = 0.0
        for (u, v), val in w.items():
            if key(find(int(u)), find(int(v))) == key(ra, rb):
                total += val
        if total <= 0.0:
            del w[(a, b)]
            continue
        parent[rb] = ra
        contracted: Dict[Tuple[int, int], float] = {}
        for (u, v), val in w.items():
            fu, fv = find(int(u)), find(int(v))
            if fu == fv:
                continue
            ck = key(fu, fv)
            contracted[ck] = contracted.get(ck, 0.0) + val
        w = contracted

    out = np.array([find(i) for i in range(int(n))], dtype=np.int64)
    _, dense = np.unique(out, return_inverse=True)
    if objective is not None:
        objective.update(clustering_objective(
            n, rows, cols, weights, dense))
    return dense.astype(np.int64)


def clustering_objective(n: int, rows: np.ndarray, cols: np.ndarray,
                         weights: np.ndarray, labels: Sequence[int]
                         ) -> Dict[str, float]:
    """How much signed weight a partition explains, and how much it swallows.

    ``agreement`` is the correlation-clustering objective (the weight inside the
    clusters); ``negative_inside`` is the repulsion the partition swallowed
    against its own evidence -- the continuous twin of contradiction purity --
    and ``positive_cut`` is the attraction it broke by splitting.
    """
    lab = np.asarray(list(labels), dtype=np.int64)
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    if rows.size == 0:
        return {"agreement": 0.0, "negative_inside": 0.0, "positive_cut": 0.0}
    same = lab[rows] == lab[cols]
    inside = weights[same]
    return {
        "agreement": float(inside.sum()),
        "negative_inside": float(-inside[inside < 0.0].sum()),
        "positive_cut": float(-weights[~same][weights[~same] > 0.0].sum()),
    }


# ------------------------------------------------------------------- the report
@dataclass
class ClusteringReport:
    """One row of the idea-01 table: a method, its clusters and its purity."""

    method: str
    n_items: int
    n_clusters: int
    ari: float
    nmi: float
    contradiction_purity: float
    contradicting_clusters: int
    ant_pairs_hidden: int
    singleton_share: float
    extra: Dict[str, float] = field(default_factory=dict)

    def as_row(self) -> Dict[str, float]:
        return {"method": self.method, "n_clusters": self.n_clusters,
                "ari": self.ari, "nmi": self.nmi,
                "cp": self.contradiction_purity,
                "hidden": self.ant_pairs_hidden,
                "singletons": self.singleton_share, **self.extra}


def evaluate_partition(items: List[Dict], labels: Sequence[int], method: str,
                       pair_index: Optional[Dict[str, Sequence[Tuple[int, int]]]]
                       = None) -> ClusteringReport:
    """Score one partition against the fixture's gold labels.

    ``pair_index`` must hold the gold ``ant`` pairs; they are what contradiction
    purity is measured with, and passing them explicitly keeps the metric from
    silently falling back to "no contradictions" when a caller forgets.
    """
    gold = [it["group"] for it in items]
    if pair_index is None:
        pair_index = gold_pair_index(items)
    ant = pair_index["ant"]
    lab = np.asarray(list(labels), dtype=np.int64)
    if lab.shape[0] != len(items):
        raise ValueError(f"{method}: {lab.shape[0]} labels for {len(items)} items")
    cp = contradiction_purity(lab, ant)
    return ClusteringReport(
        method=method,
        n_items=len(items),
        n_clusters=cp["n_clusters"],
        ari=adjusted_rand_index(lab, gold),
        nmi=normalized_mutual_info(lab, gold),
        contradiction_purity=cp["purity"],
        contradicting_clusters=cp["n_contradicting"],
        ant_pairs_hidden=cp["n_ant_pairs_hidden"],
        singleton_share=cp["singleton_share"],
    )


def gold_pair_index(items: List[Dict]) -> Dict[str, List[Tuple[int, int]]]:
    """Gold pair classes read off the item labels (syn / ant / mixed / neutral)."""
    out: Dict[str, List[Tuple[int, int]]] = {"syn": [], "ant": [], "mixed": [],
                                             "neutral": []}
    n = len(items)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = items[i], items[j]
            same_topic = a["topic"]["pl"] == b["topic"]["pl"]
            if not same_topic:
                key = "neutral"
            elif a.get("mixed") or b.get("mixed"):
                key = "mixed"
            elif a["stance"] == b["stance"]:
                key = "syn"
            else:
                key = "ant"
            out[key].append((i, j))
    return out


def typing_auc(scores: Sequence[float], pairs: Sequence[Tuple[int, int]],
               positives: Sequence[Tuple[int, int]],
               negatives: Sequence[Tuple[int, int]]) -> float:
    """AUC of a pair score separating ``positives`` from ``negatives``.

    A thin wrapper over :func:`roc_auc` that takes *pairs* rather than arrays,
    so the go/no-go test reads as what it is: "does the decoder's confidence
    rank the antonymy pairs above the synonymy pairs?"
    """
    from .consistency import roc_auc

    s = np.asarray(list(scores), dtype=float)
    index = {tuple(p): float(s[k]) for k, p in enumerate(map(tuple, pairs))}
    pos = [index[tuple(p)] for p in positives if tuple(p) in index]
    neg = [index[tuple(p)] for p in negatives if tuple(p) in index]
    if not pos or not neg:
        return float("nan")
    return roc_auc([1.0] * len(pos) + [0.0] * len(neg), pos + neg)
