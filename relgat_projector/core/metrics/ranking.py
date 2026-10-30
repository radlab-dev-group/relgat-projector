"""
Filtered link-prediction metrics (MRR / Hits@k) for the WIĄZKA pipeline.

This is the metric that answers "does the model actually predict the
relation": for a held-out edge ``a -r-> b`` we rank *every* node ``c`` by

    score(c) = cos( T_r(x_a), x_c )

and report the reciprocal rank of ``b``.  The **filtered** protocol masks
every *other* true target of the same ``(a, r)`` pair (the graph is a
multi-graph and a synonym not in the probe set is still a correct answer),
so an unlabelled-but-correct relation does not count as a miss.  A raw-base
baseline (no operator, plain cosine ``x_a`` vs ``x_c``) is reported next to
the model numbers so that the actual lift of the relational transport is
visible in a single line.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


def filtered_mrr_hits(
    query: torch.Tensor,          # [P, D]   T_r(x_a) for the probe edges
    index: torch.Tensor,          # [N, D]   refined node vectors (the index)
    target: torch.Tensor,         # [P]      row of the gold answer in index
    other_targets: Optional[Sequence[Sequence[int]]] = None,
    never_valid: Optional[Sequence[Sequence[int]]] = None,
    ks: Sequence[int] = (1, 3, 5, 10),
    chunk: int = 512,
) -> Dict[str, float]:
    """Filtered MRR and Hits@k for probe edges against the full index."""
    stats, _ = filtered_ranking(
        query, index, target, other_targets, never_valid, ks, chunk)
    return stats


def filtered_ranking(
    query: torch.Tensor,
    index: torch.Tensor,
    target: torch.Tensor,
    other_targets: Optional[Sequence[Sequence[int]]] = None,
    never_valid: Optional[Sequence[Sequence[int]]] = None,
    ks: Sequence[int] = (1, 3, 5, 10),
    chunk: int = 512,
) -> Tuple[Dict[str, float], torch.Tensor]:
    """Same protocol as :func:`filtered_mrr_hits`, but also returns the ranks.

    ``other_targets`` are *other correct answers* (the graph is a multi-graph)
    and ``never_valid`` are columns that can *never* be the answer for that
    probe -- for a query ``a -r-> b`` over the full node index, the row of
    ``a`` itself: a word is not its own synonym, and a retrieval pipeline
    would not return the query document as the answer.  Both are masked out of
    the ranking; the gold column is never masked even if it appears in either
    list.

    The per-probe ranks make it possible to average **per relation** (macro),
    which is the only honest average on a graph where one relation type holds
    half the edges and others hold a handful.
    """
    q = torch.nn.functional.normalize(query, p=2, dim=-1)
    k = torch.nn.functional.normalize(index, p=2, dim=-1)
    P = q.shape[0]
    mrr = 0.0
    hits = {kk: 0.0 for kk in ks}
    all_ranks = torch.zeros(P, dtype=torch.long)
    for start in range(0, P, chunk):
        stop = min(start + chunk, P)
        scores = q[start:stop] @ k.t()  # [C, N]
        if other_targets is not None or never_valid is not None:
            for i in range(start, stop):
                for c in (other_targets[i] if other_targets is not None else ()):
                    if c != int(target[i]):
                        scores[i - start, c] = float("-inf")
                for c in (never_valid[i] if never_valid is not None else ()):
                    if c != int(target[i]):
                        scores[i - start, c] = float("-inf")
        # rank = 1 + number of candidates scoring strictly above the gold.
        # The gold column is never strictly above itself, so it excludes itself;
        # equal scores are counted as "not beaten" (optimistic ties, matching
        # metrics.query_rewriting.rank_of_target).  Masked candidates are -inf
        # and therefore never counted.
        gold = scores.gather(1, target[start:stop].reshape(-1, 1))
        better = (scores > gold).sum(dim=-1)
        ranks = better + 1
        all_ranks[start:stop] = ranks.cpu()
        mrr += (1.0 / ranks.float()).sum().item()
        for kk in ks:
            hits[kk] += (ranks <= kk).sum().item()
    n = float(max(1, P))
    out: Dict[str, float] = {"mrr": mrr / n}
    out.update({f"hits@{kk}": hits[kk] / n for kk in ks})
    out["n"] = P
    return out, all_ranks


def edge_probe(
    edges: Sequence[Tuple[int, int, int]],
    max_probes: int,
    seed: int,
    unique_by: Optional[str] = None,
) -> List[Tuple[int, int, int]]:
    """Deterministic subset of edges to score (same every eval).

    ``unique_by="src_rel"`` caps each ``(source, relation)`` at one row: the
    ranking protocol scores one query against a *set* of answers, so a source
    with 40 true targets of one relation contributes one query, not 40.  A
    uniform draw without this cap over-represents exactly those high-target
    sources, so the metric then describes the target multiplicity of the graph
    instead of the model.
    """
    if unique_by:
        seen: set = set()
        kept: List[Tuple[int, int, int]] = []
        for e in edges:
            key = (int(e[0]), int(e[2]))
            if key in seen:
                continue
            seen.add(key)
            kept.append(e)
        edges = kept
        if len(edges) <= max_probes:
            return list(edges)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(edges), generator=g)[:max_probes]
    return [edges[i] for i in perm.tolist()]


def macro_mrr(ranks: torch.Tensor, groups: Sequence[int]) -> Dict[str, float]:
    """Mean reciprocal rank averaged over relations, not over probe edges.

    A micro average weights ``synonimia`` (half of the graph) 2000x more than a
    relation with a dozen edges; the macro average reports whether the rare
    relations were learned at all, which is where these models usually fail.
    """
    import collections

    buckets: Dict[int, List[float]] = collections.defaultdict(list)
    for g, r in zip(groups, ranks.tolist()):
        buckets[int(g)].append(1.0 / float(r))
    if not buckets:
        return {"mrr_macro": float("nan"), "num_relations": 0.0}
    per_relation = [sum(v) / len(v) for v in buckets.values()]
    return {
        "mrr_macro": sum(per_relation) / len(per_relation),
        "num_relations": float(len(per_relation)),
    }

def stratified_edge_probe(
    edges: Sequence[Tuple[int, int, int]],
    max_probes: int,
    seed: int,
    max_per_rel: Optional[int] = None,
) -> List[Tuple[int, int, int]]:
    """Relation-balanced deterministic probe.

    ``edge_probe`` draws uniformly from the edge list, so on the Slowosiec
    ``synonimia`` (178k of 360k edges) takes ~50% of every probe and the ~30
    relations with fewer than 100 edges never appear at all: the headline
    number then describes one relation, and a macro average is computed over
    a dozen arbitrary types.  Here every relation that occurs gets a share
    (round-robin over shuffled per-relation buckets), so micro and macro agree
    and rare relations are actually measured.
    """
    if not edges or max_probes <= 0:
        return []
    import collections

    g = torch.Generator().manual_seed(seed)
    buckets: Dict[int, List[Tuple[int, int, int]]] = collections.defaultdict(list)
    for e in edges:
        buckets[int(e[2])].append(tuple(e))
    rels = sorted(buckets)
    order = torch.randperm(len(rels), generator=g).tolist()
    caps = {r: max(1, max_per_rel) if max_per_rel else None for r in rels}
    for r in rels:
        perm = torch.randperm(len(buckets[r]), generator=g).tolist()
        buckets[r] = [buckets[r][i] for i in perm]
    picked: List[Tuple[int, int, int]] = []
    cursor = {r: 0 for r in rels}
    # round-robin: one edge per relation per pass, rare relations first so a
    # relation with 3 edges contributes its 3 before synonimia fills the probe
    rare_first = sorted(rels, key=lambda r: (len(buckets[r]), r))
    pass_no = 0
    while len(picked) < max_probes:
        progressed = False
        for r in (rare_first if pass_no == 0 else [rare_first[i] for i in order]):
            if len(picked) >= max_probes:
                break
            cap = caps[r]
            if cap is not None and cursor[r] >= cap:
                continue
            if cursor[r] >= len(buckets[r]):
                continue
            picked.append(buckets[r][cursor[r]])
            cursor[r] += 1
            progressed = True
        if not progressed:
            break
        pass_no += 1
    return picked
