"""
QR-Recall — does the relation transform actually help retrieval?

Everything else in ``core/metrics`` measures the *geometry* of the operator
(PCE, CCE, ISOT, HD).  This module measures the *product property*:

    given a query vector ``t`` and a relation ``r``, is the gold answer of
    ``r(t)`` ranked higher after the transform than before it?

Protocol (one gold pair ``(a, b)`` with edge ``a -r-> b``, an index of vectors
-- the contents of the real vector database -- and ``ks``):

* ``before``: rank ``b`` by cosine against ``a`` (query as-is);
* ``after`` : rank ``b`` by cosine against ``f(t)`` where
  ``f = to_retriever_space(T_r(refine(.)))``;
* the query's own vector is masked out, otherwise the identity transform wins
  trivially (cos(a, a) = 1);
* with ``filter_others`` every *other* gold answer of the same relation is
  masked too (the standard "filtered" protocol): an unlabelled but correct
  synonym must not be counted as a miss.

``delta_recall@k = after - before`` is the number to report; a relation whose
transform does not raise it is not learning anything useful, however good its
geometry metrics look.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

NEG_INF = -1e30


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x, p=2, dim=-1)


def rank_of_target(
    scores: torch.Tensor,          # [B, N] similarity, higher is better
    target_idx: torch.Tensor,      # [B] column holding the gold answer
    mask: Optional[torch.Tensor] = None,   # [B, N] bool, True = excluded
    tie_break: str = "optimistic",
) -> torch.Tensor:
    """1-based rank of the gold column among the non-excluded columns.

    ``optimistic`` counts a tie as "not beaten" (rank = #strictly greater + 1),
    ``pessimistic`` counts every equal score as beating the target.  Ties are
    common with normalised vectors, so the convention is part of the number.
    """
    if scores.dim() != 2:
        raise ValueError(f"scores must be 2-D, got {scores.dim()} dims")
    B = scores.shape[0]
    rows = torch.arange(B, device=scores.device)
    target = scores[rows, target_idx].unsqueeze(-1)
    if tie_break == "optimistic":
        better = scores > target
    elif tie_break == "pessimistic":
        better = scores >= target
        # the gold column always ties itself and ``>=`` counts that tie; rank
        # is a count of *other* columns, so the gold is dropped here and the
        # "+1" below adds it back
        better[torch.arange(B, device=scores.device), target_idx] = False
    else:
        raise ValueError(f"unknown tie_break={tie_break!r}")
    if mask is not None:
        better = better & ~mask
    return better.sum(dim=-1) + 1


def recall_at_k(ranks: torch.Tensor, k: int) -> float:
    return float((ranks <= k).float().mean().item()) if ranks.numel() else float("nan")


def mean_reciprocal_rank(ranks: torch.Tensor) -> float:
    if ranks.numel() == 0:
        return float("nan")
    return float((1.0 / ranks.float()).mean().item())


@dataclass
class QRResult:
    """Outcome of one query-rewriting evaluation."""

    relation: str
    chain: Tuple[int, ...]
    n_queries: int
    ks: Tuple[int, ...]
    before: Dict[int, float] = field(default_factory=dict)
    after: Dict[int, float] = field(default_factory=dict)
    mrr_before: float = float("nan")
    mrr_after: float = float("nan")
    applicability: Optional[torch.Tensor] = None   # [B] when a head is available
    ranks_after: Optional[torch.Tensor] = None      # [B]

    def delta(self, k: int) -> float:
        a, b = self.after.get(k), self.before.get(k)
        if a is None or b is None:
            return float("nan")
        return a - b

    def as_dict(self) -> Dict[str, float]:
        out: Dict[str, float] = {
            "relation": self.relation,
            "n_queries": float(self.n_queries),
            "mrr_before": self.mrr_before,
            "mrr_after": self.mrr_after,
        }
        for k in self.ks:
            out[f"recall_before@{k}"] = self.before.get(k, float("nan"))
            out[f"recall_after@{k}"] = self.after.get(k, float("nan"))
            out[f"delta_recall@{k}"] = self.delta(k)
        return out

    def table(self) -> str:
        head = f"  {'metric':<14}" + " ".join(f"@{k:>4}" for k in self.ks)
        lines = [
            f"  relation {self.relation} (chain={list(self.chain)}, "
            f"n={self.n_queries})",
            head,
            f"  {'recall before':<14}" + " ".join(
                f"{self.before.get(k, float('nan')):>4.2f}" for k in self.ks),
            f"  {'recall after':<14}" + " ".join(
                f"{self.after.get(k, float('nan')):>4.2f}" for k in self.ks),
            f"  {'delta':<14}" + " ".join(
                f"{self.delta(k):>4.2f}" for k in self.ks),
            f"  {'MRR':<14} before={self.mrr_before:.3f} after={self.mrr_after:.3f}",
        ]
        return "\n".join(lines)


def gold_pairs(
    edges: Sequence[Tuple[int, int, int]],
    rel_id: int,
    max_pairs: Optional[int] = None,
    rng: Optional[torch.Generator] = None,
) -> List[Tuple[int, int]]:
    """Gold ``(query, answer)`` pairs of one relation from an edge list."""
    pairs = [(s, d) for (s, d, r) in edges if int(r) == int(rel_id)]
    if max_pairs is not None and len(pairs) > max_pairs:
        perm = torch.randperm(len(pairs), generator=rng)[:max_pairs].tolist()
        pairs = [pairs[i] for i in perm]
    return pairs


def _build_mask(
    n_index: int,
    queries: torch.Tensor,        # [B] index positions of the query vectors
    answers: torch.Tensor,        # [B] index positions of the gold answers
    other_answers: Optional[torch.Tensor],  # [B, M] other golds of same relation
    exclude_self: bool,
) -> Optional[torch.Tensor]:
    B = queries.shape[0]
    mask = torch.zeros(B, n_index, dtype=torch.bool, device=queries.device)
    if exclude_self:
        mask[torch.arange(B), queries] = True
    if other_answers is not None:
        mask.scatter_(1, other_answers, True)
        mask[torch.arange(B), answers] = False  # never mask the gold itself
    return mask


def _other_golds(
    pairs: Sequence[Tuple[int, int]],
    answers: torch.Tensor,
    by_query: Optional[Mapping[int, Iterable[int]]] = None,
) -> Optional[torch.Tensor]:
    """For each query, the *other* answers of the same relation (filtered protocol).

    ``by_query`` is the gold answer table of the *whole* graph.  Without it the
    other answers are whatever else the sample happened to contain, and a query
    with 40 true answers that appears once in the probe gets no filtering at all
    -- every one of its other 39 correct answers is then counted as a miss.
    Callers that hold the graph pass the table and the metric stops depending
    on the sample.
    """
    if by_query is None:
        by_query = {}
        for q, a in pairs:
            by_query.setdefault(q, []).append(a)
    rows = []
    width = 0
    for q, a in [(q, a) for q, a in pairs]:
        others = [int(x) for x in by_query.get(q, ()) if int(x) != int(a)]
        rows.append(others)
        width = max(width, len(others))
    if width == 0:
        return None
    padded = torch.full((len(rows), width), 0, dtype=torch.long,
                        device=answers.device)
    for i, others in enumerate(rows):
        if others:
            padded[i, : len(others)] = torch.tensor(others, device=answers.device)
    return padded


@torch.no_grad()
def qr_recall(
    pairs: Sequence[Tuple[int, int]],
    index: torch.Tensor,
    transform: Callable[[torch.Tensor], torch.Tensor],
    ks: Sequence[int] = (1, 5, 10),
    exclude_self: bool = True,
    filter_others: bool = True,
    applicability: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    tie_break: str = "optimistic",
    relation: str = "?",
    chain: Tuple[int, ...] = (),
    other_map: Optional[Mapping[int, Iterable[int]]] = None,
) -> QRResult:
    """Recall before/after a query transform on gold pairs.

    ``index`` is the searchable corpus (the vector database contents); the
    pairs reference positions in it.  ``transform`` maps query vectors to
    rewritten queries, in the same space as ``index``.  ``other_map`` is the
    graph's answer table (``query -> every true answer``) used to filter
    correct-but-unlisted answers; without it only the answers inside ``pairs``
    are known, which understates the recall of high-multiplicity relations.
    """
    device = index.device
    if not pairs:
        return QRResult(relation=relation, chain=chain, n_queries=0, ks=tuple(ks))

    queries = torch.tensor([q for q, _ in pairs], dtype=torch.long, device=device)
    answers = torch.tensor([a for _, a in pairs], dtype=torch.long, device=device)

    other = _other_golds(pairs, answers, other_map) if filter_others else None
    mask = _build_mask(index.shape[0], queries, answers, other, exclude_self)

    q_raw = index[queries]
    q_new = transform(q_raw)

    S_before = _normalize(q_raw) @ _normalize(index).t()
    S_after = _normalize(q_new) @ _normalize(index).t()

    r_before = rank_of_target(S_before, answers, mask, tie_break=tie_break)
    r_after = rank_of_target(S_after, answers, mask, tie_break=tie_break)

    result = QRResult(
        relation=relation,
        chain=chain,
        n_queries=len(pairs),
        ks=tuple(ks),
        mrr_before=mean_reciprocal_rank(r_before),
        mrr_after=mean_reciprocal_rank(r_after),
        ranks_after=r_after,
    )
    for k in ks:
        result.before[int(k)] = recall_at_k(r_before, int(k))
        result.after[int(k)] = recall_at_k(r_after, int(k))
    if applicability is not None:
        result.applicability = applicability(q_raw)
    return result


@torch.no_grad()
def applicability_split(
    result: QRResult, scores: torch.Tensor, threshold: float = 0.5
) -> Dict[str, float]:
    """Recall after the transform, split by the applicability head.

    The head is useful only if the transforms it trusts actually rewrite the
    query better than the ones it rejects (PLAN.md 1.4).
    """
    if result.ranks_after is None or scores is None:
        return {}
    keep = scores >= threshold
    out: Dict[str, float] = {}
    k = max(result.ks) if result.ks else 1
    for name, sel in (("applicable", keep), ("rejected", ~keep)):
        sub = result.ranks_after[sel]
        out[f"recall_after@{k}_{name}"] = recall_at_k(sub, k)
        out[f"share_{name}"] = float(sel.float().mean().item())
    return out
