"""
Label-free consistency diagnostics (docs/IDEAS/12-consistency-diagnostics.md).

Two tests, neither of which needs an annotation:

* **12a** -- is a text close to its own negation?  For a document with claims
  ``c_1..c_m`` in an embedding space and a relation transport ``T_r`` acting
  in *that same space*::

      S(d)  = mean_k cos( T_ant(c_k), d )                    self-negation
      K(d)  = median_{i != j} cos(c_i, T_syn(c_j))
              - median_{i != j} cos(c_i, T_ant(c_j))          inner coherence
      HD(d) = || T_syn T_ant T_syn T_ant (d) - d || / ||d||  holonomy

  ``S`` and ``HD`` rise on self-contradicting text, ``K`` falls.
* **12b** -- does the model know that ``T_r T_s != T_s T_r``?  For every
  relation pair the *order separation*

      sep(r, s) = 1 - cos( T_s(T_r(x)), T_r(T_s(x)) )

  is measured and compared with the learned commutation matrix ``C_rs``:
  a trained non-commutative operator must separate the orders of the pairs
  ``C`` marks as incompatible, and ``sep`` must correlate with ``1 - C`` --
  otherwise ``C`` is noise and every idea built on it inherits that.

The only model-facing dependency is :class:`Transport`, and it is abstract:
the learned WIĄZKA operator (:class:`CheckpointTransport`) and the
embedder-free translation baseline (:class:`AdditiveTransport`) both
implement it, so the same bilingual fixtures run on any embedder and any
language, trained or not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

#: canonical relation names of the two directions 12a needs
REL_SYN = "synonimia"
REL_ANT = "antonimia właściwa"

#: a relation value of ``C_rs`` closer to 0.5 than this carries no evidence
C_NO_EVIDENCE_TOLERANCE = 0.01


def _unit(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1)


def _np(x: torch.Tensor) -> np.ndarray:
    """Host copy -- the numpy statistics below do not run on the GPU."""
    return x.detach().to("cpu", dtype=torch.float64).numpy()


def _off_diagonal(cos: torch.Tensor) -> torch.Tensor:
    """Copy of a similarity matrix with its diagonal masked as ``nan``."""
    out = cos.clone()
    out.fill_diagonal_(float("nan"))
    return out


def roc_auc(labels: Sequence[float], scores: Sequence[float]) -> float:
    """Mann-Whitney U / AUC: how well ``scores`` rank positives first."""
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    n_pos = int((labels > 0.5).sum())
    n_neg = int((labels < 0.5).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1.0, len(scores) + 1.0)
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):          # average the ranks of ties
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[i : j + 1] = ranks[i : j + 1].mean()
        i = j + 1
    rank_sum = ranks[labels > 0.5].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Rank correlation; ``nan`` when either side has no variance."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


# ------------------------------------------------------------------ transports
class Transport:
    """A relation transport acting in the space its vectors live in.

    Contract: ``apply`` takes ``[B, D]`` and returns ``[B, D]`` of the same
    space; whether vectors are unit-normalised does not matter, every metric
    here is a cosine or a normalised norm.
    """

    #: relation names this transport can apply, ordered by relation id
    relations: Tuple[str, ...] = ()

    def prepare(self, x: torch.Tensor) -> torch.Tensor:
        """Map vectors from the caller's space into the transport's space.

        Called once per vector set, never inside a chain: a chain is applied
        entirely in the transport's own space.  Identity for a transport that
        lives in the space it is given.
        """
        return x

    def apply(self, x: torch.Tensor, rel: str) -> torch.Tensor:
        raise NotImplementedError

    def apply_chain(self, x: torch.Tensor, chain: Sequence[str]) -> torch.Tensor:
        for rel in chain:
            x = self.apply(x, rel)
        return x

    def inverse(self, x: torch.Tensor, rel: str) -> torch.Tensor:
        """``T_r^{-1}``; a transport that cannot invert itself raises."""
        raise NotImplementedError(f"{type(self).__name__} has no inverse")

    def holonomy(self, x: torch.Tensor, loop: Sequence[str]) -> torch.Tensor:
        """Walk ``loop`` and undo it again; the identity means flat geometry."""
        moved = self.apply_chain(x, list(loop))
        for rel in reversed(list(loop)):
            moved = self.inverse(moved, rel)
        return moved

    def has(self, rel: str) -> bool:
        return rel in self.relations


class AdditiveTransport(Transport):
    """Embedder-free baseline transport: ``T_r(x) = x + strength * v_r``.

    ``v_r`` is the mean difference vector of a seed lexicon of ``r`` pairs, so
    nothing but an embedding function is required -- which is what makes the
    diagnostics runnable on any embedder and any language.  The family is
    *commutative by construction* (``v_r + v_s == v_s + v_r``), it is its own
    exact inverse, and its closed loops close exactly: order separation and
    holonomy are **0**, not "small".  That makes it the floor 12b measures
    against -- a learned operator that fails to beat it carries no structure a
    translation vector could not express.

    ``strength`` is an absolute step length in the units of the input vectors,
    so with unit-norm embeddings the default moves a vector by about 90 degrees
    of a right angle; scale it down for a gentler baseline.
    """

    def __init__(self, vectors: Mapping[str, torch.Tensor],
                 relations: Sequence[str], strength: float = 1.0):
        self._v = {rel: _unit(vectors[rel]) for rel in relations}
        self.strength = float(strength)
        self.relations = tuple(relations)

    @classmethod
    def from_pairs(
        cls,
        pairs: Mapping[str, Sequence[Sequence[str]]],
        embed: Callable[[Sequence[str]], torch.Tensor],
        relations: Optional[Sequence[str]] = None,
        strength: float = 1.0,
    ) -> "AdditiveTransport":
        """``pairs[rel] = [[a, b], ...]`` with ``T_r(a) ~= b``; ``embed`` embeds texts."""
        rels = list(relations) if relations else list(pairs)
        vectors: Dict[str, torch.Tensor] = {}
        for rel in rels:
            rows = list(pairs[rel])
            if not rows:
                raise ValueError(f"no seed pairs for relation {rel!r}")
            texts = [t for pair in rows for t in pair[:2]]
            emb = embed(texts)
            diffs = (emb[1::2] - emb[0::2]).mean(dim=0)
            vectors[rel] = diffs
        return cls(vectors, rels, strength=strength)

    def _step(self, x: torch.Tensor, rel: str, sign: float) -> torch.Tensor:
        # deliberately independent of ||x||: a translation has to be its own
        # exact inverse in the other direction, otherwise "commutative
        # baseline" would not mean separation *zero* but separation *small*.
        return x + sign * self.strength * self._v[rel].to(x.device, dtype=x.dtype)

    def apply(self, x: torch.Tensor, rel: str) -> torch.Tensor:
        return self._step(x, rel, +1.0)

    def inverse(self, x: torch.Tensor, rel: str) -> torch.Tensor:
        return self._step(x, rel, -1.0)


class CheckpointTransport(Transport):
    """The learned WIĄZKA operator of a checkpoint.

    Input vectors must live in the *base* (retriever) space of the checkpoint;
    :meth:`prepare` pushes them through the chart map ``base_to_op`` once, and
    every :meth:`apply` afterwards works inside the operator chart -- charting
    per step would compose ``base_to_op`` into every link of a chain and turn
    a two-step transport into a projection applied twice.
    """

    def __init__(self, model, rel2idx: Mapping[str, int], device: str = "cpu"):
        self.model = model
        self.device = torch.device(device)
        # ``relations`` is ordered by relation id: the commutation matrix is
        # indexed by id, and order_separation() pairs them up positionally.
        self.relations = tuple(
            name for name, _ in sorted(rel2idx.items(), key=lambda kv: int(kv[1]))
        )
        self.rel2idx = {str(k): int(v) for k, v in rel2idx.items()}

    @torch.no_grad()
    def prepare(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.base_to_op(x.to(self.device))

    @torch.no_grad()
    def apply(self, x: torch.Tensor, rel: str) -> torch.Tensor:
        return self.model.apply(
            x, torch.tensor([self.rel2idx[rel]], device=x.device))

    @torch.no_grad()
    def inverse(self, x: torch.Tensor, rel: str) -> torch.Tensor:
        return self.model.inverse(
            x, torch.tensor([self.rel2idx[rel]], device=x.device))

    @torch.no_grad()
    def holonomy(self, x: torch.Tensor, loop: Sequence[str]) -> torch.Tensor:
        """Walk ``loop``, then undo it with each relation's *unmodulated* inverse.

        With a base-point-dependent connection the honest way to state a
        loop defect is to come back along the relation the chart carries at
        its origin, not along the one the connection has moved to at the far
        end: undoing at the recorded base point closes the loop by construction
        (measured: exactly ``0``) and says nothing about curvature.  Inverting
        unmodulated leaves precisely the context-dependent part of the
        transport -- the same quantity the curvature loss penalises through
        ``|| H(x_A) - x_A ||``, and the reason ``C_rs`` exists at all.
        """
        moved = self.apply_chain(x, list(loop))
        for rel in reversed(list(loop)):
            moved = self.inverse(moved, rel)
        return moved


# ------------------------------------------------------------------------ 12a
def document_scores(
    claims: torch.Tensor,
    transport: Transport,
    rel_syn: str = REL_SYN,
    rel_ant: str = REL_ANT,
    negated: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """The 12a scores of one document, from its claim matrix ``[m, D]``.

    ``negated`` optionally carries the surface-negated claims (``"nie " + c``
    in Polish, ``"not " + c`` in English): their mean cosine with the document
    is the cheap negation-heuristic baseline the spec asks to be beaten.
    """
    claims = _unit(transport.prepare(claims))
    m = int(claims.shape[0])
    if m < 2:
        raise ValueError("a document needs at least two claims")
    doc = _unit(claims.mean(dim=0, keepdim=True)).squeeze(0)

    ant = transport.apply(claims, rel_ant)
    syn = transport.apply(claims, rel_syn)

    cos_syn = _off_diagonal(claims @ syn.t())
    cos_ant = _off_diagonal(claims @ ant.t())
    cos_claims = _off_diagonal(claims @ claims.t())
    return {
        "self_negation": float((ant @ doc).mean()),
        "coherence_k": float(np.nanmedian(_np(cos_syn)) - np.nanmedian(_np(cos_ant))),
        "holonomy": relative_holonomy(transport, doc, [rel_syn, rel_ant] * 2),
        "raw_incoherence": float(1.0 - np.nanmean(_np(cos_claims))),
        "negation_baseline": (
            float((_unit(transport.prepare(negated)) @ doc).mean())
            if negated is not None else float("nan")
        ),
    }


def relative_holonomy(
    transport: Transport,
    x: torch.Tensor,
    loop: Sequence[str],
) -> float:
    """Defect of a closed transport loop, as ``1 - cos(H(x), x)``.

    Accepts one vector ``[D]`` or a batch ``[B, D]`` and averages over it.

    ``loop`` is traversed forwards and then undone with the transport's own
    inverses, so a transport that composed to the identity around the cycle
    scores exactly 0::

        H = T_{loop[-1]}^-1 ... T_{loop[0]}^-1 T_{loop[0]} ... T_{loop[-1]}

    ``1 - cos`` rather than the relative norm, because cosine keeps the metric
    comparable across spaces whose vectors are not unit-norm (the base space of
    Słowosieć is not) and across transports that carry a scale.
    """
    x = _unit(x.unsqueeze(0) if x.dim() == 1 else x)
    moved = transport.holonomy(x, list(loop))
    return float((1.0 - F.cosine_similarity(moved, x, dim=-1)).mean())


def pair_contradiction_scores(
    claims: torch.Tensor,
    transport: Transport,
    gold_pairs: Sequence[Sequence[int]],
    rel_ant: str = REL_ANT,
) -> Dict[str, float]:
    """Does ``T_ant`` bring genuinely contradicting claims together?

    ``gold_pairs`` holds claim-index pairs that contradict by construction.
    A pair is scored ``0.5 * (cos(T_ant c_i, c_j) + cos(T_ant c_j, c_i))`` and
    the gold pairs are separated from the rest by :func:`roc_auc`.
    """
    claims = _unit(transport.prepare(claims))
    m = int(claims.shape[0])
    ant = transport.apply(claims, rel_ant)
    score = _off_diagonal(0.5 * (ant @ claims.t() + claims @ ant.t()))
    gold = set()
    for a, b in gold_pairs:
        gold |= {(int(a), int(b)), (int(b), int(a))}
    labels: List[float] = []
    vals: List[float] = []
    for i in range(m):
        for j in range(i + 1, m):
            labels.append(1.0 if (i, j) in gold else 0.0)
            vals.append(float(score[i, j]))
    pos = [v for v, l in zip(vals, labels) if l > 0.5]
    neg = [v for v, l in zip(vals, labels) if l < 0.5]
    return {
        "auc": roc_auc(labels, vals),
        "n_gold": len(pos),
        "n_pairs": len(vals),
        "mean_gold": float(np.mean(pos)) if pos else float("nan"),
        "mean_other": float(np.mean(neg)) if neg else float("nan"),
    }


# ------------------------------------------------------------------------ 12b
@dataclass
class OrderSeparationReport:
    """Measured ``sep(r, s)`` over relation pairs, against the learned ``C``."""

    n_pairs: int
    n_vectors: int
    sep_min: float
    sep_median: float
    sep_max: float
    corr_sep_c: float                 # spearman(sep, C)
    corr_sep_1mc: float               # spearman(sep, 1 - C)
    n_c_evidence: int                 # pairs whose C left the 0.5 prior
    c_low: int                        # C below the "no evidence" band
    c_high: int                       # C above it
    sep_c_low: float                  # mean sep where C is low
    sep_c_high: float                 # mean sep where C is high
    c_min: float
    c_max: float
    strongest: List[Dict[str, float]] = field(default_factory=list)
    weakest: List[Dict[str, float]] = field(default_factory=list)


def order_separation(
    x: torch.Tensor,
    transport: Transport,
    commutation: np.ndarray,
    top_k: int = 5,
    relation_subset: Optional[Sequence[str]] = None,
) -> OrderSeparationReport:
    """12b: order separation of every relation pair, compared with ``C_rs``.

    ``commutation`` is indexed by relation id, so ``transport.relations`` has
    to be id-ordered (:class:`CheckpointTransport` guarantees it).  Only the
    upper triangle is computed; ``relation_subset`` restricts the pairs to a
    semantically meaningful slice (e.g. the lexical relations only).
    """
    names = list(transport.relations)
    x = transport.prepare(x)
    if relation_subset:
        keep = {r for r in relation_subset if r in names}
        if len(keep) < 2:
            raise ValueError("relation_subset leaves fewer than two relations")
        idx = [i for i, n in enumerate(names) if n in keep]
    else:
        idx = list(range(len(names)))

    seps: List[float] = []
    cs: List[float] = []
    pairs: List[Tuple[str, str]] = []
    with torch.no_grad():
        for a_pos, a in enumerate(idx):
            for b in idx[a_pos + 1:]:
                u = transport.apply_chain(x, [names[b], names[a]])
                w = transport.apply_chain(x, [names[a], names[b]])
                seps.append(float(1.0 - F.cosine_similarity(u, w, dim=-1).mean()))
                cs.append(float(commutation[a, b]))
                pairs.append((names[a], names[b]))

    s = np.asarray(seps)
    c = np.asarray(cs)
    tol = C_NO_EVIDENCE_TOLERANCE
    low = c < 0.5 - tol
    high = c > 0.5 + tol
    order = np.argsort(s)

    def row(i: int) -> Dict[str, float]:
        return {"a": pairs[i][0], "b": pairs[i][1],
                "sep": float(s[i]), "c": float(c[i])}

    return OrderSeparationReport(
        n_pairs=int(s.size),
        n_vectors=int(x.shape[0]),
        sep_min=float(s.min()),
        sep_median=float(np.median(s)),
        sep_max=float(s.max()),
        corr_sep_c=spearman(s, c),
        corr_sep_1mc=spearman(s, 1.0 - c),
        n_c_evidence=int((low | high).sum()),
        c_low=int(low.sum()),
        c_high=int(high.sum()),
        sep_c_low=float(s[low].mean()) if low.any() else float("nan"),
        sep_c_high=float(s[high].mean()) if high.any() else float("nan"),
        c_min=float(c.min()),
        c_max=float(c.max()),
        strongest=[row(i) for i in order[-top_k:][::-1]],
        weakest=[row(i) for i in order[:top_k]],
    )


def order_description_separation(
    desc_rs: torch.Tensor,
    desc_sr: torch.Tensor,
    target_rs: Optional[torch.Tensor] = None,
    target_sr: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Language side of 12b: can the *space* tell the two orders apart?

    ``desc_rs`` / ``desc_sr`` are embeddings of two descriptions of the same
    chain read in the two orders (``r`` then ``s``, and ``s`` then ``r``).
    ``desc_separation`` is how far apart they land; if that is ~0 no model
    reading this space can ever pick the right one.  When the two target
    vectors are given in the same space, ``choice_accuracy`` is how often a
    description points at its own target -- coin flip for a commutative
    transport by construction, and the number to beat with a learned one.
    """
    desc_rs = _unit(desc_rs)
    desc_sr = _unit(desc_sr)
    if desc_rs.shape != desc_sr.shape:
        raise ValueError("desc_rs and desc_sr must have the same shape")
    out = {
        "desc_separation": float(1.0 - (desc_rs * desc_sr).sum(dim=-1).mean()),
    }
    if target_rs is not None and target_sr is not None:
        target_rs = _unit(target_rs)
        target_sr = _unit(target_sr)
        hit_rs = (desc_rs * target_rs).sum(-1) > (desc_rs * target_sr).sum(-1)
        hit_sr = (desc_sr * target_sr).sum(-1) > (desc_sr * target_rs).sum(-1)
        out["choice_accuracy"] = float(0.5 * (hit_rs.float().mean() + hit_sr.float().mean()))
    return out
