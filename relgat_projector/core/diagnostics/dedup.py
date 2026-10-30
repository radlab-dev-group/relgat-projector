"""
Contradiction-preserving deduplication (docs/IDEAS/02-relation-typed-dedup.md).

Semantic dedup (SemDeDup and the rest of the family) embeds, thresholds cosine
and keeps one representative per neighbourhood.  The threshold cannot tell a
paraphrase from a negation -- "the drug works" and "the drug does not work"
share topic, register and skeleton -- so the second sentence is deleted and the
corpus is left poor in exactly the contrast that teaches a model what negation
means.  The idea: make keep/drop depend on the **type of relation** between two
vectors, not on the value of their similarity.

| edge        | rule                                                     |
|-------------|----------------------------------------------------------|
| ``syn``     | one representative (the orbit is an equivalence class)   |
| ``ant``     | **keep both** -- a contrast pair, not a duplicate         |
| undefined   | keep both (different facts)                              |

Two numbers decide whether that is worth anything, and both are measurable on
the clustering fixture without a single annotation:

* **reduction** -- share of items dropped.  A rule that keeps everything is not
  a dedup, so this is the cost axis, and every method is reported at its own
  reduction so the two are never read apart;
* **conflict recall** -- share of gold ``ant`` pairs on which *both* members
  survived.  This is what a cosine threshold destroys, and it is why reduction
  alone is never a result.

The go/no-go of section 7 of the document is here too:
:func:`high_cosine_contradiction_share` measures what share of the pairs above
a cosine threshold are contradictions at all.  If it is ~0 the threshold is
fine and the idea has no economic case; the fixture's answer is in the report.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .clustering import candidate_pairs, cosine_matrix


def cosine_dedup(vectors: np.ndarray, threshold: float,
                 order: Optional[Sequence[int]] = None) -> np.ndarray:
    """SemDeDup-style dedup: drop an item that is this close to a kept item.

    Greedy in ``order`` (default: input order, i.e. what a pipeline streaming a
    corpus does).  Returns a boolean mask of the survivors.
    """
    sim = cosine_matrix(vectors)
    n = int(sim.shape[0])
    keep = np.ones(n, dtype=bool)
    for i in (range(n) if order is None else order):
        if not keep[i]:
            continue
        for j in np.flatnonzero(sim[i] >= float(threshold)):
            if int(j) != int(i):
                keep[j] = False
    return keep


def typed_dedup(vectors: np.ndarray, typer, syn_ids: Sequence[int],
                tau_conf: float = 0.5, tau_appl: float = 0.0,
                candidate_top_k: int = 8,
                order: Optional[Sequence[int]] = None
                ) -> Tuple[np.ndarray, List[Tuple[int, int, int, float]]]:
    """Relation-typed dedup: merge only the pairs the model calls ``syn``.

    ``typer`` is the same :class:`~.clustering.EdgeTyper` the clustering report
    uses, so the learned decoder, the head-free transport and the embedder-free
    baseline are all measured through one call.  Returns ``(keep, merges)``.

    Only ``syn`` edges merge.  An ``ant`` edge is deliberately *not* a rule of
    its own -- "keep both" is the default when nothing merges -- which is what
    makes the comparison with a threshold fair: both rules can only ever drop
    something, and the typed one drops strictly less on conflicts.
    """
    vectors = np.asarray(vectors)
    n = int(vectors.shape[0])
    rows, cols = candidate_pairs(cosine_matrix(vectors),
                                 top_k=int(candidate_top_k), min_cos=-np.inf)
    syn = {int(r) for r in syn_ids}
    merges: List[Tuple[int, int, int, float]] = []
    if len(rows):
        a = torch.as_tensor(vectors[rows], dtype=torch.float32)
        b = torch.as_tensor(vectors[cols], dtype=torch.float32)
        rel, conf, appl = typer.type_pairs(a, b)
        wanted = np.flatnonzero((np.isin(rel, list(syn)))
                                & (conf >= tau_conf)
                                & (appl >= tau_appl))
        merges = [(int(rows[m]), int(cols[m]), int(rel[m]), float(conf[m]))
                  for m in wanted]

    position = {v: k for k, v in enumerate(range(n) if order is None else order)}
    keep = np.ones(n, dtype=bool)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j, r, c in merges:
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        # the survivor is the earlier item in corpus order: deterministic, and
        # it does not prefer the most generic variant of the orbit
        drop, stay = ((ri, rj) if position[ri] > position[rj]
                      else (rj, ri))
        parent[drop] = stay
        keep[drop] = False
    return keep, merges


def dedup_metrics(keep: np.ndarray, ant_pairs: Sequence[Tuple[int, int]],
                  syn_pairs: Sequence[Tuple[int, int]]) -> Dict[str, float]:
    """Cost (reduction) and benefit (conflict recall) of one keep-mask."""
    n = int(keep.size)
    ant_kept = sum(1 for i, j in ant_pairs if keep[i] and keep[j])
    syn_kept = sum(1 for i, j in syn_pairs if keep[i] and keep[j])
    return {
        "kept": int(keep.sum()),
        "reduction": 1.0 - float(keep.sum()) / max(1, n),
        # the metric the idea is about: contradictions have to survive
        "conflict_recall": (ant_kept / len(ant_pairs)) if ant_pairs else float("nan"),
        # and duplicates must NOT survive -- 1.0 here means nothing was merged
        "duplicate_recall": (syn_kept / len(syn_pairs)) if syn_pairs else float("nan"),
        "ant_pairs": int(len(ant_pairs)),
        "syn_pairs": int(len(syn_pairs)),
    }


def high_cosine_contradiction_share(
        vectors: np.ndarray, ant_pairs: Sequence[Tuple[int, int]],
        syn_pairs: Sequence[Tuple[int, int]] = (),
        threshold: float = 0.9) -> Dict[str, float]:
    """Section 7 of idea 02: are the near-duplicates actually contradictions?

    Over the pairs above ``threshold``: what share are gold contradictions and
    what share gold paraphrases.  The economic argument of the whole document
    rests on the first number being well above zero.
    """
    sim = cosine_matrix(vectors)
    n = int(sim.shape[0])
    i, j = np.triu_indices(n, 1)
    above = sim[i, j] >= float(threshold)
    pairs = list(zip(i[above].tolist(), j[above].tolist()))
    ant, syn = set(ant_pairs), set(syn_pairs)
    return {
        "threshold": float(threshold),
        "pairs_above_threshold": int(len(pairs)),
        "contradiction_share": (sum(1 for p in pairs if p in ant) / len(pairs)
                                if pairs else float("nan")),
        "paraphrase_share": (sum(1 for p in pairs if p in syn) / len(pairs)
                             if pairs else float("nan")),
    }
