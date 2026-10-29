"""
Three-state uncertainty: typed semantic (orbit) entropy, ideas/04.

Binary semantic entropy samples ``k`` answers, clusters them as "same / not
the same" and calls the entropy of the clusters uncertainty.  That conflates
two different things and, worse, hides a third:

* ``k`` answers in **one** ``syn`` orbit  -- the model is sure and talkative →
  **answer**;
* answers split across orbits joined by an ``ant`` edge -- it sees two
  incompatible states of the world → **refuse, with the pair as the reason**;
* answers whose relations are **not defined at all** (low ``applicability``)
  -- it is describing something the relations do not cover → **out of scope**,
  which is not "I do not know" and must not be reported as such.

Three signals, one per state, so a decision can be *explained* by a fact
rather than by a threshold:

``H_orb``  entropy over the ``syn`` orbits      -- spread over meanings
``kappa``  max confidence on an ``ant`` edge    -- real conflict, not variety
``alpha``  mean ``applicability``               -- coverage

The decisive test of section 4 of the document is implemented here as
:func:`verbose_vs_conflict`: on **confident but verbose** answer sets (long
paraphrases of one answer) a cosine-threshold clustering raises uncertainty,
while the typed orbit partition must not.  Sets are built gold by construction
out of the ideas/12 fixture: coherent documents are verbose agreement,
self-contradicting ones are conflict, and claims spliced from unrelated topics
are out of scope.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .clustering import candidate_pairs

#: the three states of a set of answers
ANSWER = "answer"
REFUSE = "refuse"
OUT_OF_SCOPE = "out_of_scope"


def orbit_labels(n: int, rel_id: np.ndarray, conf: np.ndarray,
                 pair_rows: np.ndarray, pair_cols: np.ndarray,
                 syn_ids: Sequence[int], tau: float = 0.5) -> np.ndarray:
    """Partition ``n`` answers into ``syn`` orbits (connected components).

    Two answers are in the same orbit when the typer calls the pair ``syn``
    with confidence ``>= tau``.  Transitivity comes from the components, not
    from the typer, so an intransitive triple lands in one orbit rather than
    being reported as two contradictions.
    """
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    syn = {int(r) for r in syn_ids}
    for k, (i, j) in enumerate(zip(pair_rows.tolist(), pair_cols.tolist())):
        if int(rel_id[k]) in syn and float(conf[k]) >= tau:
            ri, rj = find(int(i)), find(int(j))
            if ri != rj:
                parent[ri] = rj
    mapping: Dict[int, int] = {}
    codes: List[int] = []
    for i in range(n):
        r = find(i)
        if r not in mapping:
            mapping[r] = len(mapping)
        codes.append(mapping[r])
    return np.asarray(codes, dtype=np.int64)


def orbit_entropy(labels: np.ndarray) -> float:
    """Shannon entropy (nats) of the orbit distribution; 0.0 for one orbit."""
    labels = np.asarray(labels)
    if labels.size == 0:
        return 0.0
    counts = np.bincount(labels)
    p = counts[counts > 0].astype(np.float64)
    p /= p.sum()
    return float(-(p * np.log(p)).sum())


def threshold_entropy(vectors: np.ndarray,
                      threshold: Optional[float] = None) -> float:
    """The binary baseline: entropy after clustering on a cosine threshold.

    This is what semantic entropy computes when "same meaning" is decided by
    distance.  It is the baseline :func:`verbose_vs_conflict` exists to beat.

    ``threshold`` defaults to a high quantile of the set's own off-diagonal
    cosines: a fixed cut is meaningless across spaces (this one sits at a mean
    pairwise cosine of 0.946, so a 0.85 cut merges every set into one cluster
    and the baseline scores a flat 0.0 without measuring anything).
    """
    from .clustering import cosine_threshold_components
    if threshold is None:
        threshold = auto_cosine_threshold(vectors)
    return orbit_entropy(cosine_threshold_components(vectors, threshold))


def auto_cosine_threshold(vectors: np.ndarray, quantile: float = 0.95) -> float:
    """A cosine cut at a ``quantile`` of a *corpus's* pair similarities.

    Call this on the whole corpus, never on one answer set: a set of two
    answers has one off-diagonal cosine, and the quantile of one number is
    that number, so the threshold always merges the pair and the baseline
    scores a flat 0.0 without measuring anything.
    """
    sim = _cos(np.asarray(vectors, dtype=np.float32))
    n = int(sim.shape[0])
    if n < 3:
        return float("nan")
    i, j = np.triu_indices(n, 1)
    return float(np.quantile(sim[i, j], quantile))


def set_signals(vectors: np.ndarray, typer, syn_ids: Sequence[int],
                ant_ids: Sequence[int], tau_syn: float = 0.0,
                top_k: int = 12) -> Dict[str, float]:
    """``H_orb``, ``kappa`` and ``alpha`` for one set of answer vectors."""
    vectors = np.asarray(vectors, dtype=np.float32)
    n = int(vectors.shape[0])
    if n < 2:
        raise ValueError("an answer set needs at least two answers")
    rows, cols = candidate_pairs(_cos(vectors), top_k=n - 1, min_cos=-np.inf)
    a = torch.as_tensor(vectors[rows], dtype=torch.float32)
    b = torch.as_tensor(vectors[cols], dtype=torch.float32)
    rel, conf, appl = typer.type_pairs(a, b)

    labels = orbit_labels(n, rel, conf, rows, cols, syn_ids, tau=tau_syn)
    ant = {int(r) for r in ant_ids}
    on_ant = np.isin(rel, list(ant))
    kappa = float(conf[on_ant].max()) if on_ant.any() else 0.0
    return {
        "H_orb": orbit_entropy(labels),
        "kappa": kappa,
        "alpha": float(np.mean(appl)) if len(appl) else float("nan"),
        "orbits": int(len(set(labels.tolist()))),
        "n": n,
    }


def _cos(x: np.ndarray) -> np.ndarray:
    """Cosine matrix of one answer set; every metric here is a cosine."""
    from .clustering import cosine_matrix
    return cosine_matrix(x)


def decide(signals: Dict[str, float], tau_kappa: float = 0.5,
           tau_alpha: float = 0.5, tau_H: float = 0.5) -> str:
    """The three-way decision, coverage first.

    Order matters: an uncovered set can show a spurious conflict (the typer
    forced a relation where none is defined), and reporting that as "these
    answers contradict" is exactly the error the idea document is about.
    """
    if signals.get("alpha", 1.0) < tau_alpha:
        return OUT_OF_SCOPE
    if signals.get("kappa", 0.0) >= tau_kappa:
        return REFUSE
    if signals.get("H_orb", 0.0) >= tau_H:
        return REFUSE
    return ANSWER


def verbose_vs_conflict(sets: Sequence[np.ndarray], gold: Sequence[str],
                        typer, syn_ids: Sequence[int], ant_ids: Sequence[int],
                        threshold: Optional[float] = None,
                        tau_syn: float = 0.0) -> Dict[str, float]:
    """Section 4's decisive test: verbose agreement must not look uncertain.

    ``gold`` labels every set ``verbose`` (one meaning, several wordings),
    ``conflict`` or ``out_of_scope``.  Reported per signal:

    ``typed_gap``   mean ``H_orb``(conflict) − mean ``H_orb``(verbose);
    ``binary_gap``  the same for the cosine-threshold entropy;

    and the coverage signal's separation of ``out_of_scope`` from the rest.
    The claim of the idea is ``typed_gap > 0`` while the binary baseline
    pays for verbosity with fake uncertainty (``binary_gap`` smaller, or the
    verbose entropy itself high).  A method that only wins by moving a
    threshold cannot fake this, because both numbers are gaps, not scores.
    """
    from .consistency import roc_auc

    rows: List[Dict[str, float]] = []
    for vecs, label in zip(sets, gold):
        s = set_signals(vecs, typer, syn_ids, ant_ids, tau_syn=tau_syn)
        s["binary_H"] = threshold_entropy(vecs, threshold)
        s["gold"] = label
        rows.append(s)

    def mean_of(label: str, key: str) -> float:
        v = [r[key] for r in rows if r["gold"] == label]
        return float(np.mean(v)) if v else float("nan")

    is_conflict = [1.0 if r["gold"] == "conflict" else 0.0 for r in rows]
    is_scope = [1.0 if r["gold"] == "out_of_scope" else 0.0 for r in rows]
    verbose_present = any(r["gold"] == "verbose" for r in rows)
    conflict_present = any(r["gold"] == "conflict" for r in rows)

    out: Dict[str, float] = {
        "n_sets": len(rows),
        "H_orb_verbose": mean_of("verbose", "H_orb"),
        "H_orb_conflict": mean_of("conflict", "H_orb"),
        "H_binary_verbose": mean_of("verbose", "binary_H"),
        "H_binary_conflict": mean_of("conflict", "binary_H"),
        "kappa_conflict": mean_of("conflict", "kappa"),
        "kappa_verbose": mean_of("verbose", "kappa"),
        "alpha_out_of_scope": mean_of("out_of_scope", "alpha"),
        "alpha_in_scope": float(np.mean([r["alpha"] for r in rows
                                         if r["gold"] != "out_of_scope"]))
        if any(r["gold"] != "out_of_scope" for r in rows) else float("nan"),
    }
    out["typed_gap"] = out["H_orb_conflict"] - out["H_orb_verbose"]
    out["binary_gap"] = out["H_binary_conflict"] - out["H_binary_verbose"]
    if verbose_present and conflict_present and len(set(is_conflict)) > 1:
        out["auc_typed"] = roc_auc(is_conflict, [r["H_orb"] for r in rows])
        out["auc_binary"] = roc_auc(is_conflict, [r["binary_H"] for r in rows])
        out["auc_kappa"] = roc_auc(is_conflict, [r["kappa"] for r in rows])
    if len(set(is_scope)) > 1:
        # coverage is the one signal the binary family does not have at all
        out["auc_coverage"] = roc_auc(is_scope, [-r["alpha"] for r in rows])
    return out
