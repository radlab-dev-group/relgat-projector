"""
Reversible semantic pseudonymisation (docs/IDEAS/06-reversible-pseudonymization.md).

We publish `v~ = T_r(lambda) v` instead of `v`; the relation picks the
*direction* of generalisation, `lambda` the depth, and the exact inverse
`T_r(lambda)^-1` (key = `theta_r` + `log_scale_r`) recovers the original.
The claim to be tested is a curve, not a point: privacy must rise with
`lambda` while usefulness (cosine structure, and therefore MRR of a
retrieval task on the published set) stays flat -- and the recovery has to
be *exact*, which no lossy noise scheme can promise.

Because no inversion decoder is available here, the privacy axis is the
standard cheap proxy the threat model of the document allows: a
**linkage attack** in which the attacker, who knows the original corpus
but not the key, has to match each published vector to its origin.  Two
numbers:

* ``linkage_auc`` -- AUC of "is candidate i the origin of protected i?":
  the correct match's rank among all candidates.  0.5 = the attack is
  blind; 1.0 = the attack recovers every identity.
* ``k_anon_median`` -- median number of corpus vectors within cosine
  epsilon of a published vector: the semantic k-anonymity of the scheme.

The control of the document, built in: **the same grid under a random
key** (a fixed random rotation with the same norm profile, no
semantics).  Whatever the random key achieves, the semantic key must
beat it on usefulness *or* the scheme's value is "it is reversible",
nothing more.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .consistency import roc_auc


def _unit(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1)


def _np(x: torch.Tensor) -> np.ndarray:
    return x.detach().to("cpu", dtype=torch.float64).numpy()


def linkage_auc(corpus: torch.Tensor, protected: torch.Tensor,
                top_k: int = 512) -> float:
    """AUC of the linkage attack: rank of the true origin among candidates.

    ``protected[i]`` was produced from ``corpus[i]``.  The attacker sees
    both sets; for each protected vector we take the ``top_k`` nearest
    corpus vectors by cosine and ask whether the true origin is among
    them (label 1) -- scored against the other candidates (label 0).
    ``top_k = 1`` is the strictest attack (exact identity).
    """
    c = _unit(corpus.float()).cpu()
    p = _unit(protected.float()).cpu()
    k = min(int(top_k), int(c.shape[0]))
    sim = p @ c.t()                                   # [n, N]
    origin = sim.diagonal().clone()
    # the impostors are the k nearest *other* vectors: leaving the origin in
    # its own candidate list would score it against itself and cap the AUC of
    # a perfect attack below 1.0 (measured: 0.916 with k = 8)
    candidates = sim.clone()
    candidates.fill_diagonal_(float("-inf"))
    impostors = candidates.topk(min(k, int(c.shape[0]) - 1), dim=-1).values
    # one origin score per row (label 1) against that row's impostors (0):
    # the attack is blind when the origin no longer outranks its own
    # neighbourhood, which is exactly what a rotation of the corpus does
    labels = [1.0] * int(p.shape[0]) + [0.0] * int(impostors.numel())
    scores = origin.tolist() + impostors.reshape(-1).tolist()
    return roc_auc(labels, scores)


def anonymity_radius(corpus: torch.Tensor, share: float = 0.01) -> float:
    """``epsilon`` whose ball holds ``share`` of the corpus: a *measured* radius.

    A fixed cosine cut is meaningless in a space this anisotropic -- this one
    sits at a mean pairwise cosine of ~0.95, so ``epsilon = 0.05`` puts 1963
    of 2048 vectors inside the ball of every point and k-anonymity degenerates
    to the size of the corpus.  Defining the radius as a quantile of the
    corpus's own distribution is the same fix the clustering report applies to
    its entropy threshold, and it makes k comparable across spaces.
    """
    c = _unit(corpus.float()).cpu()
    sim = (c @ c.t())
    iu = np.triu_indices(int(c.shape[0]), k=1)
    return float(1.0 - np.quantile(sim.numpy()[iu], 1.0 - float(share)))


def k_anonymity(corpus: torch.Tensor, protected: torch.Tensor,
                epsilon: float) -> Dict[str, float]:
    """Median/mean corpus size within cosine epsilon of each protected vector."""
    c = _unit(corpus.float()).cpu()
    p = _unit(protected.float()).cpu()
    counts = (p @ c.t()).gt(1.0 - float(epsilon)).sum(dim=-1)
    counts = _np(counts).astype(float)
    return {"k_anon_median": float(np.median(counts)),
            "k_anon_mean": float(np.mean(counts)),
            "k_anon_p10": float(np.percentile(counts, 10)),
            "epsilon": float(epsilon)}


def retrieval_mrr(corpus: torch.Tensor, protected: torch.Tensor,
                  queries: torch.Tensor, targets: Sequence[int],
                  k: int = 10) -> float:
    """MRR@k of a query against its protected origin inside the protected set.

    The query is the *original* vector (what a client holds), the index is
    the *published* set: the only way the lookup still works is the
    isometry, so the drop of this MRR versus the unprotected index is the
    usefulness cost the scheme actually pays.
    """
    c = _unit(corpus.float()).cpu()
    p = _unit(protected.float()).cpu()
    q = _unit(queries.float()).cpu()
    sim = q @ p.t()
    rr = []
    for i, t in enumerate(targets):
        rank = int((sim[i] > sim[i, t]).sum().item()) + 1
        rr.append(1.0 / min(rank, k))
    return float(np.mean(rr)) if rr else float("nan")


def protection_grid(
    protect,
    corpus: torch.Tensor,
    lambdas: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    epsilon: float = 0.05,
    top_k: int = 64,
    mrr_k: int = 10,
) -> Dict[str, Dict[str, float]]:
    """The privacy--usefulness grid of the document, for one protection.

    ``protect(corpus, lam)`` returns the published vectors; the caller
    passes one closure per arm (the semantic key, the random-key control,
    the gaussian-noise floor) so every arm is measured by the same code.
    """
    out: Dict[str, Dict[str, float]] = {}
    for lam in lambdas:
        pub = protect(corpus, float(lam))
        row: Dict[str, float] = {}
        row.update(k_anonymity(corpus, pub, epsilon))
        row["linkage_auc"] = linkage_auc(corpus, pub, top_k=top_k)
        row["mrr"] = retrieval_mrr(corpus, pub, corpus,
                                   targets=list(range(int(corpus.shape[0]))),
                                   k=mrr_k)
        out[str(lam)] = row
    return out


def cce(protect, invert, corpus: torch.Tensor,
        lambdas: Sequence[float] = (0.25, 0.5, 1.0)) -> Dict[str, float]:
    """Key check: does ``invert(protect(v, lam), lam) == v`` exactly?

    The document's assert (``max|back - v| < 1e-4``) is the contract of a
    *reversible* scheme; every 1e-3 here is the number a legal hold
    cannot accept.
    """
    out: Dict[str, float] = {}
    for lam in lambdas:
        pub = protect(corpus, float(lam))
        back = invert(pub, float(lam))
        out[f"cce_{lam}"] = float((back.float() - corpus.float()).abs().max())
    return out
