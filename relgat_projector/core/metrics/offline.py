"""
Offline / product-level metrics (PLAN.md section 5).

* **OMD** -- Off-Manifold Degradation: ratio of node MRR to shadow-text MRR.
* **APL** -- Applicability AUC.
* **ECE** -- Expected calibration error of the applicability head.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch


@torch.no_grad()
def off_manifold_degradation(node_mrr: float, text_mrr: float) -> float:
    """1.0 means no degradation; lower = the model degrades off-graph."""
    if node_mrr <= 0:
        return 0.0
    return min(1.0, text_mrr / node_mrr)


def applicability_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC (no sklearn dependency)."""
    scores = np.asarray(scores).ravel()
    labels = np.asarray(labels).ravel()
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # probability a random positive scores above a random negative
    greater = 0.0
    equal = 0.0
    for p in pos:
        greater += (neg < p).sum()
        equal += (neg == p).sum()
    return float((greater + 0.5 * equal) / (len(pos) * len(neg)))


def calibration_ece(
    probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> float:
    probabilities = np.asarray(probabilities).ravel()
    labels = np.asarray(labels).ravel()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (probabilities >= lo) & (probabilities < hi) if i < n_bins - 1 \
            else (probabilities >= lo) & (probabilities <= hi)
        if not m.any():
            continue
        conf = probabilities[m].mean()
        acc = labels[m].mean()
        ece += m.sum() * abs(conf - acc)
    return float(ece / max(1, len(labels)))
