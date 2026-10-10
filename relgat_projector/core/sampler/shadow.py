"""
Shadow-manifold sampling: real text vectors from the retriever index that are
NOT graph nodes (PLAN.md 2.6).

These are used to supervise the *off-graph* behaviour of the operator:
cycle consistency, commutation, and pull-back coherence with the projection.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np


class ShadowSampler:
    def __init__(self, vectors: np.ndarray, rng: Optional[int] = 42):
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.rng = np.random.default_rng(rng)

    def __len__(self) -> int:
        return int(self.vectors.shape[0])

    def sample(self, batch_size: int) -> np.ndarray:
        idx = self.rng.integers(0, len(self), size=batch_size)
        return self.vectors[idx]
