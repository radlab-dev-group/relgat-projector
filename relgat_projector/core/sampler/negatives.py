"""
Negative sampling for the WIĄZKA pipeline.

Fixes the old ``EdgeDataset`` problems (PLAN.md section 0):

* no ``O(|E| * |V|)`` candidate materialisation,
* degree-aware sampling ``p(i) ~ deg(i) ** power`` (default 3/4) via the
  empirical cumulative distribution, instead of uniform sampling that makes
  almost every negative trivially far,
* optional *type-aware* hard negatives (same POS / domain) for wordnets,
* optional filtering of *false negatives* (a real target of the relation is
  never used as a negative).
"""

from __future__ import annotations

import bisect
import random
from typing import List, Optional, Sequence

import numpy as np


class NegativeSampler:
    def __init__(
        self,
        num_nodes: int,
        degrees: np.ndarray,
        power: float = 0.75,
        type_ids: Optional[np.ndarray] = None,
        rng: Optional[random.Random] = None,
    ):
        self.num_nodes = int(num_nodes)
        deg = np.asarray(degrees, dtype=np.float64).copy()
        if deg.shape[0] != self.num_nodes:
            raise ValueError(
                f"degrees has {deg.shape[0]} entries, expected {self.num_nodes}"
            )
        deg[deg <= 0] = 1.0
        p = deg ** float(power)
        p = p / p.sum()
        self.cum = np.cumsum(p)
        self.cum[-1] = 1.0
        self.rng = rng or random.Random()
        self.type_ids = (
            np.asarray(type_ids, dtype=np.int64) if type_ids is not None else None
        )

    # ------------------------------------------------------------------ core
    def _draw(self) -> int:
        """One degree-biased node index."""
        u = self.rng.random()
        return int(bisect.bisect_left(self.cum, u))

    def sample_pool(self, exclude: Sequence[int], k: int) -> List[int]:
        """Sample ``k`` distinct negative nodes, avoiding ``exclude``."""
        excl = set(int(e) for e in exclude)
        out: List[int] = []
        seen: set = set()
        tries = 0
        while len(out) < k and tries < k * 200:
            tries += 1
            i = self._draw()
            if i in excl or i in seen:
                continue
            seen.add(i)
            out.append(i)
        return out

    def sample(self, exclude: Sequence[int]) -> List[int]:
        """Sample ``len(exclude)`` negatives (one per excluded node)."""
        return self.sample_pool(exclude, k=len(exclude))

    def sample_type_aware(
        self, node: int, exclude: Sequence[int], k: int
    ) -> List[int]:
        """Hard negatives: prefer nodes sharing the type of ``node``."""
        if self.type_ids is None:
            return self.sample_pool([*exclude, node], k)
        t = self.type_ids[node]
        candidates = np.flatnonzero(self.type_ids == t)
        blocked = set(int(e) for e in exclude) | {int(node)}
        candidates = [int(c) for c in candidates if c not in blocked]
        if not candidates:
            return self.sample_pool([*exclude, node], k)
        return self.rng.sample(candidates, min(k, len(candidates)))
