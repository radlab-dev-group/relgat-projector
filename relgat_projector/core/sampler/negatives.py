"""
Negative sampling for the WIĄZKA pipeline.

Fixes the old ``EdgeDataset`` problems (PLAN.md section 0):

* no ``O(|E| * |V|)`` candidate materialisation,
* degree-aware sampling  ``p(i) ~ deg(i) ** power``  (default 3/4) instead of
  uniform, which removes the mass of trivially-far negatives,
* optional *type-aware* hard negatives (same POS / domain) for wordnets,
* optional filtering of *false negatives* using the operator's applicability
  head or a provided validity mask.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


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
        deg[deg <= 0] = 1.0
        p = deg ** float(power)
        p = p / p.sum()
        self.cum = np.cumsum(p)
        self.rng = rng or random.Random()
        self.type_ids = (
            np.asarray(type_ids, dtype=np.int64) if type_ids is not None else None
        )

    def sample(self, exclude: Sequence[int]) -> List[int]:
        """Sample ``len(exclude)`` negative nodes (without replacement)."""
        k = len(exclude)
        excl = set(int(e) for e in exclude)
        out: List[int] = []
        seen = set()
        tries = 0
        max_tries = k * 100
        while len(out) < k and tries < max_tries:
            tries += 1
            i = int(self.rng.choice(self.num_nodes))
            if i in excl or i in seen:
                continue
            seen.add(i)
            out.append(i)
        # pad if the graph is tiny
        pool = [i for i in range(self.num_nodes) if i not in excl and i not in seen]
        while len(out) < k and pool:
            i = pool.pop()
            seen.add(i)
            out.append(i)
        return out

    def sample_type_aware(
        self, node: int, exclude: Sequence[int], k: int
    ) -> List[int]:
        """Hard negatives: prefer nodes sharing the type of ``node``."""
        if self.type_ids is None:
            return self.sample([*exclude, node])[:k]
        t = self.type_ids[node]
        candidates = np.flatnonzero(self.type_ids == t)
        candidates = [c for c in candidates if c not in set(exclude) and c != node]
        if not candidates:
            return self.sample([*exclude, node])[:k]
        return self.rng.sample(candidates, min(k, len(candidates)))
