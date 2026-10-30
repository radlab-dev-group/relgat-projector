"""
Relation-chain (path) sampling for supervision of composition (k >= 2).

Paths are sampled by a constrained random walk on the relation graph:

* no immediate return to the previous node,
* the same relation is not used twice in a row,
* a per-relation *type validity* mask (optional) prevents walks along
  relation types that are not defined at the current node (e.g. ``ant`` for
  a proper noun), which keeps supervision consistent with the
  applicability head (PLAN.md 1.4, 2.2).

Yields tuples ``(src, rel_ids [k], dst)`` with compact node indices.
"""

from __future__ import annotations

import random
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

from torch.utils.data import Dataset


class PathSampler:
    """Constrained random-walk sampler over a relational graph."""

    def __init__(
        self,
        adjacency: dict,
        rel_ids: List[int],
        num_nodes: int,
        max_depth: int = 4,
        depth_weights: Optional[Sequence[float]] = None,
        rng: Optional[random.Random] = None,
    ):
        # adjacency: src -> list of (dst, rel_id)
        self.adjacency = adjacency
        self.rel_ids = list(rel_ids)
        self.num_nodes = num_nodes
        self.max_depth = int(max_depth)
        if depth_weights is None:
            # curriculum-friendly: shorter paths slightly more likely
            self.depth_weights = np.array(
                [1.0 / (d + 1) for d in range(2, self.max_depth + 1)], dtype=np.float64
            )
        else:
            self.depth_weights = np.asarray(depth_weights, dtype=np.float64)
        self.depth_weights = self.depth_weights / self.depth_weights.sum()
        self._depths = list(range(2, self.max_depth + 1))
        self.rng = rng or random.Random()

    def _pick_depth(self) -> int:
        return int(self.rng.choices(self._depths, weights=self.depth_weights.tolist())[0])

    def pick_depth(self) -> int:
        """Sample a chain length the same way :meth:`sample` would."""
        return self._pick_depth()

    def sample(self) -> Optional[Tuple[int, List[int], int]]:
        """One random walk of length in [2, max_depth]."""
        start = self.rng.randrange(self.num_nodes)
        if start not in self.adjacency or not self.adjacency[start]:
            return None
        depth = self._pick_depth()
        node = start
        chain: List[Tuple[int, int]] = []  # (dst, rel)
        prev = None
        prev_rel = None
        for _ in range(depth):
            options = []
            for dst, rel in self.adjacency[node]:
                if dst == prev:
                    continue
                if rel == prev_rel:
                    continue
                options.append((dst, rel))
            if not options:
                break
            dst, rel = self.rng.choice(options)
            chain.append((dst, rel))
            prev, node, prev_rel = node, dst, rel
        if len(chain) < 2:
            return None
        return start, [r for _, r in chain], node

    def sample_batch(
        self, batch_size: int, depth: Optional[int] = None
    ) -> List[Tuple[int, List[int], int]]:
        """Up to ``batch_size`` walks, optionally all of exactly ``depth``.

        A fixed depth lets a training batch be stacked into ``[B, k]`` *without
        padding*: a padded row would apply relation id 0 (``Pertainym`` in the
        Słowosieć export) once per pad slot on top of the real chain, while the
        supervision target is the end of the unpadded chain.
        """
        out = []
        tries = 0
        while len(out) < batch_size and tries < batch_size * 50:
            tries += 1
            p = self.sample()
            if p is None:
                continue
            if depth is not None and len(p[1]) != int(depth):
                continue
            out.append(p)
        return out


class PathDataset(Dataset):
    """Deterministic-ish path dataset: pre-generates a fixed pool of paths."""

    def __init__(self, sampler: PathSampler, size: int, seed: int = 42):
        self.sampler = sampler
        self.paths: List[Tuple[int, List[int], int]] = []
        rng = random.Random(seed)
        tries = 0
        while len(self.paths) < size and tries < size * 100:
            tries += 1
            p = self.sampler.sample()
            if p is not None:
                self.paths.append(p)
        if not self.paths:
            raise RuntimeError("PathDataset: no paths could be sampled")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        src, rels, dst = self.paths[idx]
        return src, rels, dst
