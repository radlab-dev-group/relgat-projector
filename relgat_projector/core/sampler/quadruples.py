"""
Confluent-diamond (closed quadruple) sampling for the holonomy loss.

A *diamond* is a pair of two-step routes between the same endpoints::

    A --r--> C --q--> D
    A --p--> B --s--> D

The transport holonomy around it is ``H = T_p^-1 o T_s^-1 o T_q o T_r`` (a loop
based at ``A``), and penalising ``||H(x_A) - x_A||`` is what makes the operator
path-consistent (PLAN.md 2.3).

A diamond witnesses the **commutation of the pair** ``(r, p)`` only when the two
routes are exactly ``r then p`` and ``p then r`` (``q == p`` and ``s == r``).
One diamond at a single node is still weak evidence, so a pair is a positive
only when witnessed at several *distinct* base points.

Sampling is compiled once (lazily) and then drawn from, because re-scanning the
graph for every training step is pure waste -- especially on graphs that have
no diamonds at all.  For large graphs the scan is bounded, which can only
reduce the number of witnesses, never fabricate one.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

Adj = Dict[int, List[Tuple[int, int]]]


@dataclass(frozen=True)
class Diamond:
    a: int  # base point A
    r: int  # A --r--> C
    c: int
    q: int  # C --q--> D
    d: int
    p: int  # A --p--> B
    b: int  # B --s--> D
    s: int


class QuadrupleSampler:
    def __init__(
        self,
        out_adj: Adj,
        in_adj: Optional[Adj] = None,
        num_nodes: Optional[int] = None,
        num_rel: Optional[int] = None,
        rng: Optional[random.Random] = None,
        scan_attempts: int = 20_000,
        max_diamonds: int = 50_000,
    ):
        self.out_adj = out_adj
        self.in_adj = in_adj or {}
        self.num_nodes = num_nodes or len(out_adj)
        self.num_rel = num_rel
        self.rng = rng or random.Random()
        self.scan_attempts = int(scan_attempts)
        self.max_diamonds = int(max_diamonds)

        # successor lookup: node -> {successor: [relation, ...]}
        self._succ: Dict[int, Dict[int, List[int]]] = {}
        for u, edges in self.out_adj.items():
            bucket = self._succ.setdefault(u, {})
            for v, rel in edges:
                bucket.setdefault(v, []).append(rel)

        self._nodes2 = [u for u, e in self.out_adj.items() if len(e) >= 2]
        self._diamonds: Optional[List[Diamond]] = None

    # ------------------------------------------------------------ compilation
    def _find_at(self, a: int) -> Optional[Diamond]:
        edges = self.out_adj.get(a, [])
        if len(edges) < 2:
            return None
        (c, r), (b, p) = self.rng.sample(edges, 2)
        if c == b or r == p:
            return None
        succ_c = self._succ.get(c, {})
        succ_b = self._succ.get(b, {})
        common = (set(succ_c) & set(succ_b)) - {a}
        if not common:
            return None
        d = self.rng.choice(sorted(common))
        return Diamond(
            a=a, r=r, c=c, q=self.rng.choice(succ_c[d]), d=d,
            p=p, b=b, s=self.rng.choice(succ_b[d]),
        )

    def _ensure_diamonds(self) -> List[Diamond]:
        if self._diamonds is not None:
            return self._diamonds
        found: List[Diamond] = []
        if not self._nodes2:
            self._diamonds = found
            return found
        for _ in range(min(self.scan_attempts, max(1_000, 40 * len(self._nodes2)))):
            d = self._find_at(self.rng.choice(self._nodes2))
            if d is not None:
                found.append(d)
            if len(found) >= self.max_diamonds:
                break
        self._diamonds = found
        return found

    # ------------------------------------------------------------------ core
    def sample(self) -> Optional[Diamond]:
        diamonds = self._ensure_diamonds()
        if not diamonds:
            return None
        return self.rng.choice(diamonds)

    def sample_batch(self, batch_size: int) -> List[Diamond]:
        diamonds = self._ensure_diamonds()
        if not diamonds:
            return []
        return [self.rng.choice(diamonds) for _ in range(batch_size)]

    # ------------------------------------------------------- supervision (C)
    def witness_counts(self, n_draws: int = 512) -> Dict[Tuple[int, int], Set[int]]:
        """Pair ``(r, p)`` -> base points at which ``r`` and ``p`` commute."""
        counts: Dict[Tuple[int, int], Set[int]] = {}
        for d in self.sample_batch(n_draws):
            if d.q != d.p or d.s != d.r:
                continue  # routes are not r.p / p.r: no commutation witness
            if d.r == d.p:
                continue  # (r, r) commutes trivially: no information
            counts.setdefault((d.r, d.p), set()).add(d.a)
            counts.setdefault((d.p, d.r), set()).add(d.a)
        return counts

    def _unwitnessed_pairs(self, seen: Set[Tuple[int, int]], max_neg: int):
        """Pairs never witnessed commuting, drawn over ALL relations."""
        all_rel = (
            list(range(self.num_rel))
            if self.num_rel
            else sorted({r for pair in seen for r in pair})
        )
        if not all_rel or max_neg <= 0:
            return []
        neg: List[Tuple[int, int]] = []
        tries = 0
        while len(neg) < max_neg and tries < max_neg * 100:
            tries += 1
            pair = (self.rng.choice(all_rel), self.rng.choice(all_rel))
            if pair[0] == pair[1] or pair in seen or pair in neg:
                continue
            neg.append(pair)
        return neg

    def sample_relation_pairs(
        self,
        batch_size: int,
        max_neg: int = 16,
        min_support: int = 3,
        n_diamonds: Optional[int] = None,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Supervision for the commutation matrix ``C_rs`` (PLAN.md 2.3).

        Positives are pairs witnessed at ``min_support`` distinct base points
        (what a genuine algebraic identity implies); negatives are pairs over
        all relations with no witness at all.  Weakly witnessed pairs stay
        unsupervised because they are ambiguous.
        """
        n_draws = n_diamonds or max(batch_size * 8, 256)
        counts = self.witness_counts(n_draws)
        pos = [pair for pair, bases in counts.items() if len(bases) >= min_support]
        seen = set(pos)
        neg = self._unwitnessed_pairs(seen, max_neg)
        return pos, neg
