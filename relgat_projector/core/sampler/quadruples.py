"""
Closed 4-tuple (cycle) sampling for the holonomy / curvature loss.

A *closed quadruple* is a 4-cycle  ``A --r--> C <--s-- B``  completed by two
relations ``p, q`` back to ``A``, i.e. a square

    A --p--> B
    |        |
    r        s
    v        v
    C --q--> D  (or directly A --q--> B)

In practice we sample squares from the graph: for an edge ``A --r--> C`` we
look for a node ``B`` such that ``A --p--> B`` and ``B --s--> C`` exist,
giving the cycle ``A -> C -> B -> A`` with relation sequence
``(r, s, p)`` plus the closing edge ``q``.

The *shadow* variant (PLAN.md 2.3) instead builds quadruples from arbitrary
vectors in the embedding neighbourhood, so that curvature is supervised also
off the graph.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

# quadruple: (a, r, c, s, b, p, q)  -- A --r--> C, B --s--> C, A --p--> B,
# with the holonomy  H = T_p^{-1} o T_q o T_r o T_s  around A.
Quadruple = Tuple[int, int, int, int, int, int, int]


class QuadrupleSampler:
    def __init__(
        self,
        out_adj: Dict[int, List[Tuple[int, int]]],
        in_adj: Dict[int, List[Tuple[int, int]]],
        num_nodes: int,
        rng: Optional[random.Random] = None,
    ):
        self.out_adj = out_adj
        self.in_adj = in_adj
        self.num_nodes = num_nodes
        self.rng = rng or random.Random()

    def sample(self) -> Optional[Quadruple]:
        # pick A with out-neighbour C and in-neighbour B sharing a common
        # destination C
        nodes = list(self.out_adj.keys())
        if not nodes:
            return None
        a = self.rng.choice(nodes)
        out_c = self.out_adj.get(a, [])
        if not out_c:
            return None
        c, r = self.rng.choice(out_c)
        in_b = self.in_adj.get(c, [])
        if not in_b:
            return None
        b, s = self.rng.choice(in_b)
        # find p: a -> b
        p = None
        for dst, rel in self.out_adj.get(a, []):
            if dst == b:
                p = rel
                break
        # find q: c -> b (closing the square so that H is defined around A)
        q = None
        for dst, rel in self.out_adj.get(c, []):
            if dst == b:
                q = rel
                break
        if p is None or q is None:
            return None
        return (a, r, c, s, b, p, q)

    def sample_batch(self, batch_size: int) -> List[Quadruple]:
        out = []
        tries = 0
        while len(out) < batch_size and tries < batch_size * 200:
            tries += 1
            q = self.sample()
            if q is not None:
                out.append(q)
        return out
