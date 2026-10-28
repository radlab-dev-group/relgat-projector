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

import torch

Adj = Dict[int, List[Tuple[int, int]]]

#: columns of the compact ``[M, 8]`` diamond tensor
FIELDS = ("a", "r", "c", "q", "d", "p", "b", "s")


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
        max_diamonds: int = 400_000,
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
        self._diamonds: Optional[torch.Tensor] = None
        #: (r, p) -> distinct base points at which the routes r.p and p.r meet
        self._witness_table: Optional[Dict[Tuple[int, int], Set[int]]] = None
        #: relation pairs whose composed successor sets disagree (see
        #: :meth:`order_dependence_witnesses`); ``None`` = not scanned yet
        #: order-dependence teacher evidence, filled by
        #: :meth:`order_dependence_examples` (``None`` = not scanned yet)
        self._order_pairs: Optional[List[Tuple[int, int]]] = None
        self._order_rows: Optional[List[Tuple[int, int, int, int]]] = None
        #: ``(r, s) -> (n_examples, order_dependent_rate)`` for EVERY pair the
        #: scan saw, before the decisiveness filter of
        #: :meth:`order_dependence_examples` drops the mixed ones; filled by
        #: that method, ``None`` = not scanned yet.  ``C_rs`` is trained on the
        #: *rate* (a probability), which is what makes a soft, unbiased
        #: supervision possible: the filter above must drop a mixed pair
        #: because a hinge cannot express "sometimes", but a probability can.
        self._order_rates: Optional[Dict[Tuple[int, int], Tuple[int, float]]] = None

    # ------------------------------------------------------------ compilation
    def _find_at(self, a: int) -> Optional[Diamond]:
        """One diamond at ``a`` by sampling (kept for explicit single draws)."""
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

    def _compile(self) -> None:
        """Enumerate the closed quadruples of the graph exhaustively.

        The previous implementation *sampled* ``min(scan_attempts, ...)`` random
        nodes, picked two random out-edges of each and asked for a common
        successor: on the Slowosieć that produced 5 947 diamonds out of the
        203 208 the graph actually has, i.e. the holonomy loss was fitted on 3 %
        of the available evidence -- and the commutation witness table on top of
        that 3 %, which is why the learned ``C_rs`` never left 0.5.  A full scan
        of this graph costs under a second, so it is simply done once.

        Every ordered pair of out-edges of every node is considered, and every
        common successor of their heads contributes one diamond (all
        combinations of parallel relations, so a diamond witnessing the
        commutation of ``(r, p)`` is never silently replaced by another one).
        Nodes are visited in shuffled order so that the ``max_diamonds`` cap
        thins the set uniformly instead of keeping the lowest node ids.
        """
        found: List[Tuple[int, int, int, int, int, int, int, int]] = []
        witnesses: Dict[Tuple[int, int], Set[int]] = {}
        order = [u for u in self._nodes2]
        self.rng.shuffle(order)
        cap = int(self.max_diamonds)
        for a in order:
            edges = self.out_adj.get(a, [])
            if len(edges) < 2:
                continue
            by_succ: Dict[int, List[int]] = {}
            for v, r in edges:
                by_succ.setdefault(v, []).append(r)
            arms = list(by_succ.items())
            for i in range(len(arms)):
                c, r_ids = arms[i]
                succ_c = self._succ.get(c)
                if not succ_c:
                    continue
                for j in range(len(arms)):
                    if i == j:
                        continue
                    b, p_ids = arms[j]
                    succ_b = self._succ.get(b)
                    if not succ_b:
                        continue
                    for d in succ_c.keys() & succ_b.keys():
                        if d == a:
                            continue
                        for q in succ_c[d]:
                            for s in succ_b[d]:
                                for r in r_ids:
                                    for p in p_ids:
                                        if len(found) < cap:
                                            found.append(
                                                (a, r, c, q, d, p, b, s))
                                        # A --r--> C --q--> D  and
                                        # A --p--> B --s--> D commute exactly
                                        # when q == p and s == r
                                        if q == p and s == r and r != p:
                                            witnesses.setdefault((r, p), set()).add(a)
        self._diamonds = (
            torch.tensor(found, dtype=torch.long)
            if found else torch.empty(0, 8, dtype=torch.long)
        )
        self._witness_table = witnesses

    def _ensure_diamonds(self):
        if self._diamonds is None:
            self._compile()
        return self._diamonds

    def order_dependence_examples(
        self,
        scan_nodes: int = 4000,
        jaccard: float = 0.3,
        min_support: int = 3,
        min_fraction: float = 0.5,
        max_per_pair: int = 512,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int, int]]]:
        """Labelled ``(a, r, s)`` examples: does the order of ``r, s`` matter?

        The commutation witness table answers "at which base points do the two
        routes *meet*?".  That is only half of what the curvature loss needs:
        a one-sided hinge can say "do not pull these apart" but never "push
        these apart", so nothing ever asks the rotations to rotate -- measured
        closed-loop holonomy 0.0003 against 0.120 on the reference chart.

        This reads the missing half off the graph.  At a node ``a`` carrying
        both relations ``r`` and ``s``, compare the composed 2-hop successor
        *sets* of the two orders: ``a.r.s`` and ``a.s.r``.  Where both are
        non-empty they either coincide (the order reaches the same
        neighbourhood -- a commutation fact about the *orders* themselves, not
        about one closed quadruple) or they do not (the order changes what is
        reachable).  Each such ``(a, r, s)`` is returned with that verdict as
        its label, so the loss is taught per example instead of per pair, and
        pairs whose evidence is mixed -- neither decisively commutative nor
        decisively order-dependent -- are dropped entirely.

        Cost is a few seconds once; the result is cached.
        """
        if self._order_rows is not None:
            return list(self._order_pairs), list(self._order_rows)

        # node -> {relation: [successors]}, parallel edges merged
        succ_by_rel: Dict[int, Dict[int, List[int]]] = {}
        for u, edges in self.out_adj.items():
            bucket: Dict[int, List[int]] = {}
            for v, rel in edges:
                bucket.setdefault(rel, []).append(v)
            succ_by_rel[u] = bucket

        candidates = [
            u for u in self._nodes2
            if len({rel for _, rel in self.out_adj.get(u, [])}) >= 2
        ]
        order = list(candidates)
        self.rng.shuffle(order)
        if scan_nodes and int(scan_nodes) < len(order):
            order = order[: int(scan_nodes)]

        support: Dict[Tuple[int, int], int] = {}
        disagree: Dict[Tuple[int, int], int] = {}
        rows: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for a in order:
            by_rel = succ_by_rel.get(a, {})
            rels = sorted(by_rel)
            for i in range(len(rels)):
                for j in range(i + 1, len(rels)):
                    r, s = rels[i], rels[j]
                    forward: set = set()
                    for mid in by_rel[r]:
                        forward.update(succ_by_rel.get(mid, {}).get(s, ()))
                    if not forward:
                        continue
                    backward: set = set()
                    for mid in by_rel[s]:
                        backward.update(succ_by_rel.get(mid, {}).get(r, ()))
                    if not backward:
                        continue  # only one order exists here: no evidence
                    union = len(forward | backward)
                    if not union:
                        continue
                    key = (r, s)
                    label = 1 if len(forward & backward) / union < float(jaccard) else 0
                    support[key] = support.get(key, 0) + 1
                    disagree[key] = disagree.get(key, 0) + label
                    rows.setdefault(key, []).append((a, label))

        pairs: List[Tuple[int, int]] = []
        kept: List[Tuple[int, int, int]] = []
        frac = float(min_fraction)
        rates: Dict[Tuple[int, int], Tuple[int, float]] = {}
        for key, n in sorted(support.items()):
            rates[key] = (int(n), disagree[key] / n)
            if n < int(min_support):
                continue
            rate = disagree[key] / n
            # decisive either way; a pair that sometimes commutes and sometimes
            # does not carries no algebraic statement and would teach noise
            if rate < frac and rate > 1.0 - frac:
                continue
            pairs.append(key)
            for a, label in rows[key][: int(max_per_pair)]:
                kept.append((a, key[0], key[1], label))
        self._order_pairs = pairs
        self._order_rows = kept
        self._order_rates = rates
        return list(pairs), list(kept)

    def order_pair_rates(
        self, min_support: int = 3
    ) -> List[Tuple[Tuple[int, int], float]]:
        """``((r, s), rate)`` for every scanned pair: ``C_rs`` as ``P(order-dependent)``.

        The rate is the fraction of base points at which the two composed
        orders reach different neighbourhoods, over ALL base points where both
        orders exist -- a per-pair probability, not the decisive-only subset
        :meth:`order_dependence_examples` keeps for the hinges.  Supervising
        ``C`` with it makes the matrix the 12b diagnostic reads describe the
        same graph fact the order teacher moves the operator with, instead of
        the witness-count proxy (which is a relation-frequency proxy: the most
        frequent relation was witnessed -- and therefore ``C -> 0.93`` -- on
        most pairs, inverted against the operator).

        ``min_support`` drops pairs whose rate rests on too few examples to
        mean anything.
        """
        if self._order_rates is None:
            self.order_dependence_examples()
        assert self._order_rates is not None
        out = [
            (pair, rate)
            for pair, (n, rate) in sorted(self._order_rates.items())
            if n >= int(min_support)
        ]
        return out

    def sample_order_batch(
        self, batch_size: int, device=None
    ):
        """``(a, r, s, dependent)`` long tensors for the order-dependence teacher.

        Both composed orders of the drawn triple exist at ``a`` in the graph by
        construction (that is what :meth:`order_dependence_examples` checked),
        so the two routes the loss compares are both realised by real edges of
        the ontology rather than by an arbitrary pair of relations.
        """
        pairs, rows = self.order_dependence_examples()
        if not rows:
            return None
        pick = self.rng.sample(rows, min(int(batch_size), len(rows)))
        out = torch.tensor(pick, dtype=torch.long)
        return out if device is None else out.to(device)

    def commutation_witnesses(
        self, min_support: int = 3
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Exhaustive ``(positives, negatives)`` supervision for ``C_rs``.

        Positives are ordered relation pairs whose two routes ``r.p`` and ``p.r``
        meet at ``min_support`` distinct base points (a genuine algebraic
        identity of the graph); negatives are pairs with **no** witness at all.
        Drawn from the complete scan, so a pair is never missing because a
        random draw happened to miss it -- the per-step 256-diamond version of
        this saw 3 % of the graph and produced positives whose identity changed
        from step to step.
        """
        table = self._witness_table
        if table is None:
            self._ensure_diamonds()
            table = self._witness_table
        pos = [pair for pair, bases in table.items() if len(bases) >= int(min_support)]
        pos.sort()
        seen = set(pos)
        if self.num_rel:
            all_pairs = [
                (r, p) for r in range(self.num_rel)
                for p in range(self.num_rel) if r != p
            ]
        else:
            rels = sorted({r for pair in seen for r in pair})
            all_pairs = [(r, p) for r in rels for p in rels if r != p]
        neg = [pair for pair in all_pairs if pair not in seen]
        return pos, neg

    # ------------------------------------------------------------------ core
    def sample(self) -> Optional[Diamond]:
        matrix = self._ensure_diamonds()
        if matrix.shape[0] == 0:
            return None
        row = matrix[self.rng.randrange(matrix.shape[0])]
        return self._to_diamond(row)

    @staticmethod
    def _to_diamond(row) -> Diamond:
        return Diamond(**{name: int(row[i]) for i, name in enumerate(FIELDS)})

    def sample_batch(self, batch_size: int) -> List[Diamond]:
        matrix = self._ensure_diamonds()
        if matrix.shape[0] == 0:
            return []
        m = matrix.shape[0]
        pick = [self.rng.randrange(m) for _ in range(batch_size)]
        return [self._to_diamond(matrix[i]) for i in pick]

    def sample_batch_tensor(self, batch_size: int, device=None):
        """``[B, 8]`` long tensor of diamonds (columns = :data:`FIELDS`).

        Same distribution as :meth:`sample_batch` without building ``B`` Python
        objects and re-lifting them to a tensor -- which is what the holonomy
        loss needs every training step.  Returns ``None`` when the graph has no
        closed quadruple at all.
        """
        matrix = self._ensure_diamonds()
        if matrix.shape[0] == 0:
            return None
        idx = torch.randint(0, matrix.shape[0], (batch_size,))
        out = matrix[idx]
        return out if device is None else out.to(device)

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
        (what a genuine algebraic identity implies), taken from the exhaustive
        witness table; negatives are pairs with no witness at all, drawn from
        the same table.  Weakly witnessed pairs stay unsupervised.

        ``batch_size`` is the number of positives drawn per step and negatives
        are capped to the same order, so the balanced pair distribution does
        not depend on how lucky the draw was: the per-step 256-diamond version
        produced ~24 positives against 16 negatives *of a different kind*
        (frequently re-drawn positives vs uniformly spread negatives) and the
        matrix stayed at ``0.49 +- 0.01`` for a whole 35-epoch run.
        """
        all_pos, all_neg = self.commutation_witnesses(min_support=min_support)
        if n_diamonds is not None:
            # explicit draw-count callers (tests) keep the sampling semantics
            counts = self.witness_counts(n_diamonds)
            all_pos = [p for p in all_pos
                       if len(counts.get(p, ())) >= min_support]
        if not all_pos:
            return [], []
        n = max(1, min(int(batch_size), len(all_pos)))
        pos = self.rng.sample(all_pos, n)
        seen = set(pos) | {p for p in all_pos}
        neg = self._unwitnessed_pairs(seen, max_neg)
        return pos, neg
