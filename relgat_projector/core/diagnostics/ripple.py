"""
Holonomy as a probe of editing side effects (docs/IDEAS/05-ripple-holonomy.md).

Knowledge editing is scored on the edited fact and never on the neighbourhood:
an edit of one fact that moves the transport everywhere else bends the
*curvature*, and no metric in use sees that without generating thousands of
probe prompts.  The idea's claim is that the loop defect is the cheap signal.

One thing has to be true before any of that can work, and this module is
written around it: with a context-dependent connection the transport at ``x``
is a function of ``x``, so the only way a measurement at an *untouched*
vector can notice an edit is for the edit to have changed the function.
Perturbing an edge, or a relation coefficient, changes the transport of every
vector by the same amount and says nothing about locality -- that would be a
tautology, and the last run in this repository burned a stage on one.

So the edit here is a **rank-one edit of the connection itself** (the analogue
of a ROME edit: a small change of the weights that compute the transport,
aimed at one entity, leaving every node vector alone):

    P' = P + v h_e^T,   h_e = relu(W x_e)

The induced change of the coefficients at a probe ``x`` is proportional to
``<h_e, h_x>``, so it is large near ``x_e`` and small far away -- locality is
a *hypothesis of the edit*, and the measurement asks whether it holds:

    dHD(e) = HD_after(e) - HD_before(e)

``concentration_auc`` scores dHD of the edited entity and its graph neighbours
against dHD of far-away entities.  0.5 means the edit is global (no locality,
the probe is worthless for this idea); > 0.75 is the document's go threshold.
``zero_edit`` runs the identical protocol with a zero-magnitude edit: that is
the noise floor every one of these numbers has to be read against.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _unit(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1)


# ------------------------------------------------------------------ graph view
def incident_relations(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    rel_names: Sequence[str],
    rows: Sequence[int],
) -> Dict[int, List[str]]:
    """Relation names of the edges incident to each of ``rows``.

    The graph is read as undirected: an edit of the connection around ``e``
    reaches the transport of ``e`` whichever way the dictionary points its
    arrow, and the loop defect is about the pair of routes, not their sign.
    """
    wanted = torch.as_tensor(list(rows), dtype=torch.long,
                             device=edge_index.device)
    mask = torch.isin(edge_index[0], wanted) | torch.isin(edge_index[1], wanted)
    src, dst = edge_index[0][mask].tolist(), edge_index[1][mask].tolist()
    types = edge_type[mask].tolist()
    out: Dict[int, List[str]] = {int(r): [] for r in rows}
    for s, d, t in zip(src, dst, types):
        for node in (s, d):
            if node in out:
                name = rel_names[int(t)]
                if name not in out[node]:
                    out[node].append(name)
    return out


def adjacency(edge_index: torch.Tensor) -> Dict[int, List[int]]:
    """Neighbours of every node, undirected."""
    adj: Dict[int, List[int]] = defaultdict(list)
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for s, d in zip(src, dst):
        adj[int(s)].append(int(d))
        adj[int(d)].append(int(s))
    return adj


# ------------------------------------------------------------------ local HD
def loop_pairs(relations: Sequence[str]) -> List[Tuple[str, str]]:
    """Unordered pairs of distinct relations available at one vertex."""
    names = sorted(set(relations))
    return [(names[i], names[j])
            for i in range(len(names)) for j in range(i + 1, len(names))]


def local_hd(transport, x: torch.Tensor,
             pairs: Sequence[Tuple[str, str]]) -> Dict[str, float]:
    """The order defect ``1 - cos(T_s T_r x, T_r T_s x)`` at ``x``.

    Each transport in a chain is modulated at the point it is applied at --
    the connection, not a fixed rotation -- so a flat connection is what the
    defect being 0 actually means.
    """
    if not pairs:
        return {"hd": float("nan"), "n_pairs": 0}
    hds = []
    for r, s in pairs:
        u = transport.apply(transport.apply(x, r), s)
        w = transport.apply(transport.apply(x, s), r)
        hds.append(float((1.0 - (_unit(u) * _unit(w)).sum(dim=-1)).mean()))
    return {"hd": float(np.mean(hds)), "n_pairs": len(pairs),
            "hd_max": float(np.max(hds))}


# ---------------------------------------------------------------- the edit
@contextmanager
def connection_edit(model, base_vector: torch.Tensor, magnitude: float,
                    generator: Optional[torch.Generator] = None):
    """Apply ``P' = P + magnitude * v h_e^T`` to the connection, then undo it.

    ``h_e = relu(W x_e)``, so the change the coefficients see at a probe ``x``
    is driven by ``<h_e, h_x>``: large near ``x_e``, small as ``x`` leaves its
    direction.  Locality is therefore an assumption *of the edit*, and the
    measurement asks whether it holds.

    ``magnitude = 0`` touches nothing and restores nothing -- the zero edit of
    the go/no-go protocol, run through the identical code path, so what it
    measures is the floor of the procedure rather than of the model.
    """
    op = model.operator
    if magnitude == 0.0 or op.ctx_P is None:
        yield model
        return
    with torch.no_grad():
        h_e = torch.relu(op.ctx_W(base_vector)).reshape(-1)
        direction = torch.randn(op.ctx_P.weight.shape[0],
                                device=op.ctx_P.weight.device,
                                generator=generator)
        direction = F.normalize(direction, dim=-1)
        delta = float(magnitude) * torch.outer(
            direction, h_e / max(float(h_e.norm()), 1e-9))
        op.ctx_P.weight.add_(delta)
    try:
        yield model
    finally:
        with torch.no_grad():
            op.ctx_P.weight.sub_(delta)


def measure_ripple(
    transport,
    vectors: torch.Tensor,
    probes: Sequence[Tuple[int, Sequence[Tuple[str, str]]]],
) -> np.ndarray:
    """HD of every ``(row, relation pairs)`` probe, in the probes' order."""
    out = []
    for row, pairs in probes:
        x = _unit(transport.prepare(vectors[row:row + 1].float()))
        out.append(local_hd(transport, x, pairs)["hd"])
    return np.asarray(out, dtype=float)


def concentration_auc(near: Sequence[float], far: Sequence[float]) -> float:
    """AUC of "the edit damaged the neighbourhood" (``near`` > ``far``)."""
    from .consistency import roc_auc

    labels = [1.0] * len(near) + [0.0] * len(far)
    return roc_auc(labels, list(near) + list(far))


def coverage(
    vectors: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    rel_names: Sequence[str],
    sample: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    """Share of entities with at least one loop to measure (idea 05, sec. 6).

    The document makes this a gate: below ~30 % the probe has nothing to
    measure for most entities and the idea has to be narrowed to the dense
    part of the graph.
    """
    rows = list(range(int(vectors.shape[0]))) if sample is None else list(sample)
    inc = incident_relations(edge_index, edge_type, rel_names, rows)
    n_pairs = {r: len(loop_pairs(rels)) for r, rels in inc.items()}
    have = [v for v in n_pairs.values() if v >= 1]
    return {
        "coverage": len(have) / max(1, len(n_pairs)),
        "n_entities": len(n_pairs),
        "n_measurable": len(have),
        "pairs_median": float(np.median([v for v in have])) if have else 0.0,
    }
