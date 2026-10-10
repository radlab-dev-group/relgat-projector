"""
Scatter primitives with an optional ``torch_scatter`` backend.

Some environments ship a broken ``torch_scatter`` wheel (it segfaults on
import, which cannot be caught by ``try/except``).  Everything in this
package therefore imports the primitives from here instead of directly from
``torch_scatter`` and falls back to pure PyTorch implementations when the
extension is unavailable **or** when ``RELgat_NO_TORCH_SCATTER=1``.

The pure implementations cover the 1-D-index / ``dim=0`` case used by the
GAT layer and match ``torch_scatter.scatter_add`` / ``scatter_max``
semantics (nodes without any incoming edge carry ``-inf`` and argmax ``-1``).
"""

import os

import torch

if os.environ.get("RELgat_NO_TORCH_SCATTER") == "1":
    _HAS_TORCH_SCATTER = False
else:
    try:  # pragma: no cover - environment dependent
        from torch_scatter import scatter_add as _ts_add, scatter_max as _ts_max
        _HAS_TORCH_SCATTER = True
    except Exception:  # pragma: no cover
        _ts_add = None
        _ts_max = None
        _HAS_TORCH_SCATTER = False

# rebind when disabled
if not _HAS_TORCH_SCATTER:
    _ts_add = None
    _ts_max = None


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = 0,
                dim_size: int | None = None) -> torch.Tensor:
    """``torch_scatter.scatter_add`` with a pure-PyTorch fallback."""
    if _HAS_TORCH_SCATTER:
        return _ts_add(src, index, dim=dim, dim_size=dim_size)
    assert dim == 0, "fallback supports dim=0 only"
    assert src.dim() == 2, "fallback supports [N, F] sources only"
    out = src.new_zeros((dim_size, src.size(1)))
    out.index_add_(0, index, src)
    return out


def scatter_max(src: torch.Tensor, index: torch.Tensor, dim: int = 0,
                dim_size: int | None = None):
    """``torch_scatter.scatter_max`` with a pure-PyTorch fallback.

    Returns ``(values, argmax)`` where ``argmax`` is the *global* index of the
    maximum, matching ``torch_scatter``.  Nodes without any incoming edge
    carry the value ``-inf`` and argmax ``-1``.
    """
    if _HAS_TORCH_SCATTER:
        return _ts_max(src, index, dim=dim, dim_size=dim_size)
    assert dim == 0, "fallback supports dim=0 only"
    assert src.dim() == 1, "fallback supports 1-D sources only"
    neg_inf = torch.finfo(src.dtype).min
    one_hot = src.new_zeros(src.shape[0], dim_size)
    one_hot.scatter_(1, index.unsqueeze(1), 1.0)
    masked = torch.where(one_hot > 0, src.unsqueeze(1), neg_inf)
    first_max, max_idx = masked.max(dim=0)
    has = one_hot.sum(dim=0) > 0
    argmax = torch.where(has, max_idx.long(),
                         src.new_full((), -1, dtype=torch.long))
    values = torch.where(has, first_max,
                         src.new_full((), neg_inf, dtype=src.dtype))
    return values, argmax


def has_torch_scatter() -> bool:
    return _HAS_TORCH_SCATTER
