"""
Scatter primitives with an optional ``torch_scatter`` backend.

``torch_scatter`` ships as a precompiled C++ extension that is ABI-locked to
the exact PyTorch minor version it was built against.  Loading it under a
different PyTorch **segfaults the interpreter** -- no ``try/except`` can catch
it, so an incompatible wheel silently kills every run at import time (observed
with ``torch_scatter 2.1.2+pt28cu128`` under ``torch 2.9.0``).

Therefore the extension is imported only when its build tag matches the
installed PyTorch.  Otherwise this module uses the pure-PyTorch
implementations below, which cover the 1-D-index / ``dim=0`` case used by the
GAT layers and match ``torch_scatter.scatter_add`` / ``scatter_max`` semantics
(nodes without any incoming edge carry ``-inf`` and argmax ``-1``).

Environment switches:

* ``RELGAT_USE_TORCH_SCATTER=1`` -- force the extension even when the tags
  disagree (use only after verifying the wheel matches your PyTorch);
* ``RELGAT_NO_TORCH_SCATTER=1`` / ``RELgat_NO_TORCH_SCATTER=1`` -- always use
  the built-in implementations.
"""

import os
import re

import torch


def _torch_scatter_abi_ok() -> bool:
    """True when the installed wheel is built for the installed PyTorch.

    Reads only the distribution metadata: importing the package itself is
    exactly what crashes on an ABI mismatch.
    """
    try:
        from importlib.metadata import version

        build = version("torch-scatter")
    except Exception:
        return False  # not installed (or metadata unreadable)
    tag = re.search(r"pt(\d)(\d)", build)
    if not tag:
        # a plain "2.1.2" wheel carries no tag: assume the user built it locally
        return True
    built_for = f"{tag.group(1)}.{tag.group(2)}"
    installed = ".".join(torch.__version__.split(".")[:2])
    return built_for == installed


def _scatter_backend_requested() -> bool:
    if os.environ.get("RELGAT_NO_TORCH_SCATTER") == "1" or os.environ.get(
        "RELgat_NO_TORCH_SCATTER"
    ) == "1":
        return False
    if os.environ.get("RELGAT_USE_TORCH_SCATTER") == "1":
        return True
    return _torch_scatter_abi_ok()


_ts_add = None
_ts_max = None
_HAS_TORCH_SCATTER = False
if _scatter_backend_requested():
    try:  # pragma: no cover - environment dependent
        from torch_scatter import scatter_add as _ts_add, scatter_max as _ts_max
        _HAS_TORCH_SCATTER = True
    except Exception:  # pragma: no cover
        _ts_add = None
        _ts_max = None
        _HAS_TORCH_SCATTER = False


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
    """True when the compiled ``torch_scatter`` extension is in use."""
    return _HAS_TORCH_SCATTER
