"""
Checkpoint format for WIĄZKA (PLAN.md 3.4).

A checkpoint is a directory:

    ckpt/
      model.pt          model state dict + relation mapping + spec
      config.json       human readable training + model configuration
      rel2idx.json      relation name -> id (also inside model.pt)
      commutation_C.csv / .npy   learned commutation matrix C_rs (artifact)
      metrics.json      last evaluation metrics, when available

The spec stored inside ``model.pt`` is *declarative* (no pickled callables),
so a checkpoint loads without the producing code holding registry state.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

CHECKPOINT_FORMAT = "wiazka-ckpt-1"
_MODEL_FILE = "model.pt"
_CONFIG_FILE = "config.json"
_REL2IDX_FILE = "rel2idx.json"
_METRICS_FILE = "metrics.json"
_C_FILE = "commutation_C"


def _dumpable(value):
    """Recursively convert a value into JSON-serialisable primitives."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _dumpable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_dumpable(v) for v in value]
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    return str(value)


def save_checkpoint(
    ckpt_dir: str,
    model,
    rel2idx: Dict[str, int],
    config: Optional[Dict[str, Any]] = None,
    commutation: Optional[torch.Tensor] = None,
    metrics: Optional[Dict[str, float]] = None,
) -> str:
    """Persist a trained model; returns ``ckpt_dir``."""
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "model": model.state_dict(),
            "rel2idx": dict(rel2idx),
            "spec": model.spec(),
        },
        os.path.join(ckpt_dir, _MODEL_FILE),
    )
    payload = {
        "format": CHECKPOINT_FORMAT,
        "model_spec": _dumpable(model.spec()),
    }
    if config:
        payload["config"] = _dumpable(config)
    with open(os.path.join(ckpt_dir, _CONFIG_FILE), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)

    with open(os.path.join(ckpt_dir, _REL2IDX_FILE), "w", encoding="utf-8") as f:
        json.dump({str(k): int(v) for k, v in rel2idx.items()}, f, indent=2,
                  ensure_ascii=False, sort_keys=True)

    if commutation is not None:
        c = (commutation.detach().cpu().numpy()
             if isinstance(commutation, torch.Tensor)
             else np.asarray(commutation))
        np.save(os.path.join(ckpt_dir, f"{_C_FILE}.npy"), c)
        names = [k for _, k in sorted(((v, k) for k, v in rel2idx.items()))]
        np.savetxt(
            os.path.join(ckpt_dir, f"{_C_FILE}.csv"), c,
            delimiter=",", header="," + ",".join(str(n) for n in names),
            comments="", fmt="%.6f",
        )

    if metrics:
        with open(os.path.join(ckpt_dir, _METRICS_FILE), "w", encoding="utf-8") as f:
            json.dump(_dumpable(metrics), f, indent=2, sort_keys=True)
    return ckpt_dir


def load_checkpoint(
    ckpt_dir: str, device: str = "cpu"
) -> Tuple[Any, Dict[str, int], Dict[str, Any], Dict[str, Any]]:
    """Load ``(model, rel2idx, spec, meta)`` written by :func:`save_checkpoint`."""
    path = os.path.join(ckpt_dir, _MODEL_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; not a WIĄZKA checkpoint")
    bundle = torch.load(path, map_location=device, weights_only=True)
    fmt = bundle.get("format")
    if fmt != CHECKPOINT_FORMAT:
        raise ValueError(
            f"unsupported checkpoint format {fmt!r}; expected {CHECKPOINT_FORMAT!r}"
        )
    from relgat_projector.core.model.wiazka import WiazkaModel

    spec = bundle["spec"]
    state = bundle["model"]
    # the graph buffers are part of the state dict: size the placeholder from it
    node_shape = state["node_emb_fixed"].shape
    edge_shape = state["edge_index"].shape
    model = WiazkaModel.from_spec(
        spec,
        node_emb=torch.zeros(*node_shape),
        edge_index=torch.zeros(*edge_shape, dtype=torch.long),
        edge_type=torch.zeros(*state["edge_type"].shape, dtype=torch.long),
    )
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    meta: Dict[str, Any] = {}
    cfg_path = os.path.join(ckpt_dir, _CONFIG_FILE)
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            meta = json.load(f)
    return model, bundle["rel2idx"], spec, meta


def load_commutation(ckpt_dir: str) -> Optional[np.ndarray]:
    path = os.path.join(ckpt_dir, f"{_C_FILE}.npy")
    return np.load(path) if os.path.exists(path) else None
