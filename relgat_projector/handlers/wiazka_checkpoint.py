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
            delimiter=",",
            header="," + ",".join(str(n).replace(",", ";") for n in names),
            comments="", fmt="%.6f",
        )

    if metrics:
        with open(os.path.join(ckpt_dir, _METRICS_FILE), "w", encoding="utf-8") as f:
            json.dump(_dumpable(metrics), f, indent=2, sort_keys=True)
    return ckpt_dir


#: State-dict entries a model may be missing when loading an older checkpoint.
#: Each entry must be a parameter whose constructor default reproduces the
#: behaviour the model had before the parameter existed.
_TOLERATED_MISSING = frozenset({
    "transport_dial",
    # an uncentred chart has no affine bias and no centre buffer; a
    # checkpoint written before the base-centring option existed simply
    # keeps its (zero) documented initialisation for both
    "base_to_op",
    "center_mean",
})


def apply_calibrated_scalars(model, meta: Optional[Dict[str, Any]]) -> None:
    """Restore the scalars that were *fitted* during training, not learned.

    ``relation_prior_strength`` and ``relation_ensemble_weight`` are ordinary
    attributes, so they are not in the state dict and a model rebuilt from a
    checkpoint would score with the constructor defaults.  That is not neutral:
    the default prior ``1.0`` is the Bayes correction of an *unweighted* head,
    and applied to a class-balanced head it takes relation naming from macro
    0.47 to 0.05 (the balanced head already fits ``p(r|x) w(r)``).  A checkpoint
    written before the scalars were fitted gets the value its own head training
    implies -- balanced head: no correction, unweighted head: the classic one.

    Every loader has to call this, or the model a caller gets is not the model
    the run's report measured.
    """
    run_config = ((meta or {}).get("config") or {})
    for name in ("relation_ensemble_weight", "relation_prior_strength"):
        fitted = run_config.get(name)
        if fitted is None:
            if name == "relation_prior_strength":
                balanced = float(run_config.get("decoder_balance", 0.0)) > 0.0
                fitted = 0.0 if balanced else 1.0
            else:
                continue          # a neutral mix is the pre-fit behaviour
        if hasattr(model, name):
            setattr(model, name, float(fitted))


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
    # Strict in everything but the parameters this format gained *after* it was
    # written: ``strict=True`` alone makes an old checkpoint unreadable the
    # moment the model grows a tensor, and silently swallowing every missing key
    # would hide a genuinely truncated file.  A key is tolerated only when it is
    # listed below, and the model keeps its documented initialisation for it --
    # for ``transport_dial`` that initialisation is "rewrite at full strength",
    # i.e. exactly the behaviour a checkpoint written before the dial existed
    # had.
    result = model.load_state_dict(state, strict=False)
    unexpected = list(result.unexpected_keys)
    missing = [k for k in result.missing_keys if k.split(".")[0] not in _TOLERATED_MISSING]
    if unexpected or missing:
        raise RuntimeError(
            "checkpoint does not match the model architecture: "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    model = model.to(device)
    model.eval()

    meta: Dict[str, Any] = {}
    cfg_path = os.path.join(ckpt_dir, _CONFIG_FILE)
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            meta = json.load(f)
    apply_calibrated_scalars(model, meta)
    return model, bundle["rel2idx"], spec, meta


def load_commutation(ckpt_dir: str) -> Optional[np.ndarray]:
    path = os.path.join(ckpt_dir, f"{_C_FILE}.npy")
    return np.load(path) if os.path.exists(path) else None
