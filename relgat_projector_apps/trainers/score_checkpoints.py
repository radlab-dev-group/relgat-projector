"""
Score WIĄZKA checkpoints with one protocol, one probe, one code build.

Reports written by different revisions of the trainer are not comparable
(the ranking protocol and the probe changed), so a final comparison must
re-load every checkpoint with the *current* code and score it once.  This
app does exactly that and, per checkpoint, reports two rows:

* ``shipped`` -- the calibration (dial / prior / alpha) the run saved,
* ``refit``   -- the calibration refitted on train edges with the current
                 code (``calibrate_transport_dial`` + prior + alpha).

Both rows are measured on the same held-out probe, so the difference
between them is pure calibration value.  The tool also prints the
per-relation table and the QR-Recall numbers of the refitted model.

Usage (one process at a time; it takes the GPU the training run is not on
the moment you need it for a minute or two per checkpoint)::

    python -m relgat_projector_apps.trainers.score_checkpoints \\
        wiazka-models/F1_20261025_080926/best \\
        wiazka-models/F2_20261025_105002/best \\
        --probe 1024
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

import torch

from relgat_projector.handlers.wiazka_dataset import load_graph_dataset
from relgat_projector.handlers.wiazka_checkpoint import load_checkpoint
from relgat_projector.trainer.wiazka_trainer import WiazkaConfig, WiazkaTrainer

DEFAULT_QR_RELATIONS = "synonimia,antonimia właściwa,fuzzynimia,derywacyjność"

#: (short name, evaluate() key) -- the columns of every row
KEYS = [
    ("mrr", "eval/mrr"),
    ("macro", "eval/mrr_macro"),
    ("id", "eval/id_mrr"),
    ("chart", "eval/chart_mrr"),
    ("raw", "eval/raw_mrr"),
    ("prod", "eval/prod_mrr"),
    ("lift", "eval/mrr_prod_lift"),
    ("h1", "eval/hits@1"),
    ("path", "eval/path_lift"),
    ("rel", "eval/rel_acc_macro_ensemble"),
    ("rel_dec", "eval/rel_acc_macro"),
    ("prior", "eval/rel_prior_strength"),
    ("alpha", "eval/rel_ensemble_alpha"),
    ("prod_tr", "eval/prod_mrr_traffic"),
    ("raw_tr", "eval/raw_mrr_traffic"),
    ("lift_tr", "eval/mrr_prod_lift_traffic"),
    # dial-invariant: the best global lambda, i.e. what the checkpoint ships
    # regardless of how well the per-relation dial happened to be fitted
    ("prod_tr_bl", "eval/prod_mrr_traffic_bestlam"),
    ("lam_tr", "eval/prod_lambda_traffic"),
    ("score", None),
]


def selection_row(metrics) -> float:
    return WiazkaTrainer.selection_score(metrics)


def build_trainer(
    ckpt_dir: str,
    node2emb: Dict[int, torch.Tensor],
    rel2idx: Dict[str, int],
    edges: List[tuple],
    device: str,
    probe: int,
    calib_ratio: float | None = None,
) -> WiazkaTrainer:
    """Load a checkpoint into a trainer whose split matches the run's.

    ``calib_ratio`` overrides the run's own value, which is how an A/B of two
    calibration sources is measured on one set of weights.
    """
    model, _r, spec, meta = load_checkpoint(ckpt_dir, device=device)
    run_cfg = meta.get("config", {})
    cfg = WiazkaConfig(**{k: v for k, v in run_cfg.items()
                          if k in WiazkaConfig.__dataclass_fields__})
    cfg.device = device
    # the run's split: the probe must be drawn from the same held-out edges,
    # and the calibration slice (if any) from the same part of them
    cfg.train_ratio = float(run_cfg.get("train_ratio", 0.9))
    cfg.calib_ratio = float(run_cfg.get("calib_ratio", 0.0))
    if calib_ratio is not None:
        cfg.calib_ratio = float(calib_ratio)
    cfg.probe_size = probe
    trainer = WiazkaTrainer(node2emb, rel2idx, edges, cfg)
    # the F4 connection creates ``ctx_W``/``ctx_P``/``log_eps`` when it is
    # switched on, so a checkpoint that carries them only loads into a model
    # whose stage has them too; the saved ``use_context`` says which one that is
    if bool(run_cfg.get("use_context", False)) and not bool(
            getattr(trainer.model.operator, "use_context", False)):
        trainer.stage_manager.current = len(trainer.stage_manager.order) - 1
        trainer._sync_stage_state()
    trainer.load_weights_from(ckpt_dir)
    return trainer


def resave_calibration(ckpt_dir: str, trainer: WiazkaTrainer) -> str:
    """Re-write a checkpoint with its refitted calibration scalars.

    The dial is a model parameter (it travels in the state dict) but the
    prior/alpha strengths are plain attributes, so refitting them means
    writing them into ``config.json``.  Everything else in that file is kept
    verbatim -- a scoring tool must not silently restate how the run was
    configured (its probe size, its save dir, its curriculum stage).
    """
    from relgat_projector.handlers.wiazka_checkpoint import save_checkpoint

    meta = load_checkpoint(ckpt_dir, device="cpu")[3]
    config = dict(meta.get("config", {}))
    config["relation_prior_strength"] = float(
        trainer.model.relation_prior_strength)
    config["relation_ensemble_weight"] = float(
        trainer.model.relation_ensemble_weight)
    # ``commutation=None`` on purpose: ``C_rs`` is owned by the trainer's
    # CurvatureLoss and is NOT part of the model state dict, so a scoring tool
    # -- which rebuilt the trainer and never loaded ``C_rs`` -- would otherwise
    # write its freshly initialised matrix over the learned one.
    return save_checkpoint(
        ckpt_dir, trainer.model, trainer.rel2idx, config=config,
        commutation=None)


def pick_row(trainer: WiazkaTrainer, probe: int) -> Dict[str, float]:
    metrics = trainer.evaluate(n_probe=probe)
    row = {short: float(metrics[key]) for short, key in KEYS if key}
    row["score"] = WiazkaTrainer.selection_score(metrics)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoints", nargs="+", help="checkpoint dirs (e.g. <run>/best)")
    parser.add_argument("--nodes-embeddings-path", required=True)
    parser.add_argument("--relations-mapping", required=True)
    parser.add_argument("--relations-triplets", required=True)
    parser.add_argument("--probe", type=int, default=1024,
                        help="held-out probe size (stratified, (src, rel)-unique)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--only-calibration", action="store_true",
                        help="refit dial/prior/alpha on train edges and re-save "
                             "them into the checkpoint (metrics row omitted)")
    parser.add_argument("--calib-ratio", type=float, default=None,
                        help="override the run's calib_ratio (fraction of the "
                             "held-out edges used to fit dial/prior/alpha; "
                             "0 = train edges)")
    parser.add_argument("--qr-relations", default=DEFAULT_QR_RELATIONS,
                        help="comma-separated relations for QR-Recall")
    parser.add_argument("--qr-max-queries", type=int, default=256)
    parser.add_argument("--out", default=None,
                        help="write the full JSON to this path")
    args = parser.parse_args()

    node2emb, rel2idx, edges = load_graph_dataset(
        args.nodes_embeddings_path, args.relations_mapping, args.relations_triplets)

    results: Dict[str, Any] = {}
    for ckpt in args.checkpoints:
        tag = ckpt.rstrip("/").split("/")[-2] if ckpt.rstrip("/").endswith("best") \
            else ckpt.rstrip("/").split("/")[-1]
        trainer = build_trainer(ckpt, node2emb, rel2idx, edges, args.device,
                                args.probe, calib_ratio=args.calib_ratio)

        entry: Dict[str, Any] = {}
        if args.only_calibration:
            refined = trainer.model.refine()
            stats = trainer._transport_prior_stats(refined)
            counts = trainer._relation_counts.to(refined.device)
            prior = trainer.fit_relation_prior_strength(refined, counts)
            alpha = trainer.fit_relation_ensemble_weight(refined, stats)
            fit = trainer.calibrate_transport_dial()
            row = pick_row(trainer, args.probe)
            resave_calibration(ckpt, trainer)
            entry["calibration"] = {
                "prior": round(prior, 3), "alpha": round(alpha, 3),
                "dial_mean": round(sum(fit.values()) / max(1, len(fit)), 3),
                "dial_zeros": sum(1 for x in fit.values() if x <= 0.05),
                "dial_high": sum(1 for x in fit.values() if x >= 0.95),
            }
            entry["metrics_after"] = {k: round(v, 4) for k, v in row.items()}
            print(f"--- {tag}: calibration refitted and re-saved {entry['calibration']}")
            entry["dial_refit"] = {str(k): round(float(v), 3)
                                   for k, v in sorted(fit.items())}
            print(f"--- {tag}: calibration refitted and re-saved "
                  f"(mean lambda {sum(fit.values()) / max(1, len(fit)):.3f})")
        else:
            entry["shipped"] = pick_row(trainer, args.probe)
            fit = trainer.calibrate_transport_dial()
            entry["refit"] = pick_row(trainer, args.probe)
            lams = list(fit.values())
            entry["dial"] = {
                "mean": round(sum(lams) / max(1, len(lams)), 3),
                "zeros": sum(1 for x in lams if x <= 0.05),
                "high": sum(1 for x in lams if x >= 0.95),
            }
            per_rel = trainer.per_relation_report(n_per_rel=48)
            entry["per_relation"] = [
                {k: (round(v, 4) if isinstance(v, float) else v)
                 for k, v in row.items()} for row in per_rel
            ]
            qr_names = [n for n in args.qr_relations.split(",") if n]
            entry["qr"] = trainer.qr_eval(qr_names,
                                          max_queries=args.qr_max_queries)
        results[tag] = entry
        if not args.only_calibration:
            for which in ("shipped", "refit"):
                print(f"--- {tag} {which}")
                print("    " + " ".join(f"{k}={v:.4f}"
                                        for k, v in results[tag][which].items()))
            print(f"    dial {results[tag]['dial']}")

    print(json.dumps(results, indent=1, ensure_ascii=False))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1, ensure_ascii=False)
        print(f"[score] JSON written to {args.out}")


if __name__ == "__main__":
    main()
