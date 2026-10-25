"""
WIĄZKA trainer runner (PLAN.md).

Trains the relational gauge-transport model end to end on a semantic graph
(Słowosieć / WordNet / any ontology exported by plwordnet-milvus) and saves a
full checkpoint (model + config + learned commutation matrix C_rs + metrics).

Example:

    relgat-projector-wiazka \
        --nodes-embeddings-path=.../lexical_units_embedding.pickle \
        --relations-mapping=.../relation_to_idx.json \
        --relations-triplets=.../relations_triplets.json \
        --epochs=20 --batch-size=128 --train-ratio=0.9 \
        --save-dir=wiazka-models/run1
"""

import argparse
import dataclasses
import json
import os
import sys
from typing import Optional

import torch

from relgat_projector.handlers.wiazka_dataset import (
    load_graph_dataset,
    load_shadow_vectors,
)
from relgat_projector.trainer.wiazka_trainer import WiazkaConfig, WiazkaTrainer

APP_DESCRIPTION = """
WIĄZKA / RGT relational transport trainer.

Consumes the plwordnet-milvus exporter output (nodes pickle, relation
mapping json, triplet json) and runs the F1->F4 curriculum:
nodes, k=1 relations, path composition + holonomy, then the off-graph
(shadow) manifold. Saves a checkpoint loadable with RelationFlow.load().
"""


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=APP_DESCRIPTION)

    # dataset
    parser.add_argument("--nodes-embeddings-path", type=str, required=True)
    parser.add_argument("--relations-mapping", type=str, required=True)
    parser.add_argument("--relations-triplets", type=str, required=True)
    parser.add_argument(
        "--shadow-vectors", type=str, default=None,
        help="Optional .npy/.pickle with off-graph text vectors for the F4 stage",
    )

    # operator / architecture
    parser.add_argument("--operator", type=str, default="givens",
                        choices=["givens", "distmult", "transe"])
    parser.add_argument("--op-dim", type=int, default=256)
    parser.add_argument("--num-generators", type=int, default=16)
    parser.add_argument("--gat-num-layers", type=int, default=2)
    parser.add_argument("--gat-heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no-scale", action="store_true",
                        help="disable the learned per-relation scale factor")
    parser.add_argument("--no-connection", action="store_true",
                        help="keep the transport position-independent: skip the "
                             "F4 base-point-dependent connection (ablation; the "
                             "curriculum and every other loss term are "
                             "unchanged)")
    parser.add_argument("--no-gat-attention", action="store_true",
                        help="sum messages instead of relation-aware softmax "
                             "attention (legacy behaviour)")
    parser.add_argument("--no-proj-pca", action="store_true",
                        help="random instead of principal-subspace init of the "
                             "base<->operator projection")
    parser.add_argument("--probe-size", type=int, default=1024,
                        help="held-out edges scored at every evaluation")
    parser.add_argument("--anchor-weight", type=float, default=0.5,
                        help="weight of the L_anchor pull towards the base embedding")
    parser.add_argument("--anchor-margin", type=float, default=0.05,
                        help="allowed 1-cos(refined, base) before L_anchor bites")
    parser.add_argument("--path-weight", type=float, default=1.0,
                        help="weight of the composition loss L_path (PCE driver)")
    parser.add_argument("--pull-weight", type=float, default=1.0,
                        help="weight of the production loss L_pull (exported "
                             "query must retrieve the gold node from the base "
                             "index)")
    parser.add_argument("--pull-batch", type=int, default=256,
                        help="queries per step in L_pull (keys = every node)")
    parser.add_argument("--contrast-temperature", type=float, default=0.07,
                        help="softmax temperature of the contrastive terms "
                             "(L_con, L_pull); queries/keys are unit-norm, so "
                             "a smaller value weights the hard negatives more")
    parser.add_argument("--balance-power", type=float, default=0.5,
                        help="temper of relation-balanced sampling: relation p "
                             "proportional to count**power "
                             "(1.0 = raw edge distribution, 0.0 = uniform)")
    parser.add_argument("--decoder-balance", type=float, default=0.5,
                        help="class-weight temper of the relation decoder CE: "
                             "weight proportional to count**-power "
                             "(0.0 = plain CE, 1.0 = full inverse frequency)")
    parser.add_argument("--decoder-label-smoothing", type=float, default=0.05,
                        help="label smoothing of the relation decoder CE")
    parser.add_argument("--knn-weight", type=float, default=0.0,
                        help="weight of L_knn, the structure-preserving "
                             "refinement hinge on base-space neighbour pairs "
                             "(0 disables it)")
    parser.add_argument("--knn-k", type=int, default=8,
                        help="base-space neighbours preserved per anchor node")
    parser.add_argument("--knn-slack", type=float, default=0.02,
                        help="tolerated cos drop on a base-neighbour pair")

    # curriculum / training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-neg", type=int, default=8)
    parser.add_argument("--train-ratio", type=float, default=0.90,
                        help="fraction of edges used for training; the rest "
                             "is held out from the GAT topology")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log-every-n-steps", type=int, default=10)
    parser.add_argument("--lr-scheduler", type=str, default="cosine",
                        choices=["constant", "cosine", "linear"])
    parser.add_argument("--warmup-ratio", type=float, default=0.05,
                        help="fraction of total steps used for linear warmup")
    parser.add_argument("--num-shadow", type=int, default=512,
                        help="shadow vectors synthesised when none are supplied")
    parser.add_argument("--shadow-jitter", type=float, default=0.35)

    # query rewriting validation (PLAN.md 5, QR-Recall)
    parser.add_argument("--qr-relations", type=str, default=None,
                        help="comma-separated relations to validate with "
                             "QR-Recall, e.g. 'syn,ant'")
    parser.add_argument("--qr-ks", type=str, default="1,5,10")
    parser.add_argument("--qr-max-queries", type=int, default=512)
    parser.add_argument("--qr-train", action="store_true",
                        help="evaluate QR-Recall on training edges instead of "
                             "the held-out set (for debugging only)")

    # output
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--report", type=str, default=None,
                        help="optional path for a JSON summary (defaults to <save-dir>/report.json)")

    return parser.parse_args()


def _fmt_metrics(history) -> str:
    if not history:
        return "(no epochs completed)"
    last = history[-1]
    keep = [k for k in last if k.startswith(("eval/", "L_"))]
    lines = [
        f"  {k:<20} {last[k]:>10.5f}"
        for k in sorted(keep)
    ]
    return "\n".join(lines)


def _print_per_relation(rows) -> None:
    if not rows:
        print("  (no held-out relations scored)")
        return
    print(f"  {'relation':<44} {'n':>4} {'raw':>7} {'ident':>7} "
          f"{'model':>7} {'export':>7} {'lift':>7} {'cls':>6} {'lam':>5}")
    for r in rows:
        print(f"  {r['relation'][:44]:<44} {r['n']:>4d} {r['mrr_raw']:>7.3f} "
              f"{r['mrr_id']:>7.3f} {r['mrr_model']:>7.3f} "
              f"{r['mrr_prod']:>7.3f} {r['lift_over_raw']:>+7.3f} "
              f"{r.get('relation_acc', float('nan')):>6.2f} "
              f"{r.get('transport_lambda', float('nan')):>5.2f}")
    macro = sum(r["mrr_model"] for r in rows) / len(rows)
    macro_raw = sum(r["mrr_raw"] for r in rows) / len(rows)
    macro_cls = sum(r.get("relation_acc", 0.0) for r in rows) / len(rows)
    macro_lam = sum(r.get("transport_lambda", 0.0) for r in rows) / len(rows)
    print(f"  {'MACRO (mean over relations)':<44} {'':>4} {macro_raw:>7.3f} "
          f"{'':>7} {macro:>7.3f} {'':>7} {macro - macro_raw:>+7.3f} "
          f"{macro_cls:>6.2f} {macro_lam:>5.2f}")


def _print_qr(qr, ks) -> None:
    for name, row in qr.items():
        if not row or row.get("n_queries", 0) == 0:
            print(f"  {name}: no gold pairs in the scored set")
            continue
        print(f"  relation {name} (n={int(row['n_queries'])})")
        print(f"  {'':<15}" + "".join(f"@{k:<6}" for k in ks))
        for label, key in (("recall before", "recall_before@"),
                           ("recall after", "recall_after@"),
                           ("delta", "delta_recall@")):
            print(f"  {label:<15}" + "".join(
                f"{row.get(key + str(k), float('nan')):<6.2f}" for k in ks))
        print(f"  MRR before={row['mrr_before']:.3f} "
              f"after={row['mrr_after']:.3f}")


def main() -> None:
    args = get_args()

    node2emb, rel2idx, edges = load_graph_dataset(
        args.nodes_embeddings_path,
        args.relations_mapping,
        args.relations_triplets,
    )
    base_dim = int(next(iter(node2emb.values())).numel())
    shadow = load_shadow_vectors(args.shadow_vectors, dim=base_dim)

    cfg = WiazkaConfig(
        op_dim=args.op_dim,
        num_generators=args.num_generators,
        operator_type=args.operator,
        gat_num_layers=args.gat_num_layers,
        gat_heads=args.gat_heads,
        dropout=args.dropout,
        use_scale=not args.no_scale,
        enable_connection=not args.no_connection,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_neg=args.num_neg,
        epochs=args.epochs,
        device=args.device,
        seed=args.seed,
        train_ratio=args.train_ratio,
        gat_attention=not args.no_gat_attention,
        anchor_weight=args.anchor_weight,
        path_weight=args.path_weight,
        pull_weight=args.pull_weight,
        pull_batch=args.pull_batch,
        anchor_margin=args.anchor_margin,
        contrast_temperature=args.contrast_temperature,
        balance_power=args.balance_power,
        decoder_balance=args.decoder_balance,
        decoder_label_smoothing=args.decoder_label_smoothing,
        knn_weight=args.knn_weight,
        knn_k=args.knn_k,
        knn_slack=args.knn_slack,
        proj_pca=not args.no_proj_pca,
        probe_size=args.probe_size,
        num_shadow=args.num_shadow,
        shadow_jitter=args.shadow_jitter,
        lr_scheduler=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
    )
    print(f"[wiazka] config: {dataclasses.asdict(cfg)}")

    cfg.save_dir = args.save_dir
    trainer = WiazkaTrainer(node2emb, rel2idx, edges, cfg, shadow_vectors=shadow)
    trainer.train(epochs=cfg.epochs, log_every=args.log_every_n_steps)

    idx2rel = {v: k for k, v in rel2idx.items()}

    def _score(tag: str) -> dict:
        """Measure the weights currently in ``trainer.model``, both dial variants.

        The dial (per-relation rewriting strength ``lam_r``) is fitted on
        **train** edges through the exact served protocol, so it never touches
        the held-out probe the reported numbers come from; both the trained and
        the calibrated value are reported, so any overfitting is visible.
        """
        trained = trainer.evaluate()
        print(f"\n=== {tag}: evaluation (trained dial) ===")
        print(_fmt_metrics([trained]))
        fit = trainer.calibrate_transport_dial()
        print(f"=== {tag}: dial calibration (train edges, per-relation lam) ===")
        for r in sorted(fit, key=lambda r: idx2rel.get(r, "")):
            print(f"  {idx2rel.get(r, r):<44} lam={fit[r]:.2f}")
        calibrated = trainer.evaluate()
        print(f"\n=== {tag}: evaluation (calibrated dial) ===")
        print(_fmt_metrics([calibrated]))
        return {"trained": trained, "calibrated": calibrated, "dial_fit": fit}

    results = {"last": _score("last epoch")}
    last_state = {k: v.detach().clone()
                  for k, v in trainer.model.state_dict().items()}

    # training keeps the best-scoring checkpoint under <save-dir>/best, but the
    # run used to report (and ship) whatever weights the last epoch happened to
    # leave behind.  Score that checkpoint with the very criterion training used
    # and keep the better one: the reported numbers then describe the model
    # ``RelationFlow.load`` actually returns.
    best_dir = f"{args.save_dir}/best"
    if os.path.isdir(best_dir) and getattr(trainer, "best_step", 0) > 0:
        try:
            trainer.load_weights_from(best_dir)
            results["best"] = _score("best checkpoint")
        except Exception as exc:  # a broken artifact must not lose the run
            print(f"[ckpt] could not reload {best_dir}: {exc}")

    chosen = max(
        results,
        key=lambda k: WiazkaTrainer.selection_score(results[k]["calibrated"]),
    )
    if chosen == "last":
        trainer.model.load_state_dict(last_state)
        trainer.model.eval()
    else:
        trainer.load_weights_from(best_dir)
    print("\n[ckpt] shipping \"%s\" checkpoint (score %s)" % (
        chosen, ", ".join(
            f"{k}={WiazkaTrainer.selection_score(v['calibrated']):.4f}"
            for k, v in results.items())))

    # The artifact has to be the model the report measured.  Selecting a
    # checkpoint reloads its *trained* dial (the calibration lives in the model,
    # and reloading overwrites it), while the numbers chosen above are the
    # calibrated ones -- so from this point until the next calibration the model
    # in memory and the metrics describing it disagree.  Refitting here makes
    # the per-relation table, the QR-Recall block and the saved weights all
    # describe the dial the headline metrics were measured under.  Measured on
    # F1: prod_mrr 0.2345 / prod_mrr_traffic 0.4911 for the reported dial,
    # 0.2296 / 0.4468 for the dial the checkpoint used to carry.
    dial_fit = results[chosen]["dial_fit"]
    trainer.calibrate_transport_dial()
    lam = torch.sigmoid(trainer.model.transport_dial.detach().float()).cpu()
    moved = [r for r, v in dial_fit.items()
             if abs(v - float(lam[int(r)])) > 0.06]
    print(f"[ckpt] dial restored for the shipped weights: "
          f"{len(dial_fit) - len(moved)}/{len(dial_fit)} relations match the "
          f"calibration the reported metrics were measured under"
          + ("" if not moved else f" (differs for {len(moved)})"))
    metrics = results[chosen]["calibrated"]
    metrics_trained = results[chosen]["trained"]

    print("\n=== per-relation filtered MRR (held-out, calibrated dial) ===")
    per_rel = trainer.per_relation_report()
    _print_per_relation(per_rel)

    qr = None
    if args.qr_relations:
        rels = [r.strip() for r in args.qr_relations.split(",") if r.strip()]
        ks = tuple(int(k) for k in args.qr_ks.split(",") if k.strip())
        print("\n=== query rewriting (QR-Recall) ===")
        qr = trainer.qr_eval(
            rels,
            ks=ks,
            max_queries=args.qr_max_queries,
            use_heldout=not args.qr_train,
        )
        _print_qr(qr, ks)

    ckpt = trainer.save(args.save_dir, metrics=metrics)
    if chosen == "best":
        trainer.save(best_dir, metrics=metrics)

    commutation = trainer.curvature_loss.commutation().detach().cpu()
    names = sorted(rel2idx.items(), key=lambda kv: kv[1])
    names = [k for k, _ in names]
    print("\n=== learned commutation matrix C_rs (1 = commute, 0 = not) ===")
    c = commutation.numpy()
    for i, ri in enumerate(names):
        row = "  ".join(f"{c[i, j]:5.2f}" for j in range(len(names)))
        print(f"  {ri:<12} {row}")

    report_path = args.report or f"{args.save_dir}/report.json"
    report = {
        "config": dataclasses.asdict(cfg),
        "num_nodes": trainer.num_nodes,
        "num_edges_train": len(trainer.edges),
        "num_edges_heldout": len(trainer.eval_edges),
        "topology_endpoint_coverage": trainer.topology_endpoint_coverage,
        "final_stage": trainer.stage_manager.order[trainer.stage_manager.current],
        "best_step": getattr(trainer, "best_step", None),
        "best_score": getattr(trainer, "best_score", None),
        # per-epoch loss terms + eval metrics: the only place a dead loss
        # term or a stalled metric is visible across a whole run
        "history": trainer.history,
        "final_metrics": metrics,
        "final_metrics_trained_dial": metrics_trained,
        "shipped_checkpoint": chosen,
        "checkpoint_scores": {
            k: WiazkaTrainer.selection_score(v["calibrated"])
            for k, v in results.items()
        },
        "metrics_by_checkpoint": {
            k: {"trained_dial": v["trained"], "calibrated_dial": v["calibrated"]}
            for k, v in results.items()
        },
        "dial_calibration": {
            str(idx2rel.get(r, r)): lam for r, lam in dial_fit.items()
        },
        "per_relation": per_rel,
        "commutation_C": {
            f"{names[i]}|{names[j]}": float(commutation[i, j])
            for i in range(len(names))
            for j in range(len(names))
        },
        "checkpoint": ckpt,
        "qr_recall": qr,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[wiazka] checkpoint: {ckpt}")
    print(f"[wiazka] report:     {report_path}")


if __name__ == "__main__":
    main()
