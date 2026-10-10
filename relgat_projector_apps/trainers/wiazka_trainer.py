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
import sys
from typing import Optional

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
    parser.add_argument("--num-shadow", type=int, default=512,
                        help="shadow vectors synthesised when none are supplied")
    parser.add_argument("--shadow-jitter", type=float, default=0.35)

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
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_neg=args.num_neg,
        epochs=args.epochs,
        device=args.device,
        seed=args.seed,
        train_ratio=args.train_ratio,
        num_shadow=args.num_shadow,
        shadow_jitter=args.shadow_jitter,
    )
    print(f"[wiazka] config: {dataclasses.asdict(cfg)}")

    trainer = WiazkaTrainer(node2emb, rel2idx, edges, cfg, shadow_vectors=shadow)
    history = trainer.train(epochs=cfg.epochs, log_every=args.log_every_n_steps)

    metrics = trainer.evaluate(n_probe=min(256, max(32, len(trainer.edges))))
    print("\n=== final evaluation ===")
    print(_fmt_metrics([metrics]))

    ckpt = trainer.save(args.save_dir, metrics=metrics)

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
        "final_metrics": metrics,
        "commutation_C": {
            f"{names[i]}|{names[j]}": float(commutation[i, j])
            for i in range(len(names))
            for j in range(len(names))
        },
        "checkpoint": ckpt,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[wiazka] checkpoint: {ckpt}")
    print(f"[wiazka] report:     {report_path}")


if __name__ == "__main__":
    main()
