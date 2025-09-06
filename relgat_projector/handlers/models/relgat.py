import json
import torch
import pickle
import argparse

from relgat_projector.base.constants import ConstantsRelGATTrainer
from relgat_projector.trainer.relgat_projector import RelGATTrainer


class RelGATMainTrainerHandler:
    @staticmethod
    def load_embeddings_and_edges(
        path_to_nodes: str, path_to_rels: str, path_to_edges: str
    ):
        # node2emb – dict[int, torch.Tensor]
        print("Loading", path_to_nodes)
        with open(path_to_nodes, "rb") as f:
            _node2emb = pickle.load(f)
        _node2emb = {int(k): torch.tensor(v) for k, v in _node2emb.items()}
        print(f"  - number of loaded nodes: {len(_node2emb)}")

        # rel2idx – dict[str, int]
        print("Loading", path_to_rels)
        with open(path_to_rels, "r") as f:
            _rel2idx = json.loads(f.read())
        _rel2idx = {str(k): int(v) for k, v in _rel2idx.items()}
        print(f"  - number of loaded rel2idx: {len(_rel2idx)}")

        # edge_index_raw – list[(src, dst, rel_str)]
        print("Loading", path_to_edges)
        with open(path_to_edges, "r") as f:
            _edge_index_raw = json.loads(f.read())
        print(f"  - number of loaded edges: {len(_edge_index_raw)}")
        _edge_index_raw = [
            [int(f), int(t), str(r)]
            for f, t, r in _edge_index_raw
            if int(f) in _node2emb and int(t) in _node2emb
        ]
        print(f"  - number of edges after filtering: {len(_edge_index_raw)}")

        return _node2emb, _rel2idx, _edge_index_raw

    @staticmethod
    def build_trainer(
        node2emb,
        rel2idx,
        edge_index_raw,
        args: argparse.Namespace,
    ) -> RelGATTrainer:
        run_cfg = {
            # Reproduction
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "device": args.device,
            "run_name": args.run_name,
            "train_batch_size": args.batch_size,
            "eval_batch_size": args.batch_size,
            "epochs": args.epochs,
            "warmup_steps": args.warmup_steps,
            "margin": args.margin,
            "early_stop_patience": args.early_stop_patience,
            # Architecture specification
            "scorer_type": args.scorer,
            "gat_out_dim": args.gat_out_dim,
            "gat_heads": args.heads,
            "gat_num_layers": args.gat_num_layers,
            "dropout": args.dropout,
            "dropout_rel_attention": args.dropout_rel_attention,
            "architecture_name": args.architecture,
            "base_model_name": "relgat",
            "project_to_input_size": args.project_to_input_size,
            # Larning rate management
            "lr": args.lr,
            "lr_scheduler": args.lr_scheduler,
            "lr_decay": args.lr_decay,
            # Storage management
            "out_dir": args.save_dir,
            "max_checkpoints": args.max_checkpoints,
            "num_neg": args.num_neg,
            # Logging
            "log_every_n_steps": args.log_every_n_steps,
            # Training n-steps
            "save_every_n_steps": args.save_every_n_steps,
            "eval_every_n_steps": args.eval_every_n_steps,
            # Additional training args
            # "use_amp": args.use_amp,
            "disable_edge_type_mask": args.disable_edge_type_mask,
            "use_self_adv_neg": args.use_self_adv_neg,
            "self_adv_alpha": args.self_adv_alpha,
            "weight_decay": args.weight_decay,
            "grad_clip_norm": args.grad_clip_norm,
            "eval_ks_ranks": [i for i in range(1, args.num_neg + 1)],
            "relgat_weight": args.relgat_weight,
            "cosine_weight": args.cosine_weight,
            "mse_weight": args.mse_weight,
            "projection_to_base_size": args.projection_to_base_size,
        }

        trainer = RelGATTrainer(
            node2emb=node2emb,
            rel2idx=rel2idx,
            edge_index_raw=edge_index_raw,
            train_batch_size=run_cfg["train_batch_size"],
            eval_batch_size=run_cfg["eval_batch_size"],
            num_neg=run_cfg["num_neg"],
            early_stop_patience=run_cfg["early_stop_patience"],
            # Whole config
            run_config=run_cfg,
            run_name=run_cfg["run_name"],
            wandb_config=ConstantsRelGATTrainer.WandbConfig,
            device=run_cfg["device"],
            # Reproduction
            seed=run_cfg["seed"],
            train_ratio=run_cfg["train_ratio"],
            # Larning rate management
            lr=run_cfg["lr"],
            lr_decay=run_cfg["lr_decay"],
            lr_scheduler=run_cfg["lr_scheduler"],
            # Architecture specification
            scorer_type=run_cfg["scorer_type"],
            gat_out_dim=run_cfg["gat_out_dim"],
            gat_heads=run_cfg["gat_heads"],
            gat_num_layers=run_cfg["gat_num_layers"],
            dropout=run_cfg["dropout"],
            rel_attn_dropout=run_cfg["dropout_rel_attention"],
            architecture_name=run_cfg["architecture_name"],
            project_to_input_size=run_cfg["project_to_input_size"],
            # Storage
            out_dir=run_cfg["out_dir"],
            max_checkpoints=run_cfg["max_checkpoints"],
            # train_ratio=run_cfg["train_ratio"],
            # N-step
            save_every_n_steps=run_cfg["save_every_n_steps"],
            eval_every_n_steps=run_cfg["eval_every_n_steps"],
            # Logging
            log_grad_norm=True,
            log_every_n_steps=run_cfg["log_every_n_steps"],
            # Additional params
            # use_amp=run_cfg["use_amp"],
            weight_decay=run_cfg["weight_decay"],
            grad_clip_norm=run_cfg["grad_clip_norm"],
            disable_edge_type_mask=run_cfg["disable_edge_type_mask"],
            use_self_adv_neg=run_cfg["use_self_adv_neg"],
            self_adv_alpha=run_cfg["self_adv_alpha"],
            # Evaluation
            eval_ks_ranks=run_cfg["eval_ks_ranks"],
            relgat_weight=run_cfg["relgat_weight"],
            cosine_weight=run_cfg["cosine_weight"],
            mse_weight=run_cfg["mse_weight"],
            projection_to_base_size=run_cfg["projection_to_base_size"],
        )
        return trainer
