import abc
import json
import time
import shutil
import torch
import random

from tqdm import tqdm
from pathlib import Path
from collections import deque
from typing import Dict, List, Tuple, Optional, Any

# RadLab ML utils dependency
from rdl_ml_utils.handlers.wandb import WanDBHandler

from relgat_llm.base.model.model import RelGATModel
from relgat_llm.base.constants import ConstantsRelGATTrainer

from relgat_llm.trainer.core.any_lr_trainer import AnyLRTrainerI
from relgat_llm.trainer.core.any_repr_trainer import AnyReproductiveTrainerI
from relgat_llm.trainer.core.relgat_dataset import AnyRelGATModelDatasetI
from relgat_llm.trainer.core.any_storage_trainer import RelGATTrainerBaseStorageI
from relgat_llm.trainer.core.any_architecture import AnyModelArchitectureConstructorI


class RelGATTrainer(
    AnyReproductiveTrainerI,
    AnyLRTrainerI,
    AnyModelArchitectureConstructorI,
    AnyRelGATModelDatasetI,
    RelGATTrainerBaseStorageI,
):
    def __init__(
        self,
        # Whole config -- with prior higher than args
        run_config: Dict,
        wandb_config: Optional[Any],
        run_name: Optional[str],
        # Dataset
        node2emb: Dict[int, torch.Tensor],
        rel2idx: Dict[str, int],
        edge_index_raw: List[Tuple[int, int, str]],
        train_batch_size: int = 256,
        eval_batch_size: int = 256,
        # Learning environment
        device: str = "cpu",
        # Reproduction
        seed: int = 42,
        train_ratio: float = 0.90,
        # LR management
        lr: float = 0.0001,
        lr_decay: float = 1.0,
        lr_scheduler: str = "linear",
        # Architecture spec
        scorer_type: str = "distmult",
        gat_out_dim: int = 200,
        gat_heads: int = 6,
        gat_num_layers: int = 1,
        dropout: float = 0.2,
        rel_attn_dropout: float = 0.0,
        architecture_name: Optional[str] = None,
        base_model_name: Optional[str] = None,
        # Storage
        max_checkpoints: int = 5,
        out_dir: Optional[str] = None,
        num_neg: int = 4,

        # weight_decay: float = 0.0,
        # grad_clip_norm: Optional[float] = None,
        # early_stop_patience: Optional[int] = None,
        # use_amp: bool = False,
        # log_grad_norm: bool = False,
        # disable_edge_type_mask: bool = False,
        # self_adv_alpha: float = 1.0,
        # profile_steps: int = 0,
        # log_every_n_steps: int = 100,
        # save_every_n_steps: Optional[int] = None,
        # eval_every_n_steps: Optional[int] = None,
        # use_self_adv_neg: bool = False,
    ):
        AnyReproductiveTrainerI.__init__(self, seed=seed, run_config=run_config)
        AnyModelArchitectureConstructorI.__init__(
            self,
            gat_out_dim=gat_out_dim,
            gat_heads=gat_heads,
            gat_num_layers=gat_num_layers,
            dropout=dropout,
            dropout_rel_attention=rel_attn_dropout,
            scorer_type=scorer_type,
            architecture_name=architecture_name,
            base_model_name=base_model_name,
            run_config=run_config,
        )
        AnyLRTrainerI.__init__(
            self,
            lr=lr,
            lr_scheduler=lr_scheduler,
            lr_decay=lr_decay,
            run_config=run_config,
        )
        AnyRelGATModelDatasetI.__init__(
            self,
            device=device,
            node2emb=node2emb,
            rel2idx=rel2idx,
            edge_index_raw=edge_index_raw,
            train_ratio=train_ratio,
            num_neg=num_neg,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            run_config=run_config,
        )

        RelGATTrainerBaseStorageI.__init__(
            self,
            out_dir=out_dir,
            max_checkpoints=max_checkpoints,
            run_config=run_config,
        )

        # Training environment
        self.run_config = run_config
        self.wandb_config = wandb_config

        self.device = str(run_config.get("device", device))
        self.run_name = str(run_config["run_name"], run_name)

        ####################################################################################################
        self._no_improve_steps = 0
        self.log_grad_norm = log_grad_norm
        self.weight_decay = weight_decay
        self.early_stop_patience = early_stop_patience

        self.use_amp = use_amp
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        self.grad_clip_norm = grad_clip_norm
        self.profile_steps = profile_steps
        self.early_stop_patience = early_stop_patience
        self.disable_edge_type_mask = disable_edge_type_mask
        self.use_self_adv_neg = bool(use_self_adv_neg)
        self.self_adv_alpha = float(self_adv_alpha)

        # Model saving – list of best checkpoints
        self.best_mrr = -float("inf")

        # Logging
        self.global_step = 0
        self.log_every_n_steps = max(1, int(log_every_n_steps))

        # Model saving
        self.save_every_n_steps = (
            int(save_every_n_steps)
            if save_every_n_steps is not None and int(save_every_n_steps) > 0
            else None
        )

        self.eval_every_n_steps = (
            int(eval_every_n_steps)
            if eval_every_n_steps is not None and int(eval_every_n_steps) > 0
            else None
        )
        ####################################################################################################

        # Model
        self.model = RelGATModel(
            node_emb=self.node_emb_matrix,
            edge_index=self.edge_index,
            edge_type=self.edge_type,
            num_rel=self.num_rel,
            scorer_type=self.scorer_type,
            gat_out_dim=self.gat_out_dim,
            gat_heads=self.gat_heads,
            dropout=self.dropout,
            relation_attn_dropout=self.dropout_rel_attention,
            gat_num_layers=self.gat_num_layers,
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.base_lr,
            weight_decay=self.weight_decay,
        )

        # W&B initialization
        if self.wandb_config is not None:
            WanDBHandler.init_wandb(
                wandb_config=self.wandb_config,
                run_config=self.run_config,
                training_args=None,
                run_name=self.run_name,
            )

    def evaluate(self, ks: Tuple[int, ...] = (1, 3, 10)):
        self.model.eval()
        total_mrr = 0.0
        total_hits = {k: 0.0 for k in ks}
        n_examples = 0

        total_loss = 0.0
        total_pos = 0

        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc="Evaluation", leave=False):
                if not batch:
                    continue

                pos, *negs = zip(*batch)

                src_ids = torch.cat(
                    [p[0] for p in pos] + [n[0] for n in sum(negs, ())], dim=0
                ).to(self.device)
                rel_ids = torch.cat(
                    [p[1] for p in pos] + [n[1] for n in sum(negs, ())], dim=0
                ).to(self.device)
                dst_ids = torch.cat(
                    [p[2] for p in pos] + [n[2] for n in sum(negs, ())], dim=0
                ).to(self.device)

                scores = self.model(src_ids, rel_ids, dst_ids)  # [B*(1+num_neg)]

                # Zabezpieczenie: policz i napraw niefinity
                nonfinite = (~torch.isfinite(scores)).sum().item()
                if nonfinite > 0:
                    WanDBHandler.log_metrics(
                        {"eval/nonfinite_scores": nonfinite}, step=self.global_step
                    )
                    scores = torch.nan_to_num(
                        scores, nan=0.0, neginf=-1e9, posinf=1e9
                    )

                B = len(pos)
                pos_score = scores[:B]
                neg_score = scores[B:].view(B, self.num_neg)

                # Loss policz na klamrowanych wartościach
                pos_s = pos_score.clamp(-20.0, 20.0)
                neg_s = neg_score.clamp(-20.0, 20.0)

                if self.use_self_adv_neg:
                    batch_loss = self.self_adversarial_loss(
                        pos_s, neg_s, alpha=self.self_adv_alpha
                    )
                else:
                    batch_loss = self.margin_ranking_loss(pos_s, neg_s, margin=1.0)

                total_loss += batch_loss.item() * B
                total_pos += B

                for i in range(B):
                    cand_scores = torch.cat(
                        [pos_score[i].unsqueeze(0), neg_score[i]], dim=0
                    )
                    # compute_mrr_hits już sanetyzuje
                    mrr, hits = self.compute_mrr_hits(cand_scores, true_idx=0, ks=ks)
                    total_mrr += mrr
                    for k in ks:
                        total_hits[k] += hits[k]
                    n_examples += 1

        n_examples = max(1, n_examples)
        avg_mrr = total_mrr / n_examples
        avg_hits = {k: total_hits[k] / n_examples for k in ks}

        # average eval loss on positive example
        avg_eval_loss = total_loss / max(1, total_pos)

        return avg_mrr, avg_hits, avg_eval_loss

    # The main training loop
    def train(self, epochs: int = 12, margin: float = 1.0):
        # ---- Scheduler (warmup + selected decay) ----
        import math
        import math as _math  # for cosine

        steps_per_epoch = max(
            1, math.ceil(len(self.train_dataset) / self.train_batch_size)
        )
        total_steps = steps_per_epoch * max(1, int(epochs))
        warmup_steps_cfg = self.run_config.get("warmup_steps", None)
        if warmup_steps_cfg is None:
            warmup_steps = int(self.default_warmup_ratio * total_steps)
        warmup_steps = min(warmup_steps, max(0, total_steps - 1))

        def _lr_lambda_linear(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0,
                float(total_steps - current_step)
                / float(max(1, total_steps - warmup_steps)),
            )

        def _lr_lambda_cosine(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return 0.5 * (1.0 + _math.cos(_math.pi * min(1.0, max(0.0, progress))))

        def _lr_lambda_constant(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0

        if self.scheduler_type == "linear":
            lr_lambda = _lr_lambda_linear
        elif self.scheduler_type == "cosine":
            lr_lambda = _lr_lambda_cosine
        elif self.scheduler_type == "constant":
            lr_lambda = _lr_lambda_constant
        else:
            raise ValueError(f"Unknown lr_scheduler type: {self.scheduler_type}")

        if self.lr_decay != 1.0:
            # after the warm-up phase, each step is additionally
            # multiplied by the decay-factor.
            base_lambda = lr_lambda

            def lr_lambda(step: int):
                return base_lambda(step) * (
                    self.lr_decay ** max(0, step - warmup_steps)
                )

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lr_lambda
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lr_lambda
            )

        # Log scheduler info (once)
        WanDBHandler.log_metrics(
            metrics={
                "scheduler/total_steps": total_steps,
                "scheduler/warmup_steps": warmup_steps,
                "scheduler/type": self.scheduler_type,
                "config/use_self_adv_neg": float(self.use_self_adv_neg),
                "config/self_adv_alpha": float(self.self_adv_alpha),
                "train/base_lr": self.base_lr,
            },
            step=self.global_step,
        )

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            running_loss = 0.0
            running_examples = 0
            # ----------------- TRAIN -----------------
            for step_in_epoch, batch in enumerate(
                tqdm(
                    self.train_loader,
                    desc=f"Epoch {epoch:02d}/{epochs:02d} – training",
                    leave=False,
                ),
                start=1,
            ):
                step_start = time.time()
                pos, *negs = zip(*batch)

                src_ids = torch.cat(
                    [p[0] for p in pos] + [n[0] for n in sum(negs, ())], dim=0
                ).to(self.device)
                rel_ids = torch.cat(
                    [p[1] for p in pos] + [n[1] for n in sum(negs, ())], dim=0
                ).to(self.device)
                dst_ids = torch.cat(
                    [p[2] for p in pos] + [n[2] for n in sum(negs, ())], dim=0
                ).to(self.device)

                # use auto casting
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    self.optimizer.zero_grad(set_to_none=True)
                    scores = self.model(src_ids, rel_ids, dst_ids)

                    # Zabezpieczenie: policz i napraw niefinity
                    nonfinite = (~torch.isfinite(scores)).sum().item()
                    if nonfinite > 0:
                        WanDBHandler.log_metrics(
                            {"train/nonfinite_scores": nonfinite},
                            step=self.global_step,
                        )
                        scores = torch.nan_to_num(
                            scores, nan=0.0, neginf=-1e9, posinf=1e9
                        )

                    B = len(pos)
                    pos_score = scores[:B].clamp(-20.0, 20.0)
                    neg_score = scores[B:].view(B, self.num_neg).clamp(-20.0, 20.0)

                    if self.use_self_adv_neg:
                        loss = self.self_adversarial_loss(
                            pos_score, neg_score, alpha=self.self_adv_alpha
                        )
                    else:
                        loss = self.margin_ranking_loss(
                            pos_score, neg_score, margin=margin
                        )

                # Jeśli loss nie jest skończony – przerwij ten krok i zaloguj
                if not torch.isfinite(loss):
                    WanDBHandler.log_metrics(
                        {"train/nonfinite_loss_steps": 1}, step=self.global_step
                    )
                    print("Non‑finite loss encountered. Skipping step.")
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                # scaler backward if enabled
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # gradient clipping (if enabled)
                if self.grad_clip_norm is not None:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.grad_clip_norm,
                    )

                # optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                # step scheduler after optimizer step
                if self.scheduler is not None:
                    self.scheduler.step()

                # loss accumulation
                loss_item = loss.item()
                epoch_loss += loss_item * B
                running_loss += loss_item * B
                running_examples += B

                # global step and logging at every N steps
                self.global_step += 1

                # logging every n steps
                if self.global_step % self.log_every_n_steps == 0:
                    step_end = time.time()
                    step_time = step_end - step_start

                    grad_norm = -float("inf")
                    if self.log_grad_norm:
                        total_norm = 0.0
                        for p in self.model.parameters():
                            if p.grad is not None:
                                param_norm = p.grad.detach().data.norm(2)
                                total_norm += param_norm.item() ** 2
                        grad_norm = total_norm**0.5

                    avg_running_loss = running_loss / max(1, running_examples)
                    current_lr = (
                        self.scheduler.get_last_lr()[0]
                        if self.scheduler is not None
                        else self.optimizer.param_groups[0]["lr"]
                    )

                    print(
                        f"\nGlobal step {self.global_step} "
                        f"grad_norm {grad_norm:.8f} "
                        f"loss_step: {avg_running_loss:.8f} "
                        f"lr: {current_lr:.8f} "
                        f"step_time {step_time}"
                    )

                    WanDBHandler.log_metrics(
                        metrics={
                            "epoch": epoch,
                            "train/loss_step": avg_running_loss,
                            "train/step_in_epoch": step_in_epoch,
                            "train/grad_norm": grad_norm,
                            "train/lr": current_lr,
                            "train/step_time": step_time,
                            "train/pos_score_mean": (
                                pos_score.detach().mean().item() if B > 0 else 0.0
                            ),
                            "train/neg_score_mean": (
                                neg_score.detach().mean().item()
                                if neg_score.numel() > 0
                                else 0.0
                            ),
                        },
                        step=self.global_step,
                    )
                    # reset counters
                    running_loss = 0.0
                    running_examples = 0

                # Eval model every n steps
                if (
                    self.eval_every_n_steps is not None
                    and self.global_step % self.eval_every_n_steps == 0
                ):
                    avg_train_loss = epoch_loss / len(self.train_dataset)
                    # ----------------- EVAL -----------------
                    mrr, hits, eval_loss = self.evaluate(ks=(1, 2, 3))
                    hits_str = ", ".join(
                        [f"Hits@{k}: {hits[k]:.4f}" for k in sorted(hits)]
                    )
                    print(
                        f"\n=== Step {self.global_step:12d} – "
                        f"train loss: {avg_train_loss:.4f}"
                    )
                    print(
                        f"   - eval – loss: {eval_loss:.4f} |"
                        f" MRR: {mrr:.4f} | {hits_str}"
                    )
                    # ----------------- LOG TO W&B -----------------
                    WanDBHandler.log_metrics(
                        metrics={
                            "epoch": epoch,
                            "eval/loss": eval_loss,
                            "eval/mrr": mrr,
                            "eval/hits@1": hits[1],
                            "eval/hits@2": hits[2],
                            "eval/hits@3": hits[3],
                        },
                        step=self.global_step,
                    )

                    # Save
                    current_mrr = mrr
                    if current_mrr > self.best_mrr:
                        self.best_mrr = current_mrr
                        if (
                            self.save_every_n_steps is not None
                            and self.global_step % self.save_every_n_steps == 0
                        ):
                            self.best_ckpt_dir = (
                                f"best_checkpoint_{self.global_step}"
                            )
                            self._save_model_and_files(subdir=self.best_ckpt_dir)
                            self._prune_checkpoints()

                            # log saved checkpoint to w&b
                            WanDBHandler.log_metrics(
                                metrics={"checkpoint/step": self.global_step},
                                step=self.global_step,
                            )
                        self._no_improve_steps = 0
                    else:
                        self._no_improve_steps += 1

                    if (
                        self.early_stop_patience is not None
                        and self._no_improve_steps >= self.early_stop_patience
                    ):
                        print(
                            "\n  Early‑stopping triggered – no improvement for "
                            f"{self.early_stop_patience} evaluation steps."
                        )
                        break

            if self.eval_every_n_steps is None:
                avg_train_loss = epoch_loss / len(self.train_dataset)
                # ----------------- EVAL -----------------
                mrr, hits, eval_loss = self.evaluate(ks=(1, 2, 3))
                hits_str = ", ".join(
                    [f"Hits@{k}: {hits[k]:.4f}" for k in sorted(hits)]
                )
                print(f"\n=== Epoch {epoch:02d} – train loss: {avg_train_loss:.4f}")
                print(
                    f"   - eval – loss: {eval_loss:.4f} | "
                    f"MRR: {mrr:.4f} | {hits_str}"
                )
                # ----------------- LOG TO W&B -----------------
                WanDBHandler.log_metrics(
                    metrics={
                        "epoch": epoch,
                        "train/loss": avg_train_loss,
                        "eval/loss": eval_loss,
                        "eval/mrr": mrr,
                        "eval/hits@1": hits[1],
                        "eval/hits@2": hits[2],
                        "eval/hits@3": hits[3],
                    },
                    step=self.global_step,
                )

        # ----------------- SAVE FINAL MODEL  -----------------
        out_model_dir = (
            f"relgat_{self.scorer_type}"
            f"_ratio{int(self.run_config['train_ratio'] * 100)}"
        )
        self._save_model_and_files(subdir=out_model_dir)
        print(f"\nTraining finished – model saved to: {out_model_dir}")

        # ----------------- ARTIFACT W&B -----------------
        # WanDBHandler.add_model(name=f"relgat-{self.scorer_type}", local_path="..")
        WanDBHandler.finish_wand()

    def _save_checkpoint(self, subdir: str) -> str:
        """
        Saves model state_dict into self.save_dir/subdir/OUT_MODEL_NAME
        Also save:
            - run config
            - relations mapping

        Returns saved file (model) path.
        """
        self._save_model_and_files(
            subdir=subdir,
            model=self.model,
            files=[
                (
                    ConstantsRelGATTrainer.Default.TRAINING_CONFIG_FILE_NAME,
                    self.run_config,
                ),
                (
                    ConstantsRelGATTrainer.Default.TRAINING_CONFIG_REL_TO_IDX,
                    self.rel2idx,
                ),
            ],
        )

    #     out_dir = self.save_dir / subdir
    #     out_dir.mkdir(parents=True, exist_ok=True)
    #     out_path = out_dir / ConstantsRelGATTrainer.Default.OUT_MODEL_NAME
    #     torch.save(self.model.state_dict(), out_path)
    #
    #     _files = [
    #         (
    #             ConstantsRelGATTrainer.Default.TRAINING_CONFIG_FILE_NAME,
    #             self.run_config,
    #         ),
    #         (
    #             ConstantsRelGATTrainer.Default.TRAINING_CONFIG_REL_TO_IDX,
    #             self.rel2idx,
    #         ),
    #     ]
    #     for f_name, json_data in _files:
    #         out_cfg_path = out_dir / f_name
    #         with open(out_cfg_path, "w") as f:
    #             f.write(json.dumps(json_data, indent=2, ensure_ascii=False))
    #     return str(out_path)
    #
    # def _prune_checkpoints(self) -> None:
    #     """
    #     Keeps a maximum of max_checkpoints most recent (or best) checkpoints.
    #     The oldest ones are deleted from the disc.
    #     """
    #     while len(self.saved_checkpoints) > self.max_checkpoints:
    #         oldest = self.saved_checkpoints.popleft()
    #         try:
    #             shutil.rmtree(oldest)
    #             print(f"️ Removed old checkpoint: {oldest}")
    #         except Exception as exc:
    #             print(f" Could not delete {oldest}: {exc}")
