import time
import torch

from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any

from relgat_llm.utils.random_seed import RandomSeed
from relgat_llm.utils.logging_adapter import LoggerAdapter

from relgat_llm.base.constants import ConstantsRelGATTrainer
from relgat_llm.dataset.relgat_dataset import RelGATDataset
from relgat_llm.handlers.storage import RelGATStorage

from relgat_llm.core.eval import RelgatEval
from relgat_llm.core.loss import RelGATLoss
from relgat_llm.core.lr import TrainingScheduler
from relgat_llm.core.model.relgat_base.model import RelGATModel
from relgat_llm.core.architecture.constructor import ModelArchitectureConstructor

from relgat_llm.trainer.components.grad import compute_total_grad_norm
from relgat_llm.trainer.components.relgat_batching import concat_pos_negs_to_tensors


class RelGATTrainer:
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
        margin: float = 1.0,
        warmup_steps: Optional[int] = None,
        train_ratio: float = 0.90,
        early_stop_patience: Optional[int] = None,
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
        # N-step
        save_every_n_steps: Optional[int] = None,
        eval_every_n_steps: Optional[int] = None,
        # Logging
        log_grad_norm: bool = True,
        log_every_n_steps: int = 100,
        # Additional params
        weight_decay: float = 0.0,
        grad_clip_norm: Optional[float] = None,
        disable_edge_type_mask: bool = False,
        use_self_adv_neg: bool = False,
        self_adv_alpha: float = 1.0,
        # evaluation options
        eval_vectorized: bool = True,
        # use_amp: bool = False,
        eval_ks: Tuple[int, ...] = (1, 2, 3, 4, 5),
    ):
        # Experiments reproduction (seed)
        self.repr_training = RandomSeed(
            seed=seed, run_config=run_config, auto_set_seed=True
        )

        # Specify architecture
        self.architecture = ModelArchitectureConstructor(
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

        # Training scheduler (total steps, warmup steps, etc.)
        self.training_scheduler = TrainingScheduler(
            lr=lr,
            lr_scheduler=lr_scheduler,
            lr_decay=lr_decay,
            warmup_steps=warmup_steps,
            run_config=run_config,
        )

        # RelGAT dataset
        self.dataset = RelGATDataset(
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
        # Storage and training handling (checkpoints)
        self.storage = RelGATStorage(
            out_dir=out_dir,
            max_checkpoints=max_checkpoints,
            save_every_n_steps=save_every_n_steps,
            run_config=run_config,
        )

        self.log_adapter = LoggerAdapter(
            run_name=run_name,
            architecture_name=architecture_name,
            run_config=run_config,
            wandb_config=wandb_config,
            log_every_n_steps=log_every_n_steps,
        )

        # Which loss should be used?
        use_self_adv_neg = run_config.get("use_self_adv_neg", use_self_adv_neg)
        if use_self_adv_neg is not None:
            use_self_adv_neg = bool(use_self_adv_neg)
        self.loss = RelGATLoss(
            loss_type=(
                "self_adversarial_loss"
                if use_self_adv_neg
                else "margin_ranking_loss"
            ),
            self_adv_alpha=self_adv_alpha,
            margin=margin,
            clamp_limit=20,
            run_config=run_config,
        )

        # ====================================================================
        # debug_mode
        self.debug_mode = True

        # Training environment
        self.run_config = run_config
        self.device = str(run_config.get("device", device))

        # Optimizer (Adam/AdamW) w-d
        self.weight_decay = float(run_config.get("weight_decay", weight_decay))

        # Gradient clipping
        self.grad_clip_norm = run_config.get("grad_clip_norm", grad_clip_norm)

        # ====================================================================
        # ====================================================================
        # ====================================================================
        # ====================================================================
        # ====================================================================
        self.log_grad_norm = log_grad_norm
        self.early_stop_patience = int(
            run_config.get("early_stop_patience", early_stop_patience)
        )
        self.disable_edge_type_mask = bool(
            run_config.get("disable_edge_type_mask", disable_edge_type_mask)
        )
        # Best mrr
        self.best_mrr = -float("inf")

        # Logging steps
        self.global_step = 0

        # Eval steps
        self.eval_every_n_steps = (
            int(eval_every_n_steps)
            if eval_every_n_steps is not None and int(eval_every_n_steps) > 0
            else None
        )
        # New evaluation config
        self.eval_vectorized = bool(
            run_config.get("eval_vectorized", eval_vectorized)
        )
        self.eval_ks = tuple(run_config.get("eval_ks", list(eval_ks)))

        self._no_improve_steps = 0

        ####################################################################################################
        ####################################################################################################
        ####################################################################################################
        ####################################################################################################
        ####################################################################################################
        # Model definition
        self.model = RelGATModel(
            node_emb=self.dataset.node_emb_matrix,
            edge_index=self.dataset.edge_index,
            edge_type=self.dataset.edge_type,
            num_rel=self.dataset.num_rel,
            scorer_type=self.architecture.scorer_type,
            gat_out_dim=self.architecture.gat_out_dim,
            gat_heads=self.architecture.gat_heads,
            dropout=self.architecture.dropout,
            relation_attn_dropout=self.architecture.dropout_rel_attention,
            gat_num_layers=self.architecture.gat_num_layers,
        ).to(self.device)

        # Optimizer
        # TODO: Change to AdamW (as argument -> optimizer="Adam | AdamW"
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.training_scheduler.base_lr,
            weight_decay=self.weight_decay,
        )

        ####################################################################################################
        # W&B initialization
        self.log_adapter.init_wandb_if_needed()

    def on_epoch_end(self, *, epoch: int) -> None:
        """
        Callback executed at the end of each training epoch.
        Default: no-op. Override to inject custom behavior.
        """
        return

    def evaluate(self, ks: Tuple[int, ...] = (1, 3, 10)):
        self.model.eval()

        ks = tuple(sorted(set(ks)))  # sanitize
        total_mrr = 0.0
        total_hits = {k: 0.0 for k in ks}
        n_examples = 0
        total_loss = 0.0
        total_pos = 0

        with torch.no_grad():
            for batch in tqdm(
                self.dataset.eval_loader, desc="Evaluation", leave=False
            ):
                if not batch:
                    continue
                pos, *negs = zip(*batch)
                src_ids, rel_ids, dst_ids = concat_pos_negs_to_tensors(
                    pos, negs, device=self.device
                )

                pos_examples_in_batch = len(pos)
                scores = self._forward_model_scores(
                    src_ids, rel_ids, dst_ids, phase="eval"
                )
                pos_score, neg_score = self._split_scores(
                    scores, pos_examples_in_batch
                )

                # Loss for reporting
                batch_loss = self.loss.prepare_scores_and_compute_loss(
                    pos_score=pos_score, neg_score=neg_score
                )
                total_loss += batch_loss.item() * pos_examples_in_batch
                total_pos += pos_examples_in_batch

                # Metrics: vectorized or per-sample
                if self.eval_vectorized:
                    mrr_b, hits_b = RelgatEval.compute_batch_metrics_vectorized(
                        pos_score=pos_score, neg_score=neg_score, ks=ks
                    )
                    total_mrr += mrr_b * pos_examples_in_batch
                    for k in ks:
                        total_hits[k] += hits_b[k] * pos_examples_in_batch
                    n_examples += pos_examples_in_batch
                else:
                    for i in range(pos_examples_in_batch):
                        cand_scores = torch.cat(
                            [pos_score[i].unsqueeze(0), neg_score[i]], dim=0
                        )
                        mrr, hits = RelgatEval.compute_mrr_hits(
                            cand_scores, true_idx=0, ks=ks
                        )
                        total_mrr += mrr
                        for k in ks:
                            total_hits[k] += hits[k]
                        n_examples += 1

                # Optional evaluation logging of score statistics
                self.log_adapter.log_metrics(
                    metrics={
                        "eval/pos_score_mean": (
                            pos_score.detach().clamp(-20.0, 20.0).mean().item()
                            if pos_examples_in_batch > 0
                            else 0.0
                        ),
                        "eval/neg_score_mean": (
                            neg_score.detach().clamp(-20.0, 20.0).mean().item()
                            if pos_examples_in_batch > 0
                            else 0.0
                        ),
                    },
                    step=self.global_step,
                )

        n_examples = max(1, n_examples)
        avg_mrr = total_mrr / n_examples
        avg_hits = {k: total_hits[k] / n_examples for k in ks}
        avg_eval_loss = total_loss / max(1, total_pos)

        return avg_mrr, avg_hits, avg_eval_loss

    def train(self, epochs: int):
        # Prepare the number of total steps, warmup steps and learning scheduler
        self.training_scheduler.prepare(
            epochs=epochs,
            train_dataset=self.dataset.train_dataset,
            train_batch_size=self.dataset.train_batch_size,
            optimizer=self.optimizer,
        )

        self._log_metrics_on_begin()

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            running_loss = 0.0
            running_examples = 0

            epoch_loss, running_loss, running_examples = self.single_epoch(
                epoch=epoch,
                epochs=epochs,
                epoch_loss=epoch_loss,
                running_loss=running_loss,
                running_examples=running_examples,
            )

            self.on_epoch_end(epoch=epoch)
            if self._eval_if_needed_and_stop_if_needed(
                epoch=epoch, epoch_loss=epoch_loss
            ):
                break

        # Save last model
        out_model_dir = self._save_checkpoint(subdir=None)

        print(f"\nTraining finished – model saved to: {out_model_dir}")
        self.log_adapter.finish_wand_if_needed()

    def single_epoch(
        self,
        epoch: int,
        epochs: int,
        epoch_loss: float,
        running_loss: float,
        running_examples: int,
    ):
        self.model.train()

        for step_in_epoch, batch in enumerate(
            tqdm(
                self.dataset.train_loader,
                desc=f"Epoch {epoch:02d}/{epochs:02d} – training",
                leave=False,
            ),
            start=1,
        ):
            step_start_time = time.time()

            pos, *negs = zip(*batch)
            pos_examples_in_batch = len(pos)
            src_ids, rel_ids, dst_ids = concat_pos_negs_to_tensors(
                pos, negs, device=self.device
            )

            self.optimizer.zero_grad(set_to_none=True)
            scores = self._forward_model_scores(
                src_ids, rel_ids, dst_ids, phase="train"
            )
            pos_score, neg_score = self._split_scores(scores, pos_examples_in_batch)

            loss = self.loss.prepare_scores_and_compute_loss(
                pos_score=pos_score, neg_score=neg_score
            )
            if not torch.isfinite(loss):
                self.log_adapter.log_metrics(
                    {"train/nonfinite_loss_steps": 1}, step=self.global_step
                )
                print("Non‑finite loss encountered. Skipping step.")
                continue
            loss.backward()

            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.grad_clip_norm,
                )

            self.optimizer.step()
            if self.training_scheduler.scheduler is not None:
                self.training_scheduler.scheduler.step()

            # TransE stabilization of scale:
            #   -> normalization of relation vectors after each step.
            if getattr(self, "scorer_type", "").lower() == "transe":
                with torch.no_grad():
                    w = self.model.scorer.rel_emb.weight
                    self.model.scorer.rel_emb.weight.copy_(
                        torch.nn.functional.normalize(w, p=2, dim=-1)
                    )

            loss_item = loss.item()
            epoch_loss += loss_item * pos_examples_in_batch
            running_loss += loss_item * pos_examples_in_batch
            running_examples += pos_examples_in_batch

            self.global_step += 1

            running_loss, running_examples = self._log_step_if_needed(
                epoch=epoch,
                step_in_epoch=step_in_epoch,
                step_start_time=step_start_time,
                running_loss=running_loss,
                running_examples=running_examples,
                scores=scores,
                pos_examples_in_batch=pos_examples_in_batch,
            )

            if self._eval_step_if_needed_and_end_training(
                epoch=epoch, epoch_loss=epoch_loss
            ):
                break

        return epoch_loss, running_loss, running_examples

    def _forward_model_scores(
        self,
        src_ids: torch.Tensor,
        rel_ids: torch.Tensor,
        dst_ids: torch.Tensor,
        phase: str,
    ) -> torch.Tensor:
        """
        Shared forward pass with non-finite handling and contextual logging.
        phase: 'train' or 'eval' (affects metric names)
        """
        scores = self.model(src_ids, rel_ids, dst_ids)

        nonfinite = (~torch.isfinite(scores)).sum().item()
        if nonfinite > 0:
            self.log_adapter.log_metrics(
                {f"{phase}/nonfinite_scores": nonfinite},
                step=self.global_step,
            )
            scores = torch.nan_to_num(scores, nan=0.0, neginf=-1e9, posinf=1e9)
        return scores

    def _split_scores(
        self, scores: torch.Tensor, pos_examples_in_batch: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split flat scores into pos [pos_examples_in_batch] and neg [num_neg]
        """
        pos_score = scores[:pos_examples_in_batch]
        neg_score = scores[pos_examples_in_batch:].view(
            pos_examples_in_batch, self.dataset.num_neg
        )
        return pos_score, neg_score

    def _print_and_log_eval(
        self,
        *,
        epoch: int,
        avg_train_loss: float,
        mrr: float,
        hits: Dict[int, float],
        eval_loss: float,
        step_based: bool,
    ) -> None:
        hits_str = ", ".join([f"Hits@{k}: {hits[k]:.4f}" for k in sorted(hits)])
        if step_based:
            print(
                f"\n=== Step {self.global_step:12d} – "
                f"train loss: {avg_train_loss:.4f}"
            )
        else:
            print(f"\n=== Epoch {epoch:02d} – train loss: {avg_train_loss:.4f}")

        print(f"   - eval – loss: {eval_loss:.4f} | " f"MRR: {mrr:.4f} | {hits_str}")

        metrics = {
            "epoch": epoch,
            "eval/loss": eval_loss,
            "eval/mrr": mrr,
        }
        for ks in hits.keys():
            metrics[f"eval/hits@{ks}"] = hits.get(ks, 0.0)

        self.log_adapter.log_metrics(metrics=metrics, step=self.global_step)

    def _on_eval_end(self, mrr: float) -> bool:
        """
        Handle best metric tracking, checkpointing, and early stopping counter.
        Returns True if early stopping should be triggered.
        """
        improved = mrr > self.best_mrr
        if improved:
            self.best_mrr = mrr
            if (
                self.storage.save_every_n_steps is not None
                and self.global_step % self.storage.save_every_n_steps == 0
            ):
                self.best_ckpt_dir = f"best_checkpoint_{self.global_step}"
                self._save_checkpoint(subdir=self.best_ckpt_dir)
                self.storage.prune_checkpoints()

                self.log_adapter.log_metrics(
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
            return True
        return False

    def _run_eval_and_maybe_early_stop(
        self,
        *,
        epoch: int,
        avg_train_loss: float,
        step_based: bool,
    ) -> bool:
        """
        Run evaluation, log once, update best/early-stop. Returns True if training should stop.
        """
        # Use default eval_ks unless overridden here
        ks = tuple(sorted(set(self.eval_ks if self.eval_ks else (1, 2, 3))))
        mrr, hits, eval_loss = self.evaluate(ks=ks)
        self._print_and_log_eval(
            epoch=epoch,
            avg_train_loss=avg_train_loss,
            mrr=mrr,
            hits=hits,
            eval_loss=eval_loss,
            step_based=step_based,
        )
        return self._on_eval_end(mrr)

    def _log_step_if_needed(
        self,
        epoch: int,
        step_in_epoch: int,
        step_start_time,
        running_loss: float,
        running_examples: int,
        scores,
        pos_examples_in_batch: int,
    ):
        if self.global_step % self.log_adapter.log_every_n_steps != 0:
            return running_loss, running_examples

        step_end = time.time()
        step_time = step_end - step_start_time

        grad_norm = -float("inf")
        if self.log_grad_norm:
            grad_norm = compute_total_grad_norm(self.model)

        avg_running_loss = running_loss / max(1, running_examples)
        current_lr = (
            self.training_scheduler.scheduler.get_last_lr()[0]
            if self.training_scheduler.scheduler is not None
            else self.optimizer.param_groups[0]["lr"]
        )

        print(
            f"\nGlobal step {self.global_step} "
            f"grad_norm {grad_norm:.8f} "
            f"loss_step: {avg_running_loss:.8f} "
            f"lr: {current_lr:.8f} "
            f"step_time {step_time}"
        )

        metrics = {
            "epoch": epoch,
            "train/loss_step": avg_running_loss,
            "train/step_in_epoch": step_in_epoch,
            "train/grad_norm": grad_norm,
            "train/lr": current_lr,
            "train/step_time": step_time,
            "train/pos_score_mean": (
                (scores[:pos_examples_in_batch].detach().clamp(-20.0, 20.0))
                .mean()
                .item()
                if pos_examples_in_batch > 0
                else 0.0
            ),
            "train/neg_score_mean": (
                (
                    scores[pos_examples_in_batch:]
                    .view(pos_examples_in_batch, self.dataset.num_neg)
                    .detach()
                    .clamp(-20.0, 20.0)
                )
                .mean()
                .item()
                if (scores.numel() - pos_examples_in_batch) > 0
                else 0.0
            ),
        }

        self.log_adapter.log_metrics(metrics=metrics, step=self.global_step)

        return 0.0, 1

    def _eval_step_if_needed_and_end_training(self, epoch: int, epoch_loss: float):
        if (
            self.eval_every_n_steps is None
            or self.global_step % self.eval_every_n_steps != 0
        ):
            return False

        avg_train_loss = epoch_loss / len(self.dataset.train_dataset)
        should_stop = self._run_eval_and_maybe_early_stop(
            epoch=epoch,
            avg_train_loss=avg_train_loss,
            step_based=True,
        )
        if should_stop:
            return True

        # Change model to `train` mode
        self.model.train()
        return False

    def _log_metrics_on_begin(self):
        self.log_adapter.log_metrics(
            metrics={
                "scheduler/total_steps": self.training_scheduler.total_steps,
                "scheduler/warmup_steps": self.training_scheduler.warmup_steps,
                "scheduler/type": self.training_scheduler.scheduler_type,
                "config/use_self_adv_neg": float(self.loss.use_self_adv_neg),
                "config/self_adv_alpha": float(self.loss.self_adv_alpha),
                "train/base_lr": self.training_scheduler.base_lr,
                "config/eval_vectorized": float(self.eval_vectorized),
            },
            step=self.global_step,
        )

    def _save_checkpoint(self, subdir: Optional[str]) -> str:
        if subdir is None:
            subdir = (
                f"relgat_"
                f"scorer-{self.architecture.scorer_type}_"
                f"lrscheduler-{self.training_scheduler.scheduler_type}"
            )

        return self.storage.save_model_and_files(
            subdir=subdir,
            model=self.model,
            files=[
                (
                    ConstantsRelGATTrainer.Default.TRAINING_CONFIG_FILE_NAME,
                    self.run_config,
                ),
                (
                    ConstantsRelGATTrainer.Default.TRAINING_CONFIG_REL_TO_IDX,
                    self.dataset.rel2idx,
                ),
            ],
        )

    def _eval_if_needed_and_stop_if_needed(self, epoch, epoch_loss):
        should_stop = False
        if self.eval_every_n_steps is None:
            avg_train_loss = epoch_loss / len(self.dataset.train_dataset)
            should_stop = self._run_eval_and_maybe_early_stop(
                epoch=epoch,
                avg_train_loss=avg_train_loss,
                step_based=False,
            )
        return should_stop
