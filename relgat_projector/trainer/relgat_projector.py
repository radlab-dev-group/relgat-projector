import time
import torch

from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any

from relgat_projector.utils.random_seed import RandomSeed
from relgat_projector.utils.logging_adapter import LoggerAdapter

from relgat_projector.base.constants import ConstantsRelGATTrainer
from relgat_projector.dataset.relgat_dataset import RelGATDataset
from relgat_projector.handlers.storage import RelGATStorage

from relgat_projector.core.eval import RelgatEval
from relgat_projector.core.loss.relgat_loss import RelGATLoss
from relgat_projector.core.loss.multi_objective_loss import MultiObjectiveRelLoss
from relgat_projector.core.lr import TrainingScheduler
from relgat_projector.core.model.relgat_base.model import RelGATModel
from relgat_projector.core.architecture.constructor import (
    ModelArchitectureConstructor,
)

from relgat_projector.trainer.components.grad import compute_total_grad_norm
from relgat_projector.trainer.components.relgat_batching import (
    concat_pos_negs_to_tensors,
)


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
        project_to_input_size: bool = True,
        projection_layers: int = 1,
        projection_dropout: float = 0.0,
        projection_hidden_dim: int = 0,
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
        # use_amp: bool = False,
        # evaluation options
        eval_ks_ranks: Optional[List] = None,
        relgat_weight: float = 1.0,
        pos_cosine_weight: float = 1.0,
        neg_cosine_weight: float = 1.0,
        mse_weight: float = 0.0,
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
            project_to_input_size=project_to_input_size,
            projection_layers=projection_layers,
            projection_dropout=projection_dropout,
            projection_hidden_dim=projection_hidden_dim,
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
            log_to_wandb=wandb_config is not None,
            log_to_console=True,
        )

        # Which loss should be used?
        use_self_adv_neg = run_config.get("use_self_adv_neg", use_self_adv_neg)
        if use_self_adv_neg is not None:
            use_self_adv_neg = bool(use_self_adv_neg)
        self.relgat_loss = RelGATLoss(
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

        # Multi-objective: relgat + reconstruction
        self.multi_loss = None
        if project_to_input_size:
            self.multi_loss = MultiObjectiveRelLoss(
                relgat_loss=self.relgat_loss,
                relgat_weight=relgat_weight,
                pos_cosine_weight=pos_cosine_weight,
                neg_cosine_weight=neg_cosine_weight,
                mse_weight=mse_weight,
                run_config=run_config,
            )

        # ====================================================================
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
        self.best_metric_value = -float("inf")

        # Logging steps
        self.global_step = 0

        # Eval steps
        self.eval_every_n_steps = (
            int(eval_every_n_steps)
            if eval_every_n_steps is not None and int(eval_every_n_steps) > 0
            else None
        )
        if eval_ks_ranks is None or not len(eval_ks_ranks):
            eval_ks_ranks = (1, 2, 3)
        self.eval_ks_ranks = tuple(
            sorted(set(run_config.get("eval_ks_ranks", eval_ks_ranks)))
        )

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
            project_to_input_size=self.architecture.project_to_input_size,
            projection_layers=self.architecture.projection_layers,
            projection_dropout=self.architecture.projection_dropout,
            projection_hidden_dim=self.architecture.projection_hidden_dim,
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

    def evaluate(self, ks: Tuple[int, ...]):
        self.model.eval()
        total_mrr = 0.0
        total_hits = {k: 0.0 for k in ks}
        n_examples = 0
        total_loss = 0.0
        total_pos = 0
        total_cos_pos = 0.0
        total_cos_neg = 0.0
        total_mse = 0.0

        with torch.no_grad():
            for batch in tqdm(
                self.dataset.eval_loader, desc="Evaluation", leave=False
            ):
                pos, *negs = zip(*batch)
                pos_examples_in_batch = len(pos)
                neg_examples_in_batch = len(negs)
                n_examples += pos_examples_in_batch

                src_ids, rel_ids, dst_ids = concat_pos_negs_to_tensors(
                    pos, negs, device=self.device
                )
                pos_score, neg_score, batch_loss, mse, cosine_pos, cosine_neg = (
                    self._calculate_loss(
                        src_ids=src_ids,
                        rel_ids=rel_ids,
                        dst_ids=dst_ids,
                        pos_examples_in_batch=pos_examples_in_batch,
                        phase="eval",
                    )
                )

                total_loss += batch_loss.item() * pos_examples_in_batch
                total_pos += pos_examples_in_batch

                # Metrics -------------------------------------------
                # TODO: Move to the new method
                mrr_b, hits_b = RelgatEval.compute_mrr_hits(
                    pos_score=pos_score, neg_score=neg_score, ks=ks
                )
                total_mrr += mrr_b * pos_examples_in_batch
                for k in ks:
                    total_hits[k] += hits_b[k] * pos_examples_in_batch

                metrics = {
                    "eval/pos_score_mean": (
                        pos_score.detach().mean().item()
                        if pos_examples_in_batch > 0
                        else 0.0
                    ),
                    "eval/neg_score_mean": (
                        neg_score.detach().mean().item()
                        if pos_examples_in_batch > 0
                        else 0.0
                    ),
                }

                if cosine_pos is not None:
                    total_cos_pos += cosine_pos * pos_examples_in_batch
                    metrics["eval/cosine_mean_batch_pos"] = cosine_pos

                if cosine_neg is not None:
                    total_cos_neg += cosine_neg * pos_examples_in_batch
                    metrics["eval/cosine_mean_batch_neg"] = cosine_neg

                if mse is not None:
                    total_mse += mse * pos_examples_in_batch
                    metrics["eval/mse_mean_batch"] = mse

                self.log_adapter.log_metrics(
                    metrics=metrics,
                    step=self.global_step,
                )

        n_examples = max(1, n_examples)
        avg_mrr = total_mrr / n_examples
        avg_hits = {k: total_hits[k] / n_examples for k in ks}
        avg_eval_loss = total_loss / max(1, total_pos)
        avg_cosine_pos = None
        if total_cos_pos > 0.0:
            avg_cosine_pos = total_cos_pos / max(1, total_pos)

        avg_cosine_neg = None
        if total_cos_neg > 0.0:
            avg_cosine_neg = total_cos_neg / max(1, total_pos)

        avg_mse = None
        if total_mse > 0.0:
            avg_mse = total_mse / max(1, total_pos)

        return (
            avg_mrr,
            avg_hits,
            avg_eval_loss,
            avg_cosine_pos,
            avg_cosine_neg,
            avg_mse,
        )

    def train(self, epochs: int):
        # Prepare the number of total steps, warmup steps and learning scheduler
        self.training_scheduler.prepare(
            epochs=epochs,
            train_dataset=self.dataset.train_dataset,
            train_batch_size=self.dataset.train_batch_size,
            optimizer=self.optimizer,
        )

        self._log_begin_information()

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

            pos_score, neg_score, loss, mse, cosine_pos, cosine_neg = (
                self._calculate_loss(
                    src_ids=src_ids,
                    rel_ids=rel_ids,
                    dst_ids=dst_ids,
                    pos_examples_in_batch=pos_examples_in_batch,
                    phase="train",
                )
            )
            if self._log_non_finite_loss_if_needed(loss=loss):
                continue

            self.optimizer.zero_grad(set_to_none=True)
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
            # TODO: Normalization have to be done within scorer
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
                pos_score=pos_score,
                neg_score=neg_score,
                pos_examples_in_batch=pos_examples_in_batch,
                cosine_pos=cosine_pos,
                cosine_neg=cosine_neg,
                mse=mse,
            )

            if self._eval_step_if_needed_and_end_training(
                epoch=epoch, epoch_loss=epoch_loss
            ):
                break

        return epoch_loss, running_loss, running_examples

    def _calculate_loss(
        self, src_ids, rel_ids, dst_ids, pos_examples_in_batch, phase: str
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        cosine_pos, cosine_neg, mse = None, None, None
        if not self.architecture.project_to_input_size:
            scores, transformed, dst_vec = self._forward_model_scores(
                src_ids,
                rel_ids,
                dst_ids,
                phase=phase,
                transform_to_input_if_possible=False,
            )
            pos_score, neg_score = self._split_scores(scores, pos_examples_in_batch)
            loss = self.relgat_loss.prepare_scores_and_compute_loss(
                pos_score=pos_score, neg_score=neg_score
            )
        else:
            (
                pos_score,
                neg_score,
                transformed_src,
                pos_dst_vec,
                neg_dst_vec_transformed,
            ) = self._forward_scores_model_scores_transform(
                src_ids=src_ids,
                rel_ids=rel_ids,
                dst_ids=dst_ids,
                pos_examples_in_batch=pos_examples_in_batch,
            )
            loss = self.multi_loss(
                pos_score=pos_score,
                neg_score=neg_score,
                transformed_src=transformed_src,
                dst_vec=pos_dst_vec,
                neg_dst_vec_transformed=neg_dst_vec_transformed,
            )
            # Reconstruction for logging
            cosine_pos = RelgatEval.batch_cosine_similarity(
                transformed_src.detach(), pos_dst_vec.detach()
            )
            cosine_neg = 1.0 - RelgatEval.batch_cosine_similarity(
                transformed_src.detach(), neg_dst_vec_transformed.detach()
            )
            mse = RelgatEval.batch_mse(
                transformed_src.detach(), pos_dst_vec.detach()
            )
        return pos_score, neg_score, loss, mse, cosine_pos, cosine_neg

    def _forward_model_scores(
        self,
        src_ids: torch.Tensor,
        rel_ids: torch.Tensor,
        dst_ids: torch.Tensor,
        phase: str,
        transform_to_input_if_possible: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Shared forward pass with non-finite handling and contextual logging.
        phase: 'train' or 'eval' (affects metric names)
        """
        scores, transformed, dst_vec = self.model(
            src_ids,
            rel_ids,
            dst_ids,
            transform_to_input_if_possible=transform_to_input_if_possible,
        )

        nonfinite = (~torch.isfinite(scores)).sum().item()
        if nonfinite > 0:
            self.log_adapter.log_metrics(
                {f"{phase}/nonfinite_scores": nonfinite},
                step=self.global_step,
            )
            scores = torch.nan_to_num(scores, nan=0.0, neginf=-1e9, posinf=1e9)
        return scores, transformed, dst_vec

    def _forward_scores_model_scores_transform(
        self,
        *,
        src_ids: torch.Tensor,
        rel_ids: torch.Tensor,
        dst_ids: torch.Tensor,
        pos_examples_in_batch: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        """
        Single GAT-step-based calculation:
        - x = model.compute_node_repr()  (1x GAT + projekcja)
        - pozytywy: pos_score, transformed_src=f_r(A), pos_dst_vec=B
        - negatywy: neg_score [B,K]
        Zwraca: (pos_score, neg_score, transformed_src, pos_dst_vec)
        """
        x = self.model.single_gat_step()  # [N, D_sc]

        # Positives
        pos_rel_ids = rel_ids[:pos_examples_in_batch]
        pos_src_vec = x[src_ids[:pos_examples_in_batch]]  # [B, D]
        pos_dst_vec = x[dst_ids[:pos_examples_in_batch]]  # [B, D]
        pos_score = self.model.scorer(pos_src_vec, pos_rel_ids, pos_dst_vec)  # [B]
        transformed_src = self.model.scorer.transform(
            pos_src_vec, pos_rel_ids
        )  # [B, D]

        # Negatives
        neg_dst_vec_ret = None
        if self.dataset.num_neg > 0:
            neg_src_vec = x[src_ids[pos_examples_in_batch:]]  # [B*K, D]
            neg_dst_vec = x[dst_ids[pos_examples_in_batch:]]  # [B*K, D]
            neg_rel = rel_ids[pos_examples_in_batch:]  # [B*K]
            neg_scores_flat = self.model.scorer(
                neg_src_vec, neg_rel, neg_dst_vec
            )  # [B*K]
            neg_score = neg_scores_flat.view(
                pos_examples_in_batch, self.dataset.num_neg
            )  # [B, K]
            # neg_dst_vec_transformed_flat = self.model.scorer.transform(
            #     neg_dst_vec, neg_rel
            # )  # [B*K, D]
            neg_dst_vec_ret = (
                neg_dst_vec.view(
                    pos_examples_in_batch,
                    self.dataset.num_neg,
                    transformed_src.shape[1],
                )  # [B, K, D]
                .permute(1, 0, 2)  # [K, B, D]
                .contiguous()
            )
        else:
            neg_score = pos_score.new_zeros((pos_examples_in_batch, 0))

        # Sanitization (NaN/Inf)
        pos_score = torch.nan_to_num(pos_score, nan=0.0, neginf=-1e9, posinf=1e9)
        neg_score = torch.nan_to_num(neg_score, nan=0.0, neginf=-1e9, posinf=1e9)
        return (
            pos_score,
            neg_score,
            transformed_src,
            pos_dst_vec,
            neg_dst_vec_ret,
        )

    def _split_scores(
        self, scores: torch.Tensor, pos_examples_in_batch: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split flat scores into pos [pos_examples_in_batch] and neg [num_neg]
        """
        pos_score = scores[:pos_examples_in_batch]
        neg_flat = scores[pos_examples_in_batch:]

        # Negatives in the batch are organized in blocks of "k", meaning:
        #   [neg_k=0 for the entire batch]
        #   [neg_k=1 for the entire batch]
        #   ...
        # Therefore, they need to be transformed into [B, K], not just .view(B, K).
        neg_score = (
            neg_flat.view(self.dataset.num_neg, pos_examples_in_batch)
            .transpose(0, 1)
            .contiguous()
        )
        return pos_score, neg_score

    def _print_and_log_eval(
        self,
        *,
        epoch: int,
        mrr: float,
        hits: Dict[int, float],
        eval_loss: float,
        avg_cosine_pos: Optional[float] = None,
        avg_cosine_neg: Optional[float] = None,
        avg_mse: Optional[float] = None,
    ) -> None:
        metrics = {
            "epoch": epoch,
            "eval/loss": eval_loss,
            "eval/mrr": mrr,
        }
        if avg_cosine_pos is not None:
            metrics["eval/cosine_pos"] = avg_cosine_pos

        if avg_cosine_neg is not None:
            metrics["eval/cosine_neg"] = avg_cosine_neg

        if avg_mse is not None:
            metrics["eval/mse"] = avg_mse

        for ks in hits.keys():
            metrics[f"eval/hits@{ks}"] = hits.get(ks, 0.0)

        self.log_adapter.log_metrics(metrics=metrics, step=self.global_step)

    def _on_eval_end(self, mrr: float, cosine: Optional[float] = None) -> bool:
        """
        Handle the best metric tracking, checkpointing, and early stopping counter.
        Returns True if early stopping should be triggered.
        """
        metric_value = mrr
        if cosine is not None:
            metric_value = cosine

        improved = metric_value > self.best_metric_value
        if improved:
            self.best_metric_value = metric_value
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

    def _run_eval_and_maybe_early_stop(self, *, epoch: int) -> bool:
        """
        Run evaluation, log once, update the best/early-stop.
        Returns True if training should stop.
        """
        avg_mrr, avg_hits, avg_eval_loss, avg_cosine_pos, avg_cosine_neg, avg_mse = (
            self.evaluate(ks=self.eval_ks_ranks)
        )
        self._print_and_log_eval(
            epoch=epoch,
            mrr=avg_mrr,
            hits=avg_hits,
            eval_loss=avg_eval_loss,
            avg_cosine_pos=avg_cosine_pos,
            avg_cosine_neg=avg_cosine_neg,
            avg_mse=avg_mse,
        )
        return self._on_eval_end(avg_mrr, avg_cosine_pos)

    def _log_non_finite_loss_if_needed(self, loss):
        if not torch.isfinite(loss):
            self.log_adapter.log_metrics(
                {"train/nonfinite_loss_steps": 1}, step=self.global_step
            )
            return True
        return False

    def _log_step_if_needed(
        self,
        epoch: int,
        step_in_epoch: int,
        step_start_time,
        running_loss: float,
        running_examples: int,
        pos_score,
        neg_score,
        pos_examples_in_batch: int,
        mse: Optional[float] = None,
        cosine_pos: Optional[float] = None,
        cosine_neg: Optional[float] = None,
    ):
        if self.global_step % self.log_adapter.log_every_n_steps != 0:
            return running_loss, running_examples

        step_end = time.time()
        step_time = step_end - step_start_time

        grad_norm = -float("inf")
        if self.log_grad_norm:
            grad_norm = compute_total_grad_norm(self.model)

        # Ranking metrics
        mrr, hits = RelgatEval.compute_mrr_hits(
            pos_score=pos_score, neg_score=neg_score, ks=self.eval_ks_ranks
        )

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
            "train/mrr": mrr,
            "train/pos_score_mean": (
                pos_score.detach().mean().item()
                if pos_examples_in_batch > 0
                else 0.0
            ),
            "train/neg_score_mean": (
                neg_score.detach().mean().item() if neg_score.numel() > 0 else 0.0
            ),
        }

        # Multi loss
        if self.architecture.project_to_input_size:
            metrics["train/cosine_pos"] = cosine_pos
            metrics["train/cosine_neg"] = cosine_neg
            metrics["train/mse"] = mse

        # Hits@k
        for k in self.eval_ks_ranks:
            metrics[f"train/hits@{k}"] = hits.get(k, 0.0)

        self.log_adapter.log_metrics(metrics=metrics, step=self.global_step)

        return 0.0, 1

    def _eval_step_if_needed_and_end_training(self, epoch: int, epoch_loss: float):
        if (
            self.eval_every_n_steps is None
            or self.global_step % self.eval_every_n_steps != 0
        ):
            return False
        should_stop = self._run_eval_and_maybe_early_stop(epoch=epoch)
        if should_stop:
            return True

        # Change model to `train` mode
        self.model.train()
        return False

    def _log_begin_information(self):
        self.log_adapter.log_metrics(
            metrics={
                "scheduler/total_steps": self.training_scheduler.total_steps,
                "scheduler/warmup_steps": self.training_scheduler.warmup_steps,
                "scheduler/type": self.training_scheduler.scheduler_type,
                "config/use_self_adv_neg": float(self.relgat_loss.use_self_adv_neg),
                "config/self_adv_alpha": float(self.relgat_loss.self_adv_alpha),
                "train/base_lr": self.training_scheduler.base_lr,
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
            should_stop = self._run_eval_and_maybe_early_stop(epoch=epoch)
        return should_stop
