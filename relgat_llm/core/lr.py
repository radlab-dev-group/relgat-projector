import math
import torch


from typing import Any, Dict, Optional

from relgat_llm.base.constants import ConstantsRelGATTrainer


class TrainingScheduler:
    def __init__(
        self,
        lr: float,
        lr_scheduler: str,
        lr_decay: float,
        warmup_steps: Optional[int],
        run_config: Dict[str, Any],
    ):
        self.lr = float(run_config.get("lr", lr))
        self.lr_decay = float(run_config.get("lr_decay", lr_decay))
        self.lr_scheduler = str(run_config.get("lr_scheduler", lr_scheduler))

        # LR/scheduler config
        self.scheduler = None
        self.base_lr = lr

        self.total_steps = None
        self.warmup_steps = run_config.get("warmup_steps", warmup_steps)

        self.scheduler_type = str(
            run_config.get("lr_scheduler", lr_scheduler)
        ).lower()

        self.default_warmup_ratio = (
            ConstantsRelGATTrainer.Default.DEFAULT_WARMUP_RATIO
        )

    def prepare(self, epochs: int, train_dataset, train_batch_size: int, optimizer):
        self.__prepare_warmup_and_total_steps(
            epochs=epochs,
            train_dataset=train_dataset,
            train_batch_size=train_batch_size,
        )
        self.__prepare_lr_scheduler(optimizer=optimizer)

    def __prepare_lr_scheduler(self, optimizer):
        def _lr_lambda_linear(current_step: int):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            return max(
                0.0,
                float(self.total_steps - current_step)
                / float(max(1, self.total_steps - self.warmup_steps)),
            )

        def _lr_lambda_cosine(current_step: int):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            progress = float(current_step - self.warmup_steps) / float(
                max(1, self.total_steps - self.warmup_steps)
            )
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))

        def _lr_lambda_constant(current_step: int):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
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
            # multiplied by the decay factor.
            base_lambda = lr_lambda

            def lr_lambda(step: int):
                return base_lambda(step) * (
                    self.lr_decay ** max(0, step - self.warmup_steps)
                )

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lr_lambda
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lr_lambda
            )

    def __prepare_warmup_and_total_steps(
        self, epochs: int, train_dataset, train_batch_size: int
    ):
        steps_per_epoch = max(1, math.ceil(len(train_dataset) / train_batch_size))
        self.total_steps = steps_per_epoch * max(1, int(epochs))

        if self.warmup_steps is None:
            self.warmup_steps = int(self.default_warmup_ratio * self.total_steps)
        self.warmup_steps = min(self.warmup_steps, max(0, self.total_steps - 1))
