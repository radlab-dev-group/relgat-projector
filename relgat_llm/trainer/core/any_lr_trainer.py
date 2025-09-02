import abc
import math
import torch


from typing import Any, Dict

from relgat_llm.base.constants import ConstantsRelGATTrainer


class AnyLRTrainerI(abc.ABC):
    def __init__(
        self,
        lr: float,
        lr_scheduler: str,
        lr_decay: float,
        run_config: Dict[str, Any],
    ):
        self.lr = float(run_config.get("lr", lr))
        self.lr_decay = float(run_config.get("lr_decay", lr_decay))
        self.lr_scheduler = str(run_config.get("lr_scheduler", lr_scheduler))

        # LR/scheduler config
        self.scheduler = None
        self.base_lr = lr
        self.scheduler_type = str(
            run_config.get("lr_scheduler", lr_scheduler)
        ).lower()

        self.default_warmup_ratio = (
            ConstantsRelGATTrainer.Default.DEFAULT_WARMUP_RATIO
        )

    def prepare_lr_scheduler(self, optimizer, warmup_steps: int, total_steps: int):
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
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))

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
                optimizer, lr_lambda=lr_lambda
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lr_lambda
            )
