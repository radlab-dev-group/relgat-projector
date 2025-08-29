import abc
import torch
import random

import numpy as np

from typing import Tuple, List, Any, Dict, Optional

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
