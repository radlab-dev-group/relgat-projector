import torch
import random

import numpy as np

from typing import Any, Dict


class RandomSeed:
    def __init__(
        self, seed: int, run_config: Dict[str, Any], auto_set_seed: bool = True
    ):
        self.seed = int(run_config.get("seed", seed))

        if auto_set_seed:
            self.set_random_state()

    def set_random_state(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
