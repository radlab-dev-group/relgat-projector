import abc
import torch
import random

import numpy as np

from typing import Tuple, List, Any, Dict, Optional


class AnyReproductiveTrainerI(abc.ABC):
    def __init__(self, seed: int, run_config: Dict[str, Any]):
        self.seed = int(run_config.get("seed", seed))
        self._set_random_state()

    def _set_random_state(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
