"""Utility for reproducible random seeds across Python, NumPy and PyTorch."""

import random
from typing import Any, Dict, Optional

import torch
import numpy as np


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set the random seed for ``random``, ``numpy`` and ``torch`` globally.

    Parameters
    ----------
    seed : int
        The seed value to use.
    deterministic : bool, default ``True``
        When ``True`` forces deterministic CUDA kernels (if available) and
        disables cuDNN benchmark to guarantee reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class RandomSeed:
    """
    Small wrapper around :func:`set_global_seed` that reads the seed from a
    configuration dictionary and optionally applies it immediately.
    """

    def __init__(
        self,
        seed: int,
        run_config: Dict[str, Any],
        auto_set_seed: bool = True,
        deterministic: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        seed : int
            Default seed value used when ``run_config`` does not contain a ``seed`` key.
        run_config : dict
            Mapping that may contain a ``seed`` entry.
        auto_set_seed : bool, default ``True``
            If ``True`` the seed is set during construction.
        deterministic : bool, default ``True``
            Propagate to :func:`set_global_seed`.
        """
        self.seed: int = int(run_config.get("seed", seed))
        self.deterministic: bool = deterministic

        if auto_set_seed:
            self.set_random_state()

    def set_random_state(self) -> None:
        """Apply the stored seed to all supported RNGs."""
        set_global_seed(self.seed, deterministic=self.deterministic)

    def __repr__(self) -> str:
        return f"RandomSeed(seed={self.seed}, deterministic={self.deterministic})"
