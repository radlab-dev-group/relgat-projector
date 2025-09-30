"""Adapter that forwards logging calls to WandB or to the standard logger."""

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from rdl_ml_utils.handlers.wandb_handler import WanDBHandler


class LoggerAdapter:
    """
    Thin wrapper around :class:`WanDBHandler` that also emits human‑readable
    logs to the console via the standard ``logging`` module.
    """

    def __init__(
        self,
        run_name: Optional[str],
        architecture_name: Optional[str],
        run_config: Optional[Dict[str, Any]] = None,
        wandb_config: Optional[Any] = None,
        log_every_n_steps: int = 100,
        log_to_wandb: bool = False,
        log_to_console: bool = True,
    ) -> None:
        self.run_name = (run_name or "").strip()
        self.architecture_name = architecture_name
        self.run_config = run_config or {}
        self.wandb_config = wandb_config
        self.log_to_wandb = log_to_wandb
        self.log_to_console = log_to_console

        # Resolve logging interval
        self.log_every_n_steps = int(
            self.run_config.get("log_every_n_steps", log_every_n_steps)
        )
        if self.log_every_n_steps <= 0:
            self.log_every_n_steps = 1

        # Prepare a deterministic run identifier
        self.run_name = self._prepare_run_name()

        # Configure standard logger (only once)
        self._logger = logging.getLogger("relgat")
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                fmt="[{asctime}] %(levelname)s %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                style="{",
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

    # --------------------------------------------------------------------- #
    # Private helpers
    # --------------------------------------------------------------------- #
    def _prepare_run_name(self) -> str:
        """
        Build a run name in the form:

        ``{prefix}-{YYYYMMDD_HHMMSS}``

        ``prefix`` is taken from ``run_config['run_name']`` or, if missing,
        from ``base_model_name`` + ``architecture_name`` (fallback ``run``).
        """
        run_name_cfg = self.run_config.get("run_name", "").strip()
        if run_name_cfg:
            prefix = run_name_cfg
        else:
            base = self.run_config.get("base_model_name", "").strip()
            prefix = (
                f"{base}-{self.architecture_name}"
                if base
                else self.architecture_name or "run"
            )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}-{timestamp}"

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def init_wandb_if_needed(self, training_args: Optional[Any] = None) -> None:
        """Initialise WandB if the flag is set."""
        if self.log_to_wandb:
            WanDBHandler.init_wandb(
                wandb_config=self.wandb_config,
                run_config=self.run_config,
                training_args=training_args,
                run_name=self.run_name,
            )

    def log_metrics(self, metrics: Any, step: int) -> None:
        """
        Log a metrics dictionary both to WandB (if enabled) and to the
        console logger.
        """
        if self.log_to_wandb:
            WanDBHandler.log_metrics(metrics=metrics, step=step)

        if self.log_to_console:
            if isinstance(metrics, (dict, list)):
                formatted = json.dumps(metrics, indent=2, ensure_ascii=False)
            else:
                formatted = str(metrics)
            self._logger.info("Step %d:\n%s", step, formatted)

    def finish_wand_if_needed(self) -> None:
        """Terminate the WandB session when training ends."""
        if self.log_to_wandb:
            WanDBHandler.finish_wand()
