import json

from datetime import datetime
from typing import Optional, Dict, Any

from rdl_ml_utils.handlers.wandb_handler import WanDBHandler


class LoggerAdapter:
    """
    Thin adapter preserving behavior; currently just forwards to WanDBHandler.
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
    ):
        self.run_name = run_name if run_name is not None else ""
        self.log_to_wandb = log_to_wandb
        self.log_to_console = log_to_console

        self.wandb_config = wandb_config
        self.log_every_n_steps = run_config.get(
            "log_every_n_steps", log_every_n_steps
        )
        if self.log_every_n_steps is None or int(self.log_every_n_steps) < 0:
            self.log_every_n_steps = 1
        else:
            self.log_every_n_steps = int(self.log_every_n_steps)

        self.architecture_name = architecture_name
        self.run_config = run_config

        self._prepare_run_name()

    def _prepare_run_name(self) -> str:
        """
        Prepare a run name in the form: '{model-name}-{architecture_name}-YYYYMMDD_HHMMSS'.

        If architecture_name is None or empty, 'run' is used as a default prefix.
        """
        run_name = self.run_config.get("run_name", self.run_name)

        prefix = ""
        if run_name is not None and len(run_name.strip()):
            prefix = run_name.strip()
        else:
            base_model_name = self.run_config.get("base_model_name", "")
            if base_model_name is not None and len(base_model_name):
                prefix = base_model_name.strip() + "-"
            prefix += self.architecture_name if self.architecture_name else "run"
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{prefix}-{date_str}"
        return self.run_name

    def init_wandb_if_needed(self, training_args=None):
        if self.log_to_wandb:
            WanDBHandler.init_wandb(
                wandb_config=self.wandb_config,
                run_config=self.run_config,
                training_args=training_args,
                run_name=self.run_name,
            )

    def log_metrics(self, metrics, step: int):
        if self.log_to_wandb:
            WanDBHandler.log_metrics(metrics=metrics, step=step)

        if self.log_to_console:
            print(f"[{self.run_name}] Step {step}:")
            if type(metrics) in [dict, list]:
                metrics = json.dumps(metrics, indent=2, ensure_ascii=False)
            print(metrics)

    def finish_wand_if_needed(self):
        if self.log_to_wandb:
            WanDBHandler.finish_wand()
