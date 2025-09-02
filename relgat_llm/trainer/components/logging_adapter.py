from rdl_ml_utils.handlers.wandb import WanDBHandler


class LoggerAdapter:
    """
    Thin adapter preserving behavior; currently just forwards to WanDBHandler.
    """

    @staticmethod
    def log_metrics(metrics, step: int):
        WanDBHandler.log_metrics(metrics=metrics, step=step)

    @staticmethod
    def init_wandb(wandb_config, run_config, training_args, run_name):
        WanDBHandler.init_wandb(
            wandb_config=wandb_config,
            run_config=run_config,
            training_args=training_args,
            run_name=run_name,
        )

    @staticmethod
    def finish_wand():
        WanDBHandler.finish_wand()
