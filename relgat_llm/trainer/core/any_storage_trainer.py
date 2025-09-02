import abc
import json
import shutil

import torch

from pathlib import Path
from collections import deque
from typing import Tuple, List, Any, Dict, Optional

from relgat_llm.base.constants import ConstantsRelGATTrainer


class RelGATTrainerBaseStorageI(abc.ABC):
    def __init__(
        self,
        out_dir: str,
        run_config: Dict[str, Any],
        max_checkpoints: Optional[int] = None,
    ):
        self.out_dir = str(run_config.get("out_dir", out_dir))
        self.max_checkpoints = int(
            run_config.get("max_checkpoints", max_checkpoints)
        )

        # Fifo queue
        self.best_ckpt_dir: Path | str | None = None
        self.saved_checkpoints: deque[Path] = deque()

        self.save_dir = Path(
            out_dir
            if out_dir is not None
            else ConstantsRelGATTrainer.Default.DEFAULT_TRAINER_OUT_DIR
        )
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _save_model_and_files(
        self, subdir: str, model, files: List[Tuple[str, Dict[Any, Any]]]
    ) -> str:
        out_dir = self.save_dir / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / ConstantsRelGATTrainer.Default.OUT_MODEL_NAME
        torch.save(model.state_dict(), out_path)
        for f_name, json_data in files:
            out_cfg_path = out_dir / f_name
            with open(out_cfg_path, "w") as f:
                f.write(json.dumps(json_data, indent=2, ensure_ascii=False))
        return str(out_path)

    def _prune_checkpoints(self) -> None:
        """
        Keeps a maximum of max_checkpoints most recent (or best) checkpoints.
        The oldest ones are deleted from the disc.
        """
        if self.max_checkpoints is None or self.max_checkpoints < 1:
            return

        while len(self.saved_checkpoints) > self.max_checkpoints:
            oldest = self.saved_checkpoints.popleft()
            try:
                shutil.rmtree(oldest)
                print(f"️ Removed old checkpoint: {oldest}")
            except Exception as exc:
                print(f" Could not delete {oldest}: {exc}")
