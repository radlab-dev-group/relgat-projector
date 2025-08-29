import abc
import torch
import random

import numpy as np

from typing import Tuple, List, Any, Dict, Optional

from relgat_llm.base.constants import ConstantsRelGATTrainer

# distmult or transe


class AnyModelArchitectureConstructorI(abc.ABC):
    def __init__(
        self,
        gat_out_dim: int,
        gat_heads: int,
        gat_num_layers: int,
        dropout: float,
        dropout_rel_attention: float,
        scorer_type: str,
        architecture_name: Optional[str],
        base_model_name: Optional[str],
        run_config: Dict[str, Any],
    ):
        self.gat_out_dim = int(run_config.get("gat_out_dim", gat_out_dim))
        self.gat_heads = int(run_config.get("gat_heads", gat_heads))
        self.gat_num_layers = int(run_config.get("gat_num_layers", gat_num_layers))
        self.dropout = float(run_config.get("dropout", dropout))
        self.dropout_rel_attention = float(
            run_config.get("dropout_rel_attention", dropout_rel_attention)
        )
        self.scorer_type = str(run_config.get("scorer_type", scorer_type))
        self.architecture_name = architecture_name
        self.base_model_name = base_model_name

        if self.architecture_name is not None:
            self.__check_architecture()

    def __check_architecture(self):
        # TODO: Check if name of architecture is consistent with predefined size
        pass
