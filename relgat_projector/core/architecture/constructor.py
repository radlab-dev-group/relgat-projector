from typing import Any, Dict, Optional


class ModelArchitectureConstructor:
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
        project_to_input_size: bool,
        projection_layers: int,
        projection_dropout: float,
        projection_hidden_dim: int,
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

        self.project_to_input_size = run_config.get(
            "project_to_input", project_to_input_size
        )
        self.projection_layers = run_config.get(
            "projection_layers", projection_layers
        )
        self.projection_dropout = run_config.get(
            "projection_dropout", projection_dropout
        )
        self.projection_hidden_dim = run_config.get(
            "projection_hidden_dim", projection_hidden_dim
        )

        if self.architecture_name is not None:
            self.__check_architecture()

    def __check_architecture(self):
        # TODO: Check if name of architecture is consistent with predefined size
        pass
