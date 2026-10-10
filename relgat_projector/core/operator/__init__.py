from relgat_projector.core.operator.base import RelationalOperator
from relgat_projector.core.operator.givens import GivensRelationOperator
from relgat_projector.core.operator.compiled import CompiledGivensOperator
from relgat_projector.core.operator.builtin_distmult import DistMultOperator
from relgat_projector.core.operator.builtin_transe import TransEOperator

OPERATOR_REGISTRY = {
    "givens": GivensRelationOperator,
    "distmult": DistMultOperator,
    "transe": TransEOperator,
}


def build_operator(kind: str, **kwargs):
    kind = str(kind).lower()
    if kind not in OPERATOR_REGISTRY:
        raise ValueError(
            f"Unknown operator_type: {kind}. "
            f"Available: {sorted(OPERATOR_REGISTRY)}"
        )
    return OPERATOR_REGISTRY[kind](**kwargs)


__all__ = [
    "RelationalOperator",
    "GivensRelationOperator",
    "CompiledGivensOperator",
    "DistMultOperator",
    "TransEOperator",
    "OPERATOR_REGISTRY",
    "build_operator",
]
