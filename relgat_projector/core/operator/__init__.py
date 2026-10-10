import inspect

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


def operator_spec(kind: str) -> list:
    """Constructor keyword names accepted by ``kind`` (for the ablation grid)."""
    kind = str(kind).lower()
    if kind not in OPERATOR_REGISTRY:
        raise ValueError(f"Unknown operator_type: {kind}. Available: {sorted(OPERATOR_REGISTRY)}")
    params = inspect.signature(OPERATOR_REGISTRY[kind].__init__).parameters
    return [name for name, p in list(params.items())[1:] if p.kind is not p.VAR_KEYWORD]


def build_operator(kind: str, **kwargs):
    """Instantiate an operator by name, dropping options it does not accept.

    The trainer passes the full ``givens`` option set for every operator so a
    single configuration drives the ablation matrix (PLAN.md 5); baselines such
    as ``distmult``/``transe`` simply do not implement generators, scales or
    context, so their unsupported keywords are ignored here rather than raising.
    """
    kind = str(kind).lower()
    if kind not in OPERATOR_REGISTRY:
        raise ValueError(
            f"Unknown operator_type: {kind}. "
            f"Available: {sorted(OPERATOR_REGISTRY)}"
        )
    cls = OPERATOR_REGISTRY[kind]
    params = inspect.signature(cls.__init__).parameters
    accepted = [
        name
        for name, p in list(params.items())[1:]
        if p.kind is not p.VAR_KEYWORD
    ]
    kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    op = cls(**kwargs)
    # record the *effective* construction (explicit arguments plus defaults):
    # this is what a checkpoint has to reproduce to rebuild the same operator
    effective = {
        name: params[name].default
        for name in accepted
        if params[name].default is not inspect.Parameter.empty
    }
    effective.update(kwargs)
    op.ctor_kwargs = {"type": kind, "kwargs": effective}
    return op


__all__ = [
    "RelationalOperator",
    "GivensRelationOperator",
    "CompiledGivensOperator",
    "DistMultOperator",
    "TransEOperator",
    "OPERATOR_REGISTRY",
    "build_operator",
    "operator_spec",
]
