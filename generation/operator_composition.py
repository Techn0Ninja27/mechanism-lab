"""Small, typed, executable operator chains.

This module deliberately implements only linear chains.  Operators are named
registry entries (never dynamically imported or evaluated), and every runtime
boundary is checked.  A chain is an executable hypothesis, not a novelty
claim.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from .molecules import atom_record, molecule_record


class OperatorCompositionError(ValueError):
    pass


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _callable_digest(fn: Callable[..., Any]) -> str:
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError) as exc:
        raise OperatorCompositionError("callable source is unavailable; provide implementation_digest") from exc
    return hashlib.sha256(source.encode()).hexdigest()


def _list_ints(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(x, int) and not isinstance(x, bool) for x in value)


def _unsigned(value: Any) -> bool:
    return _list_ints(value) and all(x >= 0 for x in value)


TYPE_VALIDATORS: dict[str, Callable[[Any], bool]] = {
    "signed_ints": _list_ints,
    "unsigned_ints": _unsigned,
    "bytes": lambda x: isinstance(x, bytes),
    "string": lambda x: isinstance(x, str),
    "strings": lambda x: isinstance(x, list) and all(isinstance(y, str) for y in x),
}


@dataclass(frozen=True)
class Operator:
    operator_id: str
    input_type: str
    output_type: str
    encode: Callable[[Any], Any]
    decode: Callable[[Any], Any]
    version: str = "1"
    source: str = "unspecified"
    provenance: tuple[Mapping[str, Any], ...] = ()
    assumptions: tuple[str, ...] = ()
    contract_status: str = "untested"
    implementation_digest: str | None = None

    def __post_init__(self) -> None:
        if not self.operator_id.strip() or not self.input_type.strip() or not self.output_type.strip():
            raise OperatorCompositionError("operator ID and types must be non-empty")
        if not callable(self.encode) or not callable(self.decode):
            raise OperatorCompositionError("encode and decode must be callable")
        if self.contract_status not in {"tested", "untested"}:
            raise OperatorCompositionError("contract_status must be tested or untested")
        if not self.version.strip() or not self.source.strip():
            raise OperatorCompositionError("version and source must be non-empty")
        if self.implementation_digest is not None and (len(self.implementation_digest) != 64 or any(c not in "0123456789abcdefABCDEF" for c in self.implementation_digest)):
            raise OperatorCompositionError("implementation_digest must be a SHA-256 hex digest")

    @property
    def content_id(self) -> str:
        digest = self.implementation_digest or _digest({"encode": _callable_digest(self.encode), "decode": _callable_digest(self.decode)})
        return "op_" + _digest({"operator_id": self.operator_id, "input": self.input_type, "output": self.output_type,
                                 "version": self.version, "source": self.source, "implementation_digest": digest,
                                 "contract_status": self.contract_status, "assumptions": self.assumptions,
                                 "provenance": list(self.provenance)})[:32]


class OperatorRegistry:
    def __init__(self, *, type_validators: Mapping[str, Callable[[Any], bool]] | None = None) -> None:
        self._operators: dict[str, Operator] = {}
        self.type_validators = dict(TYPE_VALIDATORS)
        if type_validators:
            self.type_validators.update(type_validators)

    def register(self, operator: Operator) -> Operator:
        if operator.operator_id in self._operators:
            raise OperatorCompositionError(f"duplicate operator ID: {operator.operator_id}")
        if operator.input_type not in self.type_validators or operator.output_type not in self.type_validators:
            raise OperatorCompositionError("both operator types need registered validators")
        self._operators[operator.operator_id] = operator
        return operator

    def get(self, operator_id: str) -> Operator:
        try:
            return self._operators[operator_id]
        except KeyError as exc:
            raise OperatorCompositionError(f"unknown operator: {operator_id}") from exc

    def __iter__(self):
        return iter(self._operators.values())

    def enumerate_chains(self, input_type: str, *, max_depth: int = 3, max_candidates: int = 100,
                         allowed_registry_ids: Iterable[str] | None = None, no_repeated_operators: bool = True,
                         output_type: str | None = None) -> list["OperatorChain"]:
        if type(max_depth) is not int or type(max_candidates) is not int or not 1 <= max_depth <= 8 or not 1 <= max_candidates <= 4096:
            raise OperatorCompositionError("max_depth must be an int in [1, 8] and max_candidates an int in [1, 4096]")
        allowed = set(allowed_registry_ids) if allowed_registry_ids is not None else set(self._operators)
        unknown = allowed - set(self._operators)
        if unknown:
            raise OperatorCompositionError("unknown allowed operator: " + sorted(unknown)[0])
        out: list[OperatorChain] = []

        def visit(current: str, chosen: list[Operator]) -> None:
            if len(out) >= max_candidates or len(chosen) >= max_depth:
                return
            for op in sorted(self._operators.values(), key=lambda x: x.operator_id):
                if op.operator_id not in allowed or op.input_type != current or (no_repeated_operators and op in chosen):
                    continue
                chain = OperatorChain(tuple(chosen + [op]), self)
                if output_type is None or chain.output_type == output_type:
                    out.append(chain)
                visit(op.output_type, chosen + [op])
                if len(out) >= max_candidates:
                    return
        visit(input_type, [])
        return out


@dataclass(frozen=True)
class OperatorChain:
    operators: tuple[Operator, ...]
    registry: OperatorRegistry

    def __post_init__(self) -> None:
        if not self.operators:
            raise OperatorCompositionError("a chain must contain at least one operator")
        for op in self.operators:
            try:
                registered = self.registry.get(op.operator_id)
            except OperatorCompositionError:
                raise OperatorCompositionError(f"operator is not registered: {op.operator_id}") from None
            if registered is not op and registered.content_id != op.content_id:
                raise OperatorCompositionError(f"operator definition does not match registry: {op.operator_id}")
        for left, right in zip(self.operators, self.operators[1:]):
            if left.output_type != right.input_type:
                raise OperatorCompositionError(f"type mismatch: {left.operator_id} outputs {left.output_type}, {right.operator_id} expects {right.input_type}")

    @property
    def input_type(self) -> str: return self.operators[0].input_type
    @property
    def output_type(self) -> str: return self.operators[-1].output_type
    @property
    def content_id(self) -> str: return "chain_" + _digest({"operators": [x.content_id for x in self.operators]})[:32]

    def _check(self, type_name: str, value: Any) -> None:
        validator = self.registry.type_validators[type_name]
        if not validator(value):
            raise OperatorCompositionError(f"value does not satisfy type {type_name}")

    def execute(self, value: Any) -> Any:
        self._check(self.input_type, value)
        current = value
        for op in self.operators:
            current = op.encode(current)
            self._check(op.output_type, current)
        return current

    def decode(self, value: Any) -> Any:
        self._check(self.output_type, value)
        current = value
        for op in reversed(self.operators):
            current = op.decode(current)
            self._check(op.input_type, current)
        return current

    def to_molecule(self) -> dict[str, Any]:
        node_ids = [op.operator_id if self.operators.count(op) == 1 else f"{op.operator_id}#{i}" for i, op in enumerate(self.operators)]
        # Occurrence IDs make reuse explicit while retaining the underlying operator identity.
        atoms = [atom_record(node_id, f"typed operator {op.operator_id} (content {op.content_id}): {op.input_type} -> {op.output_type}",
                             tuple(op.provenance) + ({"source": op.source, "operator_id": op.operator_id, "content_id": op.content_id},))
                for node_id, op in zip(node_ids, self.operators)]
        relations = [{"source": a, "target": b, "type": "composes"} for a, b in zip(node_ids, node_ids[1:])]
        assumptions = [f"operator {op.operator_id} contract status is {op.contract_status}" for op in self.operators]
        assumptions.extend(a for op in self.operators for a in op.assumptions)
        return molecule_record(atoms, relations, mechanism="; ".join(f"{op.operator_id}: {op.input_type}->{op.output_type}" for op in self.operators),
                               assumptions=assumptions or ["composition is evaluated only through registered callables"],
                               falsifiable_prediction="decode(execute(x)) equals x for values satisfying the input contract",
                               status="combination", provenance=[{"chain_id": self.content_id, "operator_ids": [op.operator_id for op in self.operators]}])


def compose_chain(registry: OperatorRegistry, operator_ids: Iterable[str], *, allowed_registry_ids: Iterable[str] | None = None) -> OperatorChain:
    ids = list(operator_ids)
    if not ids:
        raise OperatorCompositionError("operator_ids must not be empty")
    if allowed_registry_ids is not None and not set(ids) <= set(allowed_registry_ids):
        raise OperatorCompositionError("chain contains an operator outside allowed_registry_ids")
    return OperatorChain(tuple(registry.get(x) for x in ids), registry)


# Descriptive aliases keep call sites readable while retaining one implementation.
ExecutableOperator = Operator
OperatorComposition = OperatorChain


def enumerate_operator_chains(registry: OperatorRegistry, input_type: str, **kwargs: Any) -> list[OperatorChain]:
    """Bounded convenience wrapper around :meth:`OperatorRegistry.enumerate_chains`."""
    return registry.enumerate_chains(input_type, **kwargs)
