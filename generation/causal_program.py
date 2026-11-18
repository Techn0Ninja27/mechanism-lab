"""A small fail-closed causal vector-program runtime.

Programs are JSON ASTs, not Python expressions.  They are intended as a
common execution target for proposal-generation arms.  A program has a
``predict`` expression and an ``update`` expression.  Both expressions see a
snapshot of the state before the step; the update is committed only after the
prediction is produced.  The runtime exposes observations only through
``observe`` at synchronization/warmup points, so a candidate cannot read
intermediate or future truth.

This module provides a deterministic Torch backend and makes no claim about
novelty, usefulness, or scientific validity.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any

import torch

MAX_NODES = 32
MAX_DEPTH = 6
EPS = 1e-4
VARS = {"last_sync", "previous_sync", "memory", "age", "observed_interval"}
OPS = {"add", "sub", "mul", "divsafe", "neg", "abs", "relu", "tanh", "clip", "norm", "mean"}


def _number(x: Any, name: str = "constant") -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        raise ValueError(f"{name} must be a finite number")
    value = float(x)
    if not -4.0 <= value <= 4.0:
        raise ValueError(f"{name} is outside [-4, 4]")
    return value


def _shape(value: torch.Tensor, expected: tuple[int, int]) -> None:
    if value.ndim not in (0, 2):
        raise ValueError("expression values must be scalar or rank-2")
    if value.ndim == 2 and tuple(value.shape) not in {expected, (expected[0], 1)}:
        raise ValueError(f"invalid vector shape {tuple(value.shape)}; expected {expected} or Nx1")
    if not torch.isfinite(value).all().item():
        raise ValueError("expression produced a non-finite value")


def _broadcast(a: torch.Tensor, b: torch.Tensor, expected: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    _shape(a, expected); _shape(b, expected)
    def expand(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 0: return x.expand(expected)
        if tuple(x.shape) == (expected[0], 1): return x.expand(expected)
        return x
    return expand(a), expand(b)


@dataclass(frozen=True)
class CompiledProgram:
    predict: Mapping[str, Any]
    update: Mapping[str, Any]
    node_count: int

    def __call__(self, state: "CausalState", t: int) -> torch.Tensor:
        return state.predict(t, self)


def _validate(node: Any, depth: int, count: list[int], *, constant_only: bool = False) -> None:
    if depth > MAX_DEPTH or not isinstance(node, Mapping):
        raise ValueError("malformed or too-deep AST")
    keys = set(node)
    if "var" in node:
        if keys != {"var"} or not isinstance(node.get("var"), str) or node["var"] not in VARS or constant_only:
            raise ValueError("unknown or unavailable variable")
        count[0] += 1; return
    if "const" in node:
        if keys != {"const"}: raise ValueError("malformed constant")
        _number(node["const"]); count[0] += 1; return
    if (keys != {"op", "args"} or not isinstance(node.get("op"), str)
            or node["op"] not in OPS or not isinstance(node["args"], list)):
        raise ValueError("malformed operation")
    op, args = node["op"], node["args"]
    arity = {"add":2,"sub":2,"mul":2,"divsafe":2,"neg":1,"abs":1,"relu":1,"tanh":1,
             "clip":3,"norm":1,"mean":1}[op]
    if len(args) != arity: raise ValueError(f"{op} has wrong arity")
    count[0] += 1
    for i, arg in enumerate(args):
        if op == "clip" and i in (1, 2):
            if not isinstance(arg, Mapping) or set(arg) != {"const"}:
                raise ValueError("clip bounds must be constant leaves")
        _validate(arg, depth + 1, count, constant_only=False)
    if op == "clip":
        _number(args[1]["const"], "clip bound"); _number(args[2]["const"], "clip bound")
        if float(args[1]["const"]) > float(args[2]["const"]): raise ValueError("clip bounds reversed")


def validate_program(program: Any) -> CompiledProgram:
    if not isinstance(program, Mapping) or set(program) != {"predict", "update"}:
        raise ValueError("program must contain exactly predict and update")
    predict_count = [0]; update_count = [0]
    _validate(program["predict"], 1, predict_count); _validate(program["update"], 1, update_count)
    if predict_count[0] > MAX_NODES or update_count[0] > MAX_NODES:
        raise ValueError("an expression exceeds the node limit")
    def freeze(value: Any) -> Any:
        if isinstance(value, Mapping):
            return MappingProxyType({k: freeze(v) for k, v in value.items()})
        if isinstance(value, list): return tuple(freeze(v) for v in value)
        return value
    return CompiledProgram(freeze(program["predict"]), freeze(program["update"]),
                           predict_count[0] + update_count[0])


def _eval(node: Mapping[str, Any], ctx: Mapping[str, torch.Tensor], expected: tuple[int, int]) -> torch.Tensor:
    if "var" in node: return ctx[node["var"]]
    if "const" in node: return torch.as_tensor(float(node["const"]), dtype=ctx["memory"].dtype, device=ctx["memory"].device)
    op, args = node["op"], node["args"]
    vals = [_eval(a, ctx, expected) for a in args]
    if op in {"add", "sub", "mul", "divsafe"}:
        a, b = _broadcast(vals[0], vals[1], expected)
        if op == "add": out = a + b
        elif op == "sub": out = a - b
        elif op == "mul": out = a * b
        else:
            sign = torch.where(b < 0, -torch.ones_like(b), torch.ones_like(b))
            out = a / torch.where(b.abs() < EPS, sign * EPS, b)
    elif op == "neg": out = -vals[0]
    elif op == "abs": out = vals[0].abs()
    elif op == "relu": out = torch.relu(vals[0])
    elif op == "tanh": out = torch.tanh(vals[0])
    elif op == "clip": out = torch.clamp(vals[0], min=float(args[1]["const"]), max=float(args[2]["const"]))
    elif op in {"norm", "mean"}:
        _shape(vals[0], expected)
        if vals[0].ndim != 2:
            raise ValueError(f"{op} requires an NxD vector")
        out = (torch.linalg.vector_norm(vals[0], dim=1, keepdim=True)
               if op == "norm" else vals[0].mean(dim=1, keepdim=True))
    else: raise ValueError("unknown operation")
    _shape(out, expected); return out


class CausalState:
    """Runtime state. ``observe`` is the only way exact values enter it."""
    def __init__(self, shape: tuple[int, int], *, device: str | torch.device = "cpu", dtype=torch.float32):
        if (not isinstance(shape, tuple) or len(shape) != 2
                or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in shape)):
            raise ValueError("invalid state shape")
        self.shape = shape; self.memory = torch.zeros(shape, device=device, dtype=dtype)
        self.last_sync = torch.zeros_like(self.memory); self.previous_sync = torch.zeros_like(self.memory)
        self.last_t: int | None = None; self.previous_t: int | None = None
        self.last_action_t: int | None = None; self.last_action_kind: str | None = None

    def _ctx(self, *, last: torch.Tensor, previous: torch.Tensor, memory: torch.Tensor,
             age: int, interval: int) -> dict[str, torch.Tensor]:
        scalar = lambda x: torch.tensor(float(x), device=self.memory.device, dtype=self.memory.dtype)
        return {"last_sync": last, "previous_sync": previous, "memory": memory,
                "age": scalar(age), "observed_interval": scalar(interval)}

    def observe(self, value: torch.Tensor, t: int, program: CompiledProgram) -> torch.Tensor:
        if not isinstance(program, CompiledProgram): program = validate_program(program)
        value = torch.as_tensor(value, device=self.memory.device, dtype=self.memory.dtype).detach().clone()
        if tuple(value.shape) != self.shape or not torch.isfinite(value).all().item(): raise ValueError("invalid observation")
        if (isinstance(t, bool) or not isinstance(t, int) or t < 0
                or (self.last_action_t is not None and (t < self.last_action_t
                    or (t == self.last_action_t and self.last_action_kind != "predict")))):
            raise ValueError("time must increase")
        old_last, old_t, old_memory = self.last_sync.clone(), self.last_t, self.memory.clone()
        interval = t - old_t if old_t is not None else 1
        ctx = self._ctx(last=value, previous=old_last, memory=old_memory, age=0, interval=interval)
        updated = _eval(program.update, ctx, self.shape)
        if updated.ndim == 0 or tuple(updated.shape) == (self.shape[0], 1): updated = updated.expand(self.shape)
        updated = updated.detach().clone()
        self.previous_sync, self.previous_t = old_last, old_t
        self.last_sync, self.last_t, self.memory = value, t, updated
        self.last_action_t, self.last_action_kind = t, "observe"
        return value.clone()

    def predict(self, t: int, program: CompiledProgram) -> torch.Tensor:
        if self.last_t is None: raise ValueError("observe a warmup value before predict")
        if (isinstance(t, bool) or not isinstance(t, int) or t <= self.last_t
                or (self.last_action_t is not None and t <= self.last_action_t)):
            raise ValueError("prediction time must be after the previous action")
        age = t - self.last_t; interval = self.last_t - self.previous_t if self.previous_t is not None else 1
        ctx = self._ctx(last=self.last_sync, previous=self.previous_sync, memory=self.memory,
                        age=age, interval=max(1, interval))
        predicted = _eval(program.predict, ctx, self.shape)
        if predicted.ndim == 0 or tuple(predicted.shape) == (self.shape[0], 1): predicted = predicted.expand(self.shape)
        updated = _eval(program.update, ctx, self.shape)
        if updated.ndim == 0 or tuple(updated.shape) == (self.shape[0], 1): updated = updated.expand(self.shape)
        self.memory = updated.detach().clone(); self.last_action_t, self.last_action_kind = t, "predict"
        return predicted.detach().clone()


def compile_program(program: Any) -> CompiledProgram:
    return validate_program(program)
