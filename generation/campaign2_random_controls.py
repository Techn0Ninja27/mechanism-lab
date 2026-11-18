"""Deterministic, bounded random controls for campaign 2.

The family is intentionally narrow: each program predicts ``last_sync`` plus
an elementwise perturbation ``g * tanh(vector) * last_sync`` with
``0 < abs(g) <= .1``.  A fixed PRNG seed samples signed gain magnitudes uniformly in [.01,.1],
EMA coefficients uniformly in [.1,.9], and six vector bases with replacement.  This is a control family, not an optimizer or
result-dependent search.
"""
from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from generation.causal_program import compile_program


VECTOR_BASES = (
    {"var": "last_sync"},
    {"var": "previous_sync"},
    {"op": "sub", "args": [{"var": "last_sync"}, {"var": "previous_sync"}]},
    {"var": "memory"},
    {"op": "sub", "args": [{"var": "previous_sync"}, {"var": "last_sync"}]},
    {"op": "add", "args": [{"var": "memory"}, {"var": "last_sync"}]},
)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _program(gain: float, alpha: float, vector: dict[str, Any]) -> dict[str, Any]:
    # tanh(vector) is in [-1, 1], so the perturbation is bounded elementwise by
    # |gain|*|last_sync|.  The update is an EMA and therefore makes the memory
    # vector informative after the first observation, including group 3.
    perturb = {"op": "mul", "args": [
        {"const": gain},
        {"op": "mul", "args": [
            {"op": "tanh", "args": [_copy(vector)]},
            {"var": "last_sync"},
        ]},
    ]}
    update = {"op": "add", "args": [
        {"op": "mul", "args": [{"const": alpha}, {"var": "last_sync"}]},
        {"op": "mul", "args": [{"const": round(1.0 - alpha, 6)}, {"var": "memory"}]},
    ]}
    program = {"predict": {"op": "add", "args":[{"var":"last_sync"}, perturb]},
               "update": update}
    compile_program(program)
    return program


def random_inventory(replicates: int = 6) -> list[dict[str, Any]]:
    """Return the frozen six-group, four-program-per-group control inventory.

    ``replicates`` is retained as an explicit plan parameter and must be six;
    silently changing it would change the predeclared control denominator.
    Programs are deterministic and do not inspect task results or model output.
    """
    if type(replicates) is not int or replicates != 6:
        raise ValueError("campaign2 random controls require exactly six groups")
    rng=random.Random(20261119)
    rows: list[dict[str, Any]] = []
    for group in range(replicates):
        for candidate in range(4):
            vector=rng.choice(VECTOR_BASES)
            candidate_gain=round(rng.choice((-1,1))*rng.uniform(.01,.1),6)
            alpha=round(rng.uniform(.1,.9),6)
            program = _program(candidate_gain, alpha, vector)
            row = {
                "id": f"random-{group:02d}-cand-{candidate:02d}",
                "call_id": f"random-{group:02d}", "arm": "random",
                "program": _copy(program), "program_sha256": _digest(program),
                "status": "valid", "control_family": "bounded_relative_perturbation_v1",
                "group": group, "gain": candidate_gain, "memory_alpha": alpha,
            }
            rows.append(row)
    if len(rows) != 24 or len({_digest(r["program"]) for r in rows}) != 24:
        raise AssertionError("random control inventory shape changed")
    return rows
