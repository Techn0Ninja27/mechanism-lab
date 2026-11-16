"""Deterministic signed-int64 streams for the integer-composition study.

The public fixture API intentionally returns only the input stream.  It does not
return a distribution name, labels, or a target for a compressor.  ``final`` is
an explicit caller-visible mode and is refused unless the caller also passes
``allow_final=True``; this is a protocol guard, not a security boundary.
"""
from __future__ import annotations

import random
from typing import Literal

N = 4096
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1
MODES = ("development", "validation", "final")
SEEDS = {
    "development": (19011, 19013, 19017),
    "validation": (19111, 19113, 19117),
    "final": (19311, 19313, 19317),
}


def _clip(x: int) -> int:
    return max(INT64_MIN, min(INT64_MAX, int(x)))


def _rng(seed: int, stream: int) -> random.Random:
    return random.Random((seed * 0x9E3779B185EBCA87 + stream * 0xD1B54A32D192ED03) & ((1 << 64) - 1))


def _walk(rng: random.Random, n: int, *, drift: int = 0, jump: int = 9) -> list[int]:
    x = rng.randint(-200, 200)
    out = []
    for _ in range(n):
        x += drift + rng.randint(-jump, jump)
        out.append(_clip(x))
    return out


def _development_or_validation(seed: int) -> tuple[int, ...]:
    """Four stationary source families, with independent split seeds."""
    # Distinct substreams prevent accidental coupling between families while
    # preserving reproducibility and a fixed ordering for the benchmark.
    a = _walk(_rng(seed, 1), N, drift=2, jump=3)  # linear/smooth random walk

    r = _rng(seed, 2)
    x = r.randint(-40, 40)
    b: list[int] = []
    while len(b) < N:
        x = _clip(x + r.randint(-2, 2))
        b.extend([x] * r.randint(8, 64))  # bursty stationary runs
    b = b[:N]

    r = _rng(seed, 3)
    c = []
    for i in range(N):
        base = 100 * ((i % 32) - 16)
        spike = r.choice((0, 0, 0, 0, r.randint(-5000, 5000)))
        c.append(_clip(base + spike))  # periodic signal with sparse spikes

    r = _rng(seed, 4)
    d = [r.randint(-(1 << 31), (1 << 31) - 1) for _ in range(N)]
    # Interleave families into independent blocks.  The stream contains no
    # boundary metadata; this is intentionally just the raw telemetry stream.
    streams = (a, b, c, d)
    block = N // len(streams)
    out: list[int] = []
    for i, stream in enumerate(streams):
        out.extend(stream[i * block : (i + 1) * block])
    return tuple(out)


def _final_streams(seed: int) -> tuple[int, ...]:
    """Distribution-shifted families, reserved for the approved final run."""
    r = _rng(seed, 11)
    out: list[int] = []
    # Piecewise mixed blocks: boundaries and composition differ from the
    # development/validation construction.
    while len(out) < N:
        kind = r.randrange(4)
        width = r.randint(24, 240)
        if kind == 0:
            block = _walk(r, width, drift=r.choice((-5, 0, 5)), jump=r.randint(2, 10))
        elif kind == 1:
            value = r.randint(-1000, 1000)
            block = [_clip(value + r.randint(-3, 3))] * width
        elif kind == 2:
            block = [_clip(250 * ((j % r.randint(7, 41)) - 10) + (r.randint(-8000, 8000) if r.random() < 0.03 else 0)) for j in range(width)]
        else:
            block = [r.randint(-(1 << 31), (1 << 31) - 1) for _ in range(width)]
        out.extend(block)

    # A second, independent shift: drift with resets and geometric jumps.
    r = _rng(seed, 12)
    for i in range(N // 2, N):
        if r.random() < 0.025:
            out[i] = r.randint(-10_000, 10_000)
        elif r.random() < 0.20:
            out[i] = _clip(out[i - 1] + (1 if r.random() < 0.5 else -1) * (1 << min(20, int(r.expovariate(1 / 4)))))
        else:
            out[i] = _clip(out[i] + r.randint(-8, 8))

    # Alternating run/random regime in the first half.
    r = _rng(seed, 13)
    i = 0
    while i < N // 2:
        run = r.randint(5, 80)
        if (i // 80) % 2 == 0:
            value = r.randint(-500, 500)
            end = min(i + run, N // 2)
            out[i:end] = [_clip(value + r.randint(-1, 1))] * (end - i)
        else:
            end = min(i + run, N // 2)
            out[i:end] = [r.randint(-(1 << 31), (1 << 31) - 1) for _ in range(end - i)]
        i += run
    return tuple(out[:N])


def load(mode: Literal["development", "validation", "final"], seed: int, *, allow_final: bool = False) -> tuple[int, ...]:
    """Return one deterministic raw stream; no labels or family metadata."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if seed not in SEEDS[mode]:
        raise ValueError(f"seed {seed} is not preregistered for {mode}: {SEEDS[mode]}")
    if mode == "final":
        if not allow_final:
            raise PermissionError("final fixture requires explicit allow_final=True after parent approval")
        values = _final_streams(seed)
    else:
        values = _development_or_validation(seed)
    if len(values) != N or any(x < INT64_MIN or x > INT64_MAX for x in values):
        raise AssertionError("fixture violated signed-int64 contract")
    return values

