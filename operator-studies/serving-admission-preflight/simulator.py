"""Small deterministic admission-state simulator for the reviewed proposal.

This is a preflight of accounting and ordering semantics, not a serving
implementation.  Output lengths are used only by the simulated EOS oracle;
the policy receives observed tokens and the certified lower bound.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class JobSpec:
    id: int
    arrival: int
    prompt: int
    prefill_slots: int
    true_output: int
    lower_bound: int = 1


@dataclass(frozen=True)
class PolicySpec:
    id: int
    arrival: int
    prompt: int
    prefill_slots: int


@dataclass
class Job:
    spec: PolicySpec
    state: str = "future"
    observed: int = 0
    lower: int = 1
    prefill_done: int | None = None
    completion: int | None = None
    restarts: int = 0
    events: list[str] = field(default_factory=list)


ORDERINGS = ("fifo", "largest_prompt", "smallest_reservation", "smallest_prefill")


def _usage_prefill(jobs):
    return sum(j.spec.prompt for j in jobs if j.state in {"prefill", "waiting"})


def _usage_decode(jobs, projected=False):
    return sum(j.spec.prompt + max(j.observed + (1 if projected else 0), j.lower) for j in jobs if j.state == "decode")


def _order(waiting, policy):
    if policy == "fifo":
        return sorted(waiting, key=lambda j: (j.spec.arrival, j.spec.id))
    if policy == "largest_prompt":
        return sorted(waiting, key=lambda j: (-j.spec.prompt, j.spec.arrival, j.spec.id))
    if policy == "smallest_reservation":
        return sorted(waiting, key=lambda j: (j.spec.prompt + j.lower, j.spec.arrival, j.spec.id))
    if policy == "smallest_prefill":
        return sorted(waiting, key=lambda j: (j.spec.prefill_slots, j.spec.arrival, j.spec.id))
    raise ValueError(f"unknown ordering {policy}")


def simulate(specs, *, prefill_capacity, decode_capacity, policy, horizon=100):
    if policy not in ORDERINGS or prefill_capacity <= 0 or decode_capacity <= 0:
        raise ValueError("invalid simulator configuration")
    if not specs or any(s.true_output <= 0 or s.prompt <= 0 or s.prefill_slots <= 0 or s.lower_bound <= 0 for s in specs):
        raise ValueError("positive job fields required")
    if len({s.id for s in specs}) != len(specs) or any(s.lower_bound > s.true_output for s in specs):
        raise ValueError("unique ids and valid lower bounds required")
    # Feasibility is a fixture-oracle check, not information exposed to policy.
    if any(s.prompt > prefill_capacity or s.prompt + s.true_output > decode_capacity for s in specs):
        raise ValueError("Oracle fixture contains an individually infeasible request")
    oracle = {s.id: s.true_output for s in specs}
    jobs = {s.id: Job(PolicySpec(s.id, s.arrival, s.prompt, s.prefill_slots), lower=s.lower_bound) for s in specs}
    trace = []
    for t in range(horizon):
        for j in jobs.values():
            if j.state == "future" and j.spec.arrival <= t:
                j.state = "unassigned"; j.events.append(f"{t}:arrive")
        # Prefill completions enter the waiting set with their prompt KV retained.
        for j in jobs.values():
            if j.state == "prefill" and j.prefill_done == t:
                j.state = "waiting"; j.events.append(f"{t}:prefill_done")
        # Decode growth happens only after admissions. Evict until the projected
        # next-token accounting is feasible, using the proposal's lower-bound key.
        while _usage_decode(list(jobs.values()), projected=True) > decode_capacity:
            active = [j for j in jobs.values() if j.state == "decode"]
            if not active:
                raise RuntimeError("decode capacity cannot admit projected state")
            victim = min(active, key=lambda j: (j.lower, j.spec.arrival, j.spec.id))
            victim.lower = max(victim.lower, victim.observed)
            victim.observed = 0
            victim.state = "unassigned"
            victim.prefill_done = None
            victim.restarts += 1
            victim.events.append(f"{t}:evict_restart:lower={victim.lower}")
        waiting = [j for j in jobs.values() if j.state == "waiting"]
        for j in _order(waiting, policy):
            if _usage_decode(list(jobs.values()), projected=True) + j.spec.prompt + max(1, j.lower) <= decode_capacity:
                j.state = "decode"; j.events.append(f"{t}:decode_admit")
        # Admit all feasible prefills in arrival order. A prefill occupies prompt
        # memory for its duration and then remains in waiting memory.
        for j in sorted((x for x in jobs.values() if x.state == "unassigned"), key=lambda x: (x.spec.arrival, x.spec.id)):
            if _usage_prefill(list(jobs.values())) + j.spec.prompt <= prefill_capacity:
                j.state = "prefill"; j.prefill_done = t + j.spec.prefill_slots
                j.events.append(f"{t}:prefill_admit")
        # One token is generated for each admitted decode request. EOS is the
        # only access to true_output, and occurs after this slot's generation.
        for j in list(jobs.values()):
            if j.state == "decode":
                j.observed += 1
        # Check the allocation peak before EOS releases memory. Checking only
        # after completions could hide an illegal transient over-allocation.
        decode_peak = _usage_decode(list(jobs.values()))
        if decode_peak > decode_capacity:
            raise AssertionError(f"decode growth peak exceeds capacity: {decode_peak}")
        for j in jobs.values():
            if j.state == "decode" and j.observed >= oracle[j.spec.id]:
                j.state = "complete"; j.completion = t + 1; j.events.append(f"{t+1}:complete")
        pre = _usage_prefill(list(jobs.values())); dec = _usage_decode(list(jobs.values()), projected=False)
        if pre > prefill_capacity or dec > decode_capacity:
            raise AssertionError(f"capacity invariant failed at t={t}: {pre},{dec}")
        trace.append({"t": t, "prefill_usage": pre, "decode_usage": dec, "decode_peak_before_eos": decode_peak,
                      "states": {str(k): v.state for k, v in sorted(jobs.items())}})
        if all(j.state == "complete" for j in jobs.values()):
            break
    if any(j.state != "complete" for j in jobs.values()):
        raise RuntimeError("horizon exhausted; increase horizon or inspect infeasible case")
    completion = {j.spec.id: j.completion for j in jobs.values()}
    return {"policy": policy, "completion_times": completion,
            "sum_completion_time": sum(completion.values()),
            "sum_flow_time": sum(completion[j.spec.id] - j.spec.arrival for j in jobs.values()),
            "restarts": sum(j.restarts for j in jobs.values()), "trace": trace,
            "jobs": {str(j.spec.id): {"events": j.events, "restarts": j.restarts, "lower_final": j.lower} for j in jobs.values()}}
