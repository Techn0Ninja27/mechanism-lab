"""Fail-closed readiness triage for grounded research results.

This is an allocation/diagnostic gate over stored structured assertions.  It
does not verify science, establish novelty, or authorize execution.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SCHEMA_VERSION = "review-readiness.v1"
STATUSES = {"incomplete", "needs_revision", "not_recommended",
            "source_only_diagnostic", "test_design_candidate", "not_selected"}
MECHANISM_FIELDS = ("status", "assumption_scope", "specialized_operation",
                    "remaining_effect", "simplest_baseline", "distinguishing_test")
PATH_FIELDS = ("input", "operation", "output", "possible_leakage")
EXPERIMENT_FIELDS = ("development", "heldout", "metric", "kill_threshold")
COVERAGE = {"covered", "partial", "not_found", "unknown"}
REVIEW_DECISIONS = {"test", "revise", "reject", "insufficient_evidence"}
FAILURE_TYPES = {"none_found", "algebraic_noop", "unobservable_input",
                 "unsupported_assumption", "prior_art_overlap", "weak_control",
                 "insufficient_evidence"}


def _text(value: Any, limit: int = 6000) -> bool:
    return isinstance(value, str) and bool(value.strip()) and len(value) <= limit


def _member(value: Any, values: set[str]) -> bool:
    return isinstance(value, str) and value in values


def _reason(code: str, detail: str) -> dict[str, str]:
    return {"code": code, "detail": detail[:1200]}


def _base(status: str, reasons: list[dict[str, str]], limitations: list[str]) -> dict[str, Any]:
    return {"schema": SCHEMA_VERSION, "status": status,
            "reasons": reasons[:24], "limitations": list(dict.fromkeys(limitations))[:24],
            "auto_execute": False, "novelty_established": False,
            "train_eligible": False}


def _claims_shape(proposal: Mapping[str, Any], review: Mapping[str, Any]) -> tuple[bool, list[dict[str, Any]], list[str]]:
    claims = proposal.get("contribution_claims")
    assessments = review.get("claim_assessments")
    if not isinstance(claims, list) or not 1 <= len(claims) <= 6:
        return False, [], ["proposal contribution_claims is missing or malformed"]
    ids: list[str] = []
    for claim in claims:
        if not isinstance(claim, Mapping) or not _text(claim.get("id"), 80) or not _text(claim.get("text"), 1200):
            return False, [], ["a contribution claim is malformed"]
        if (not _member(claim.get("role"), {"central", "supporting"})
                or not _member(claim.get("kind"), {"problem", "method", "resource", "expected_result"})):
            return False, [], [f"claim {claim['id']!r} has an invalid role or kind"]
        if claim["id"] in ids:
            return False, [], [f"duplicate contribution claim {claim['id']!r}"]
        ids.append(claim["id"])
    if not any(c.get("role") == "central" for c in claims):
        return False, [], ["at least one central contribution claim is required"]
    if not isinstance(assessments, list) or len(assessments) != len(ids):
        return False, [], ["claim assessments must cover every contribution claim exactly once"]
    by_id = {c["id"]: c for c in claims}
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for assessment in assessments:
        if not isinstance(assessment, Mapping):
            return False, [], ["a claim assessment is malformed"]
        cid = assessment.get("claim_id")
        if not isinstance(cid, str) or cid not in by_id or cid in seen:
            return False, [], ["claim assessments contain an unknown or duplicate claim ID"]
        if not _member(assessment.get("coverage"), COVERAGE):
            return False, [], [f"claim {cid!r} has invalid coverage"]
        checked = assessment.get("checked_chunk_ids")
        evidence = assessment.get("evidence")
        if not isinstance(checked, list) or len(checked) > 30 or any(not isinstance(x, str) for x in checked):
            return False, [], [f"claim {cid!r} has malformed checked_chunk_ids"]
        if len(set(checked)) != len(checked):
            return False, [], [f"claim {cid!r} repeats checked evidence IDs"]
        if not isinstance(evidence, list) or len(evidence) > 5:
            return False, [], [f"claim {cid!r} has malformed evidence"]
        for item in evidence:
            if not isinstance(item, Mapping) or not isinstance(item.get("chunk_id"), str) or item["chunk_id"] not in checked:
                return False, [], [f"claim {cid!r} has evidence outside checked_chunk_ids"]
            if not _text(item.get("quote"), 600) or not _text(item.get("supports"), 800):
                return False, [], [f"claim {cid!r} has malformed evidence text"]
        if assessment["coverage"] in {"covered", "partial"} and not evidence:
            return False, [], [f"claim {cid!r} reports overlap without evidence"]
        if assessment["coverage"] != "unknown" and not checked:
            return False, [], [f"claim {cid!r} reports coverage without checked evidence"]
        seen.add(cid)
        if not _text(assessment.get("reason"), 2400):
            return False, [], [f"claim {cid!r} has an invalid reason"]
        rows.append(dict(assessment, claim=by_id[cid]))
    return seen == set(ids), rows, []


def _proposal_shape(proposal: Any) -> tuple[bool, list[str]]:
    if not isinstance(proposal, Mapping):
        return False, ["proposal is missing or not an object"]
    reasons: list[str] = []
    for field in ("title", "hypothesis", "baseline", "operator_delta", "counterexample", "matched_control", "prior_art_difference"):
        if not _text(proposal.get(field)):
            reasons.append(f"proposal {field} is missing or empty")
    check = proposal.get("mechanism_check")
    if not isinstance(check, Mapping) or set(check) != set(MECHANISM_FIELDS):
        reasons.append("proposal mechanism_check is incomplete")
    elif not _member(check.get("status"), {"nontrivial", "redundant", "unresolved"}) or any(not _text(check.get(f), 3000) for f in MECHANISM_FIELDS[1:]):
        reasons.append("proposal mechanism_check has invalid fields")
    evidence = proposal.get("evidence")
    if not isinstance(evidence, list) or not 1 <= len(evidence) <= 12:
        reasons.append("proposal evidence must be a non-empty bounded list")
    else:
        for item in evidence:
            if (not isinstance(item, Mapping) or not isinstance(item.get("chunk_id"), str)
                    or not _text(item.get("quote"), 600) or not _text(item.get("supports"), 800)):
                reasons.append("proposal evidence has an incomplete entry")
                break
    paths = proposal.get("data_paths")
    if not isinstance(paths, list) or not 1 <= len(paths) <= 32:
        reasons.append("proposal data_paths is missing")
    else:
        for path in paths:
            if not isinstance(path, Mapping) or any(not _text(path.get(f), 2000) for f in PATH_FIELDS):
                reasons.append("proposal data_paths has an incomplete path")
                break
    experiment = proposal.get("experiment")
    if not isinstance(experiment, Mapping) or any(not _text(experiment.get(f), 3000) for f in EXPERIMENT_FIELDS):
        reasons.append("proposal experiment is incomplete")
    if not isinstance(proposal.get("assumptions"), list) or not 1 <= len(proposal["assumptions"]) <= 32 or any(not _text(x, 2000) for x in proposal["assumptions"]):
        reasons.append("proposal assumptions are missing or malformed")
    if not _member(proposal.get("decision"), {"investigate", "revise", "reject"}):
        reasons.append("proposal decision is invalid")
    if proposal.get("novelty_established") is not False:
        reasons.append("proposal novelty_established must remain false")
    return not reasons, reasons


def _review_shape(review: Any, proposal: Mapping[str, Any]) -> tuple[bool, list[str], list[dict[str, Any]]]:
    if not isinstance(review, Mapping):
        return False, ["review is missing or not an object"], []
    reasons: list[str] = []
    if not _member(review.get("decision"), REVIEW_DECISIONS):
        reasons.append("review decision is invalid")
    if not _member(review.get("failure_type"), FAILURE_TYPES):
        reasons.append("review failure_type is invalid")
    if not _text(review.get("reason")) or not _text(review.get("required_test")) or not _text(review.get("strongest_counterargument")):
        reasons.append("review explanation fields are incomplete")
    check = review.get("mechanism_check")
    if not isinstance(check, Mapping) or set(check) != set(MECHANISM_FIELDS):
        reasons.append("review mechanism_check is incomplete")
    elif not _member(check.get("status"), {"nontrivial", "redundant", "unresolved"}) or any(not _text(check.get(f), 3000) for f in MECHANISM_FIELDS[1:]):
        reasons.append("review mechanism_check has invalid fields")
    if review.get("novelty_established") is not False:
        reasons.append("review novelty_established must remain false")
    evidence = review.get("evidence")
    if not isinstance(evidence, list) or not 1 <= len(evidence) <= 8:
        reasons.append("review evidence must be a non-empty bounded list")
    else:
        for item in evidence:
            if (not isinstance(item, Mapping) or not isinstance(item.get("chunk_id"), str)
                    or not _text(item.get("quote"), 600) or not _text(item.get("supports"), 800)):
                reasons.append("review evidence has an incomplete entry")
                break
    ok, assessments, claim_reasons = _claims_shape(proposal, review)
    reasons.extend(claim_reasons)
    return not reasons and ok, reasons, assessments


def _evaluate_pair(proposal: Any, review: Any, retrieved_count: Any) -> dict[str, Any]:
    ok_p, p_reasons = _proposal_shape(proposal)
    ok_r, r_reasons, assessments = _review_shape(review, proposal if isinstance(proposal, Mapping) else {})
    if not ok_p or not ok_r:
        return _base("incomplete", [_reason("incomplete_record", x) for x in (p_reasons + r_reasons)],
                     ["Readiness is structural triage; it does not validate the research claim."])
    reasons: list[dict[str, str]] = []
    limitations = ["This is a model-review allocation signal, not a novelty, usefulness, feasibility or scientific-validity score.",
                   "Evidence spans, source fidelity and attached factual audits are not rechecked by this gate.",
                   "Retrieved-source count does not assess source relevance, recall or completeness.",
                   "Presence of metric and control text is not empirical validation."]
    for row in assessments:
        limitations.append(f"Claim {row['claim_id']} coverage is {row['coverage']}; this does not establish novelty.")
    pcheck = proposal["mechanism_check"]; rcheck = review["mechanism_check"]
    if proposal.get("decision") == "reject" or review.get("decision") == "reject" or pcheck.get("status") == "redundant" or rcheck.get("status") == "redundant":
        reasons.append(_reason("model_rejection_or_redundancy", "Stored proposal/review labels reject or redundant; this is preserved as model judgment."))
        return _base("not_recommended", reasons, limitations)
    if proposal.get("decision") == "revise" or review.get("decision") in {"revise", "insufficient_evidence"} or review.get("failure_type") != "none_found" or pcheck.get("status") == "unresolved" or rcheck.get("status") == "unresolved":
        if proposal.get("decision") == "revise": reasons.append(_reason("proposal_requests_revision", "The proposal itself is marked revise."))
        if review.get("decision") in {"revise", "insufficient_evidence"}: reasons.append(_reason("review_requests_revision", "The stored review is not a clean test recommendation."))
        if review.get("failure_type") != "none_found": reasons.append(_reason("review_failure_type", f"Stored failure_type is {review.get('failure_type')!r}."))
        if pcheck.get("status") == "unresolved" or rcheck.get("status") == "unresolved": reasons.append(_reason("mechanism_unresolved", "The mechanism check remains unresolved."))
        return _base("needs_revision", reasons, limitations)
    if not isinstance(retrieved_count, int) or isinstance(retrieved_count, bool) or retrieved_count <= 0:
        return _base("source_only_diagnostic", [_reason("no_external_retrieval", "No positive retrieved-source count is recorded; this is source-only diagnostic evidence.")], limitations)
    return _base("test_design_candidate", [_reason("clean_test_design_candidate", "Stored fields pass the structural gate and review requests a test design; no execution is authorized.")], limitations)


def evaluate_result(result: Any) -> dict[str, Any]:
    """Derive a conservative readiness record from raw grounded output."""
    if not isinstance(result, Mapping) or result.get("status") != "reviewed":
        return _base("incomplete", [_reason("run_not_reviewed", "Only reviewed grounded results can enter readiness triage.")], ["No experiment is authorized."])
    if "portfolio" in result and result.get("portfolio") is not None and not isinstance(result.get("portfolio"), Mapping):
        return _base("incomplete", [_reason("portfolio_shape", "A non-null portfolio must be an object.")], ["No candidate is authorized."])
    if isinstance(result.get("portfolio"), Mapping):
        portfolio = result["portfolio"]
        candidates = portfolio.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 3:
            return _base("incomplete", [_reason("portfolio_shape", "Portfolio must contain exactly three candidates.")], ["No candidate is authorized."])
        ids = [c.get("id") if isinstance(c, Mapping) else None for c in candidates]
        if (any(not isinstance(x, str) for x in ids) or len(set(ids)) != 3
                or set(ids) != {"cand-00", "cand-01", "cand-02"}):
            return _base("incomplete", [_reason("portfolio_ids", "Portfolio candidate IDs must be cand-00, cand-01 and cand-02 exactly.")], ["No candidate is authorized."])
        reviews = portfolio.get("review", {}).get("reviews") if isinstance(portfolio.get("review"), Mapping) else None
        review_ids = [x.get("candidate_id") if isinstance(x, Mapping) else None for x in reviews] if isinstance(reviews, list) else None
        if (not isinstance(reviews, list) or len(reviews) != 3
                or any(not isinstance(x, str) for x in review_ids or [])
                or len(set(review_ids or [])) != 3 or set(review_ids or []) != set(ids)
                or any(not isinstance(x, Mapping) or not isinstance(x.get("review"), Mapping) for x in reviews)):
            return _base("incomplete", [_reason("portfolio_reviews", "Every portfolio candidate needs exactly one raw review." )], ["No candidate is authorized."])
        by_id = {c["id"]: c for c in candidates}; review_by = {x["candidate_id"]: x["review"] for x in reviews}
        gates: dict[str, Any] = {}
        for cid in ids:
            gates[cid] = _evaluate_pair(by_id[cid].get("proposal"), review_by[cid], result.get("retrieved_source_count"))
        selected = portfolio.get("review", {}).get("selected_candidate_id")
        if selected is None:
            out = _base("not_selected", [_reason("no_selection", "Portfolio review selected no candidate.")], ["Candidate gates are diagnostics; no experiment is authorized."])
        elif not isinstance(selected, str) or selected not in gates:
            return _base("incomplete", [_reason("unknown_selection", "Portfolio selected_candidate_id is unknown.")], ["No candidate is authorized."])
        else:
            out = dict(gates[selected])
            out["reasons"] = [_reason("selected_candidate", f"Portfolio selected {selected}."), *out["reasons"]]
        out["candidate_gates"] = gates
        return out
    out = _evaluate_pair(result.get("proposal"), result.get("review"), result.get("retrieved_source_count"))
    return out

