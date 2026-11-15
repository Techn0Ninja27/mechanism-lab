"""Cheap preflight checks for proposed mechanism experiments.

These checks catch algebraic or observability failures before training.  They
are diagnostics, not a general proof checker and cannot establish novelty or
causal efficacy.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


def _finding(code: str, severity: str, unsupported: bool, required: str, evidence: str) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "unsupported_mechanism": unsupported,
        "test_required": required,
        "evidence": evidence,
    }


def check_feature_scaling_then_standardization(weights: Iterable[float]) -> dict[str, Any]:
    """Check whether constant feature scales survive a fitted standardizer.

    For positive ``c``, ``(cx-cμ)/(cσ) == (x-μ)/σ``.  A zero scale remains a
    feature mask.  Negative scales preserve magnitude but reverse orientation,
    so they are reported separately rather than incorrectly called invariant.
    """
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("weights must be a non-empty finite one-dimensional array")
    positive = np.flatnonzero(values > 0).tolist()
    negative = np.flatnonzero(values < 0).tolist()
    zero = np.flatnonzero(values == 0).tolist()
    findings = []
    if positive:
        findings.append(_finding("positive_scale_cancels", "warning", True, "unweighted standardized baseline", "For c>0, fitted standardization gives (cx-cμ)/(cσ)=(x-μ)/σ."))
    if negative:
        findings.append(_finding("negative_scale_reverses", "warning", True, "sign-aware ablation", "For c<0, standardized values equal -(x-μ)/σ; magnitude cancels but orientation changes."))
    if zero:
        findings.append(_finding("zero_scale_masks_feature", "info", False, "mask-preserving baseline", "For c=0, the feature is constant and its information is absent before standardization."))
    return {"check": "feature_scaling_then_standardization", "positive_scale_indices": positive,
            "negative_scale_indices": negative, "zero_scale_indices": zero,
            "findings": findings, "severity": "warning" if positive or negative else "info",
            "scope": "Continuous nonzero weight magnitudes do not survive fitted standardization; feature masking and sign changes are separate interventions.",
            "unsupported_mechanism": bool(positive or negative),
            "test_required": "mask-preserving baseline" if zero else "sign-aware ablation" if negative else "unweighted standardized baseline"}


def check_mask_cluster_capacity(mask_matrix: Any, k: int) -> dict[str, Any]:
    """Check whether a mask representation has at least ``k`` observed states."""
    matrix = np.asarray(mask_matrix)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("mask_matrix must be a non-empty two-dimensional array")
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer")
    unique = int(np.unique(matrix, axis=0).shape[0])
    insufficient = unique < int(k)
    finding = _finding("mask_capacity_below_k", "error", True, "increase mask diversity or lower k",
                       f"Observed {unique} unique mask rows but the mechanism requires {k}.") if insufficient else _finding(
                           "mask_capacity_meets_k", "info", False, "none", f"Observed {unique} unique mask rows for k={k}.")
    return {"check": "mask_cluster_capacity", "unique_mask_rows": unique, "required_k": int(k),
            "nonidentifiable": insufficient, "findings": [finding], "severity": finding["severity"],
            "unsupported_mechanism": insufficient, "test_required": finding["test_required"]}


def check_anchor_volatile_sets(anchors: Iterable[Any], volatile: Iterable[Any]) -> dict[str, Any]:
    """Check that anchor and volatile sets can support reconstruction."""
    anchor_set, volatile_set = set(anchors), set(volatile)
    overlap = anchor_set & volatile_set
    usable_targets = {target for target in volatile_set if anchor_set - {target}}
    impossible = not anchor_set or not volatile_set or not usable_targets
    if impossible:
        finding = _finding("reconstruction_inputs_absent", "error", True, "provide observed predictors distinct from the missing target", "No nonempty anchor-to-target prediction problem can be formed.")
    else:
        finding = _finding("reconstruction_structurally_possible", "warning" if overlap else "info", False, "held-out reconstruction test; exclude each target from its own predictors", "Disjoint anchor and target feature sets are valid. Their relationship and predictive information must still be established from data.")
    return {"check": "anchor_volatile_sets", "anchor_count": len(anchor_set), "volatile_count": len(volatile_set),
            "overlap_count": len(overlap), "usable_target_count": len(usable_targets),
            "reconstruction_possible": not impossible, "identifiability_established": False,
            "findings": [finding], "severity": finding["severity"], "unsupported_mechanism": impossible,
            "test_required": finding["test_required"]}


def check_mask_information_claim(*, explicitmaskremoved: bool, native_nan_routing: bool, mask_gated_selection: bool) -> dict[str, Any]:
    """Flag claims that a mask was removed when downstream paths still expose it."""
    carries = bool(native_nan_routing or mask_gated_selection)
    issue = bool(explicitmaskremoved and carries)
    finding = _finding("mask_representation_still_carries_information", "warning", True, "path ablation and permutation test",
                       "Native NaN routing or mask-gated selection remains an observable function of the missingness pattern.") if issue else _finding(
                           "mask_claim_not_contradicted", "info", False, "none", "No retained mask route was declared.")
    return {"check": "mask_information_claim", "explicit_mask_removed": bool(explicitmaskremoved),
            "native_nan_routing": bool(native_nan_routing), "mask_gated_selection": bool(mask_gated_selection),
            "mask_information_preserved": carries, "actual_prediction_dependence_established": False, "findings": [finding], "severity": finding["severity"],
            "unsupported_mechanism": issue, "test_required": finding["test_required"]}

