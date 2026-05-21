# src/crispdm/interpretation/pipeline_auditor_interpretation.py
"""Process audit for CRISP-DM Phase 5.3.

Provides functions to check for feature leakage risks (e.g., post‑triage
features) and to verify reproducibility of key artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib
import json

from crispdm.common.logging_adapter_common import get_logger

log = get_logger(__name__)


def check_leakage_sanity(
    checked_features: List[str],
    leakage_risk: str = "suspect",
) -> Dict[str, Any]:
    """Perform a leakage sanity check on a list of feature names.

    This function validates that each feature in ``checked_features``
    is indeed a known leakage‑risk feature. In a production pipeline,
    this would compare against a metadata registry; here we apply a
    rule‑based check (e.g., known post‑triage columns).

    Parameters
    ----------
    checked_features : List[str]
        Feature names to check (e.g., ``["SuspicionLevel", "LastVerdict"]``).
    leakage_risk : str, optional
        Risk level expected (default ``"suspect"``). Currently used only
        for documentation.

    Returns
    -------
    Dict[str, Any]
        A report with keys:
        - ``"checked_features"`` : list of feature names examined
        - ``"leakage_risk_label"`` : the risk label provided
        - ``"results"`` : list of dicts per feature with keys:
            ``"feature"``, ``"is_known_leakage"``, ``"reason"``
        - ``"all_passed"`` : bool (True if all features are known leakage)
        - ``"unexpected_features"`` : list of features that are NOT known leakage
    """
    # Hard‑coded set of known leakage‑risk features (post‑triage or target‑related)
    # This could be extended or loaded from a configuration file.
    KNOWN_LEAKAGE_FEATURES: set = {
        "SuspicionLevel",
        "LastVerdict",
        "IncidentGrade",
        "ActionGrouped",
        "ActionGranular",
        "ThreatFamily",
        "MitreTechniques",
    }

    results: List[Dict[str, Any]] = []
    unexpected: List[str] = []

    for feature in checked_features:
        is_known = feature in KNOWN_LEAKAGE_FEATURES
        reason = (
            f"Feature '{feature}' is recognised as a known leakage‑risk column "
            f"(e.g., post‑triage or target‑related)."
            if is_known
            else (
                f"Feature '{feature}' is NOT in the known leakage set. "
                "Please verify its origin."
            )
        )
        results.append({
            "feature": feature,
            "is_known_leakage": is_known,
            "reason": reason,
        })
        if not is_known:
            unexpected.append(feature)

    all_passed = len(unexpected) == 0

    # If leakage_risk is "suspect", we consider the check passed only if
    # all checked features are indeed known leakage features.
    # The risk label is recorded for documentation.
    report = {
        "checked_features": checked_features,
        "leakage_risk_label": leakage_risk,
        "results": results,
        "all_passed": all_passed,
        "unexpected_features": unexpected,
        "summary": (
            "Leakage sanity check passed – all checked features are known leakage risks."
            if all_passed
            else f"Leakage sanity check WARNING – features not in known leakage set: {unexpected}"
        ),
    }

    log.info("[leakage_sanity] checked %d features – all_passed=%s",
             len(checked_features), all_passed)
    return report


def check_reproducibility(
    artifacts_to_verify: List[str],
    target_seed: Optional[int] = None,
    run_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Verify reproducibility by checking existence (and optionally seed) of artifacts.

    Parameters
    ----------
    artifacts_to_verify : List[str]
        List of relative paths to artifacts (e.g., training data, model pickle,
        evaluation summary).
    target_seed : int, optional
        Expected random seed value (e.g., 7). If provided, the function will
        include a note about the seed in the report.
    run_dir : Path, optional
        Base directory for resolving artifact paths. If provided, existence
        checks will be performed.

    Returns
    -------
    Dict[str, Any]
        A report with keys:
        - ``"target_seed"`` : the seed (or None)
        - ``"artifacts_checked"`` : list of artifact paths checked
        - ``"results"`` : list of dicts per artifact with keys:
            ``"path"``, ``"exists"``, ``"size_bytes"``
        - ``"all_exist"`` : bool
        - ``"missing_artifacts"`` : list of missing paths
        - ``"summary"`` : string
    """
    results: List[Dict[str, Any]] = []
    missing: List[str] = []

    for artifact_rel_path in artifacts_to_verify:
        exists = False
        size = None
        if run_dir is not None:
            full_path = run_dir / artifact_rel_path
            if full_path.exists():
                exists = True
                size = full_path.stat().st_size
            else:
                missing.append(artifact_rel_path)
        else:
            # If no run_dir, we can only note the path
            log.info("[reproducibility] no run_dir provided – skipping existence check for %s",
                     artifact_rel_path)

        results.append({
            "path": artifact_rel_path,
            "exists": exists,
            "size_bytes": size,
        })

    all_exist = len(missing) == 0

    summary = (
        "Reproducibility check passed – all required artifacts exist."
        if all_exist and run_dir is not None
        else (
            f"Reproducibility check WAITING – no base directory provided."
            if run_dir is None
            else f"Reproducibility check WARNING – missing artifacts: {missing}"
        )
    )

    report = {
        "target_seed": target_seed,
        "artifacts_checked": artifacts_to_verify,
        "results": results,
        "all_exist": all_exist,
        "missing_artifacts": missing,
        "summary": summary,
    }

    log.info("[reproducibility] checked %d artifacts – all_exist=%s (seed=%s)",
             len(artifacts_to_verify), all_exist, target_seed)
    return report