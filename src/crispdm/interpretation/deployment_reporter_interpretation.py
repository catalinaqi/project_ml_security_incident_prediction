# src/crispdm/interpretation/deployment_reporter_interpretation.py
"""Deployment readiness assessment and recommendations – Phase 5.4.

Provides functions to evaluate whether a model is ready for deployment
based on business criteria, and to generate actionable recommendations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from crispdm.common.logging_adapter_common import get_logger

log = get_logger(__name__)


def evaluate_deployment_readiness(
    evaluation_summary: Dict[str, Any],
    ari_consolidated: Dict[str, Any],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate deployment readiness based on evaluation metrics and business criteria.

    Parameters
    ----------
    evaluation_summary : Dict[str, Any]
        Consolidated evaluation summary from Phase 4.5.
        Expected structure::

            {
                "best_model": "kmeans_n2",
                "silhouette": { "kmeans_n2": 0.35, "kmeans_n3": 0.34 },
                "davies_bouldin": { "kmeans_n2": 1.2, "kmeans_n3": 1.3 },
                "calinski_harabasz": { "kmeans_n2": 1500, "kmeans_n3": 1400 },
                "models": ["kmeans_n2", "kmeans_n3"]
            }

    ari_consolidated : Dict[str, Any]
        Consolidated ARI scores from ground truth comparison.
        Expected structure::

            {
                "kmeans_n2": { "ari": 0.08 },
                "kmeans_n3": { "ari": 0.12 }
            }

    params : Dict[str, Any]
        Parameters from YAML configuration:
        - ``min_ari_threshold`` : float (minimum ARI to consider alignment)
        - ``winning_model_criteria`` : str (e.g. ``"ari_score"``)
        - ``document_limitations`` : bool
        - ``test_set_status`` : str (documentation note)

    Returns
    -------
    Dict[str, Any]
        Readiness report with keys:
        - ``winning_model`` : str
        - ``winning_criteria`` : str
        - ``ari_scores`` : dict
        - ``min_ari_threshold`` : float
        - ``threshold_met`` : bool
        - ``best_silhouette`` : float or None
        - ``best_davies_bouldin`` : float or None
        - ``best_calinski_harabasz`` : float or None
        - ``document_limitations`` : bool
        - ``test_set_status`` : str
        - ``is_ready_for_deployment`` : bool (True if threshold met and at least one model has positive ARI)
        - ``deployment_barriers`` : list of strings
    """
    min_ari = params.get("min_ari_threshold", 0.05)
    winning_criteria = params.get("winning_model_criteria", "ari_score")

    # --- Extract ARI scores ---
    ari_scores: Dict[str, float] = {}
    for model_key, data in ari_consolidated.items():
        if isinstance(data, dict) and "ari" in data:
            ari_scores[model_key] = data["ari"]
        elif isinstance(data, (int, float)):
            ari_scores[model_key] = data

    if not ari_scores:
        log.warning("[deployment_readiness] No ARI scores found – using default 0")
        ari_scores["unknown"] = 0.0

    # --- Determine winning model based on selected criteria ---
    if winning_criteria == "ari_score":
        # Higher is better
        winning_model = max(ari_scores, key=ari_scores.get)
        winning_ari = ari_scores[winning_model]
    else:
        # Fallback: use first model
        winning_model = list(ari_scores.keys())[0]
        winning_ari = ari_scores[winning_model]
        log.warning("[deployment_readiness] Unrecognised winning_model_criteria '%s' – using first model", winning_criteria)

    threshold_met = winning_ari >= min_ari

    # --- Extract other metrics from evaluation_summary ---
    best_silhouette = None
    best_davies_bouldin = None
    best_calinski_harabasz = None

    if evaluation_summary:
        # Try different possible keys (depends on how summary is built)
        for metric_key in ["silhouette", "silhouette_score"]:
            if metric_key in evaluation_summary:
                sil_dict = evaluation_summary[metric_key]
                if isinstance(sil_dict, dict):
                    best_silhouette = max(sil_dict.values()) if sil_dict else None
                break
        for metric_key in ["davies_bouldin", "davies_bouldin_score"]:
            if metric_key in evaluation_summary:
                db_dict = evaluation_summary[metric_key]
                if isinstance(db_dict, dict):
                    best_davies_bouldin = min(db_dict.values()) if db_dict else None  # lower is better
                break
        for metric_key in ["calinski_harabasz", "calinski_harabasz_score"]:
            if metric_key in evaluation_summary:
                ch_dict = evaluation_summary[metric_key]
                if isinstance(ch_dict, dict):
                    best_calinski_harabasz = max(ch_dict.values()) if ch_dict else None
                break

    # --- Compile deployment barriers ---
    barriers: List[str] = []
    if not threshold_met:
        barriers.append(
            f"ARI threshold not met: {winning_ari:.3f} < {min_ari:.3f}. "
            "Clusters show weak alignment with known ground truth."
        )
    if best_silhouette is not None and best_silhouette < 0.25:
        barriers.append(
            f"Low silhouette score ({best_silhouette:.3f} < 0.25) "
            "indicates poor cluster compactness."
        )
    if best_davies_bouldin is not None and best_davies_bouldin > 1.5:
        barriers.append(
            f"High Davies-Bouldin index ({best_davies_bouldin:.3f} > 1.5) "
            "suggests overlapping clusters."
        )
    if best_calinski_harabasz is not None and best_calinski_harabasz < 500:
        barriers.append(
            f"Low Calinski-Harabasz score ({best_calinski_harabasz:.1f} < 500) "
            "indicates weak cluster separation."
        )

    is_ready = threshold_met and len(barriers) == 0

    report = {
        "winning_model": winning_model,
        "winning_criteria": winning_criteria,
        "winning_ari": winning_ari,
        "ari_scores": ari_scores,
        "min_ari_threshold": min_ari,
        "threshold_met": threshold_met,
        "best_silhouette": best_silhouette,
        "best_davies_bouldin": best_davies_bouldin,
        "best_calinski_harabasz": best_calinski_harabasz,
        "document_limitations": params.get("document_limitations", True),
        "test_set_status": params.get("test_set_status", "N/A"),
        "is_ready_for_deployment": is_ready,
        "deployment_barriers": barriers,
        "summary": (
            "Model is ready for deployment." if is_ready
            else f"Model not ready. Barriers: {'; '.join(barriers)}"
        ),
    }

    log.info("[deployment_readiness] Winning model: %s | ARI=%.4f | threshold_met=%s | ready=%s",
             winning_model, winning_ari, threshold_met, is_ready)
    return report


def generate_recommendations(
    readiness_metrics: Dict[str, Any],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate deployment recommendations based on readiness assessment.

    Parameters
    ----------
    readiness_metrics : Dict[str, Any]
        Output of :func:`evaluate_deployment_readiness`.
    params : Dict[str, Any]
        Parameters from YAML:
        - ``include_deployment_considerations`` : bool
        - ``include_monitoring_suggestions`` : bool

    Returns
    -------
    Dict[str, Any]
        Recommendations with keys:
        - ``overall_recommendation`` : str
        - ``deployment_considerations`` : list of strings (if enabled)
        - ``monitoring_suggestions`` : list of strings (if enabled)
        - ``limitations_acknowledged`` : bool
        - ``test_set_status`` : str
    """
    include_deploy = params.get("include_deployment_considerations", True)
    include_monitor = params.get("include_monitoring_suggestions", True)

    is_ready = readiness_metrics.get("is_ready_for_deployment", False)
    winning_model = readiness_metrics.get("winning_model", "unknown")
    barriers = readiness_metrics.get("deployment_barriers", [])

    if is_ready:
        overall = (
            f"Proceed with deployment of {winning_model}. "
            "Clusters show acceptable alignment with known threat categories. "
            "Proceed to supervised deployment phase."
        )
    else:
        overall = (
            f"Deployment of {winning_model} is NOT recommended. "
            "Address the following barriers before proceeding: "
            + "; ".join(barriers)
        )

    recommendations: Dict[str, Any] = {
        "overall_recommendation": overall,
        "winning_model": winning_model,
        "deployment_considerations": [],
        "monitoring_suggestions": [],
        "limitations_acknowledged": readiness_metrics.get("document_limitations", False),
        "test_set_status": readiness_metrics.get("test_set_status", "N/A"),
    }

    if include_deploy:
        recommendations["deployment_considerations"] = [
            "If deploying, ensure cluster labels are mapped to human-readable categories.",
            "Consider using the winning model as a triage pre-filter before supervised ML.",
            f"Test set ('{readiness_metrics.get('test_set_status', '')}') is not used in unsupervised clustering; reserve for supervised model validation.",
            "Document cluster characterisation rules (top-10 features) for SOC analysts.",
            "Set up version control for the pipeline configuration and transformer serialisation.",
        ]

    if include_monitor:
        recommendations["monitoring_suggestions"] = [
            "Track cluster distribution over time for drift detection.",
            "Alert if any cluster size drops below 1% of total assignments.",
            "Periodically recompute ARI with new ground truth to validate alignment.",
            "Monitor silhouette score on incoming data to detect feature drift.",
            "If ARI falls below threshold, retrain or re-evaluate the model.",
        ]

    log.info("[recommendations] Generated %d deploy considerations, %d monitoring suggestions",
             len(recommendations.get("deployment_considerations", [])),
             len(recommendations.get("monitoring_suggestions", [])))
    return recommendations