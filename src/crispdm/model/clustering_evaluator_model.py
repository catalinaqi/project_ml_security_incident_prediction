# src/crispdm/model/clustering_evaluator_model.py
"""Pure model functions for CRISP-DM Phase 4.5 – Model Evaluation.

Computes evaluation metrics for clustering models.
No side effects except logging for traceability.
All numeric parameters come from YAML config; fallback to 42 if missing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
)

from crispdm.common.logging_adapter_common import get_logger

log = get_logger(__name__)


def evaluate_all_models(
    techniques: dict[str, Any],
    X_train: np.ndarray,
    cluster_labels: dict[str, np.ndarray],
    best_models: dict[str, Any],
    y_true: np.ndarray | None = None,
    global_seed: int = 42,
) -> dict[str, Any]:
    """Evaluate all clustering models using configured techniques.

    Parameters
    ----------
    techniques : dict
        Configuration dictionary from ``methods.model_evaluation.techniques``.
    X_train : np.ndarray
        Feature matrix (n_samples, n_features).
    cluster_labels : dict
        Mapping from model_name (e.g. ``"kmeans_n2"``) to 1D array of cluster assignments.
    best_models : dict
        Mapping from model_name to trained model object (e.g. KMeans).
    y_true : np.ndarray or None
        Ground truth labels (``IncidentGrade``) for Adjusted Rand Index.
    global_seed : int
        Default random seed if not specified in technique params.

    Returns
    -------
    dict
        Nested results: ``{technique_name: {target: metric_dict}, ...}``
        plus ``consolidated_summary``, ``consolidated_profiling``.
    """
    log.info("[evaluate_all_models] starting evaluation for %d technique(s)", len(techniques))

    results: dict[str, Any] = {}
    consolidated_metrics: dict[str, dict] = {}
    consolidated_profiles: dict[str, dict] = {}

    for technique_name, technique_cfg in techniques.items():
        if not technique_cfg.get("enabled", False):
            log.debug("[evaluate_all_models] technique '%s' disabled – skipping", technique_name)
            continue

        log.info("[evaluate_all_models] evaluating technique '%s'", technique_name)
        targets = technique_cfg.get("targets", [])
        params = technique_cfg.get("params", {})
        # Use random_state from params if provided, else fallback to global_seed
        technique_seed = params.get("random_state", global_seed)

        if not targets:
            log.warning("[evaluate_all_models] technique '%s' has no targets defined – skipping", technique_name)
            continue

        technique_results = {}

        for target in targets:
            log.debug("[evaluate_all_models] technique '%s', target='%s'", technique_name, target)

            labels = cluster_labels.get(target)
            if labels is None:
                log.error("[evaluate_all_models] target '%s' not found in cluster_labels – skipping", target)
                continue

            if technique_name == "silhouette":
                score = _compute_silhouette(X_train, labels, params, seed=technique_seed)
            elif technique_name == "davies_bouldin":
                score = _compute_davies_bouldin(X_train, labels, params)
            elif technique_name == "calinski_harabasz":
                score = _compute_calinski_harabasz(X_train, labels, params)
            elif technique_name == "adjusted_rand_index":
                if y_true is None:
                    log.warning("[evaluate_all_models] y_true not provided – cannot compute Adjusted Rand Index for '%s'", target)
                    score = {"error": "ground_truth_not_provided"}
                else:
                    score = _compute_adjusted_rand_index(labels, y_true, params)
            elif technique_name == "cluster_profiling":
                model = best_models.get(target)
                profile = _compute_cluster_profiling(X_train, labels, model, params, seed=technique_seed)
                technique_results[target] = profile
                consolidated_profiles[target] = profile
                log.info("[evaluate_all_models] profiling completed for '%s' – %d clusters", target, len(np.unique(labels)))
                continue
            else:
                log.warning("[evaluate_all_models] unknown technique '%s' – skipping", technique_name)
                continue

            technique_results[target] = score
            log.info("[evaluate_all_models] technique '%s', target='%s' – score computed: %s",
                     technique_name, target, score)

        results[technique_name] = technique_results
        if technique_name != "cluster_profiling":
            consolidated_metrics[technique_name] = technique_results

    # Build consolidated summary
    consolidated_summary = _build_consolidated_summary(consolidated_metrics)
    results["consolidated_summary"] = consolidated_summary
    results["consolidated_profiling"] = consolidated_profiles

    log.info("[evaluate_all_models] evaluation completed – %d technique(s) processed", len(results) - 2)
    return results


# -----------------------------------------------------------------------------
# Individual metric functions
# -----------------------------------------------------------------------------


def _compute_silhouette(X: np.ndarray, labels: np.ndarray, params: dict[str, Any], seed: int) -> dict:
    """Compute silhouette score with optional subsample."""
    sample_size = params.get("sample_size", None)
    metric = params.get("metric", "euclidean")

    if sample_size and X.shape[0] > sample_size:
        rng = np.random.RandomState(seed)
        idx = rng.choice(X.shape[0], sample_size, replace=False)
        X_sub = X[idx]
        labels_sub = labels[idx]
    else:
        X_sub = X
        labels_sub = labels

    score = float(silhouette_score(X_sub, labels_sub, metric=metric))
    log.debug("[_compute_silhouette] silhouette=%.4f (sample=%d, seed=%d)", score, X_sub.shape[0], seed)
    return {"score": score, "sample_size": X_sub.shape[0], "metric": metric, "random_seed": seed}


def _compute_davies_bouldin(X: np.ndarray, labels: np.ndarray, params: dict[str, Any]) -> dict:
    """Compute Davies-Bouldin index (lower is better)."""
    score = float(davies_bouldin_score(X, labels))
    log.debug("[_compute_davies_bouldin] davies_bouldin=%.4f", score)
    return {"score": score, "lower_is_better": True}


def _compute_calinski_harabasz(X: np.ndarray, labels: np.ndarray, params: dict[str, Any]) -> dict:
    """Compute Calinski-Harabasz index (higher is better)."""
    score = float(calinski_harabasz_score(X, labels))
    log.debug("[_compute_calinski_harabasz] calinski_harabasz=%.4f", score)
    return {"score": score, "higher_is_better": True}


def _compute_adjusted_rand_index(labels_pred: np.ndarray, y_true: np.ndarray, params: dict[str, Any]) -> dict:
    """Compute Adjusted Rand Index between predicted clusters and ground truth."""
    score = float(adjusted_rand_score(y_true, labels_pred))
    log.debug("[_compute_adjusted_rand_index] ARI=%.4f", score)
    return {"score": score, "metric": "adjusted_rand_index"}


def _compute_cluster_profiling(
    X: np.ndarray,
    labels: np.ndarray,
    model: Any,
    params: dict[str, Any],
    seed: int,
) -> dict:
    """Compute cluster profiling: per-cluster statistics and top features.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    labels : np.ndarray
        Cluster assignments.
    model : Any
        Trained clustering model (e.g., KMeans). Can be None.
    params : dict
        Configuration parameters (samples_per_cluster, top_features, etc.).
    seed : int
        Random seed for reproducibility (if needed in future).

    Returns
    -------
    dict
        Profile with per-cluster statistics.
    """
    samples_per_cluster = params.get("samples_per_cluster", 100)
    top_features = params.get("top_features", 10)

    unique_labels = np.unique(labels)
    profile: dict = {}
    for cluster_id in unique_labels:
        mask = labels == cluster_id
        cluster_X = X[mask]
        cluster_size = cluster_X.shape[0]

        # Centroid (model centroids if available, else empirical mean)
        if model is not None and hasattr(model, "cluster_centers_"):
            centroid = model.cluster_centers_[int(cluster_id)]
        else:
            centroid = cluster_X.mean(axis=0)

        # Top features by absolute value from centroid
        top_indices = np.argsort(np.abs(centroid))[::-1][:top_features]
        top_features_list = [
            {"feature_index": int(idx), "value": float(centroid[idx])}
            for idx in top_indices
        ]

        profile[int(cluster_id)] = {
            "size": int(cluster_size),
            "percentage": float(cluster_size / X.shape[0] * 100),
            "centroid_summary": top_features_list,
            "samples_per_cluster": min(samples_per_cluster, cluster_size),
        }
        log.debug("[_compute_cluster_profiling] cluster %d: size=%d, top features computed",
                  int(cluster_id), cluster_size)

    return profile


# -----------------------------------------------------------------------------
# Helper: build consolidated summary
# -----------------------------------------------------------------------------


def _build_consolidated_summary(metrics: dict[str, dict]) -> dict:
    """Build a consolidated summary comparing metrics across models."""
    all_targets = set()
    for tech_results in metrics.values():
        all_targets.update(tech_results.keys())

    summary: dict = {}
    for target in sorted(all_targets):
        target_summary = {}
        for tech_name, tech_results in metrics.items():
            if target in tech_results:
                score_val = tech_results[target].get("score")
                if score_val is not None:
                    target_summary[tech_name] = score_val
        summary[target] = target_summary
    log.debug("[_build_consolidated_summary] built summary for %d target(s): %s",
              len(summary), list(summary.keys()))
    return summary