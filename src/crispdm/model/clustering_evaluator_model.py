# src/crispdm/model/clustering_evaluator_model.py
"""Pure model functions for CRISP-DM Phase 4.5 – Model Evaluation."""






from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (

    silhouette_score, davies_bouldin_score,
    calinski_harabasz_score, adjusted_rand_score,

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























    log.info("[evaluate_all_models] starting evaluation for %d technique(s)", len(techniques))

    results: dict[str, Any] = {}
    consolidated_metrics: dict[str, dict] = {}
    consolidated_profiles: dict[str, dict] = {}
    # NUEVO — almacena subsets exportados por variante
    cluster_subsets: dict[str, pd.DataFrame] = {}

    for technique_name, technique_cfg in techniques.items():
        if not technique_cfg.get("enabled", False):
            log.debug("[evaluate_all_models] technique '%s' disabled – skipping", technique_name)
            continue

        log.info("[evaluate_all_models] evaluating technique '%s'", technique_name)
        targets = technique_cfg.get("targets", [])
        params = technique_cfg.get("params", {})

        technique_seed = params.get("random_state", global_seed)

        if not targets:
            log.warning("[evaluate_all_models] technique '%s' has no targets – skipping", technique_name)
            continue

        technique_results = {}

        for target in targets:


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
                    log.warning("[evaluate_all_models] y_true not provided – skipping ARI for '%s'", target)
                    score = {"error": "ground_truth_not_provided"}
                else:
                    score = _compute_adjusted_rand_index(labels, y_true, params)
            elif technique_name == "cluster_profiling":
                model = best_models.get(target)
                # MODIFICADO — ahora retorna profile + subsets
                profile, subsets_df = _compute_cluster_profiling(
                    X_train, labels, model, params, seed=technique_seed
                )
                technique_results[target] = profile
                consolidated_profiles[target] = profile
                if subsets_df is not None:
                    cluster_subsets[target] = subsets_df
                    log.info("[evaluate_all_models] cluster subsets exported for '%s' – %d rows",
                             target, len(subsets_df))
                log.info("[evaluate_all_models] profiling completed for '%s' – %d clusters",
                         target, len(np.unique(labels)))
                continue
            else:
                log.warning("[evaluate_all_models] unknown technique '%s' – skipping", technique_name)
                continue

            technique_results[target] = score
            log.info("[evaluate_all_models] '%s' / '%s' – score: %s", technique_name, target, score)


        results[technique_name] = technique_results
        if technique_name != "cluster_profiling":
            consolidated_metrics[technique_name] = technique_results


    consolidated_summary = _build_consolidated_summary(consolidated_metrics)
    results["consolidated_summary"] = consolidated_summary
    results["consolidated_profiling"] = consolidated_profiles
    # NUEVO — subsets disponibles para el runner persistirlos
    results["cluster_subsets"] = cluster_subsets

    log.info("[evaluate_all_models] completed – %d technique(s) processed", len(results) - 3)
    return results


# -----------------------------------------------------------------------------
# Individual metric functions
# -----------------------------------------------------------------------------


def _compute_silhouette(X, labels, params, seed):

    sample_size = params.get("sample_size", None)
    metric = params.get("metric", "euclidean")

    if sample_size and X.shape[0] > sample_size:
        rng = np.random.RandomState(seed)
        idx = rng.choice(X.shape[0], sample_size, replace=False)
        X_sub, labels_sub = X[idx], labels[idx]

    else:

        X_sub, labels_sub = X, labels

    score = float(silhouette_score(X_sub, labels_sub, metric=metric))

    return {"score": score, "sample_size": X_sub.shape[0], "metric": metric, "random_seed": seed}


def _compute_davies_bouldin(X, labels, params):

    score = float(davies_bouldin_score(X, labels))

    return {"score": score, "lower_is_better": True}


def _compute_calinski_harabasz(X, labels, params):

    score = float(calinski_harabasz_score(X, labels))

    return {"score": score, "higher_is_better": True}


def _compute_adjusted_rand_index(labels_pred, y_true, params):

    score = float(adjusted_rand_score(y_true, labels_pred))

    return {"score": score, "metric": "adjusted_rand_index"}


def _compute_cluster_profiling(
        X: np.ndarray,
        labels: np.ndarray,
        model: Any,
        params: dict[str, Any],
        seed: int,
) -> tuple[dict, pd.DataFrame | None]:
    """Compute cluster profiling + optional subset export.














    Returns
    -------
    tuple[dict, pd.DataFrame | None]
        profile dict + DataFrame con subsets (columna cluster_id) o None si no se pide export
    """
    samples_per_cluster = params.get("samples_per_cluster", 100)
    top_features        = params.get("top_features", 10)
    export_subsets      = params.get("export_cluster_subsets", False)   # NUEVO
    subset_sample_size  = params.get("subset_sample_size", 5000)        # NUEVO

    unique_labels = np.unique(labels)
    profile: dict = {}
    subset_frames = []

    for cluster_id in unique_labels:
        mask = labels == cluster_id
        cluster_X = X[mask]
        cluster_size = cluster_X.shape[0]


        if model is not None and hasattr(model, "cluster_centers_"):
            centroid = model.cluster_centers_[int(cluster_id)]
        else:
            centroid = cluster_X.mean(axis=0)


        top_indices = np.argsort(np.abs(centroid))[::-1][:top_features]
        top_features_list = [
            {"feature_index": int(i), "value": float(centroid[i])}
            for i in top_indices
        ]

        profile[int(cluster_id)] = {
            "size": int(cluster_size),
            "percentage": float(cluster_size / X.shape[0] * 100),
            "centroid_summary": top_features_list,
            "samples_per_cluster": min(samples_per_cluster, cluster_size),
        }

        # NUEVO — exportar subset como DataFrame
        if export_subsets:
            n_export = min(subset_sample_size, cluster_size)
            rng = np.random.RandomState(seed)
            idx = rng.choice(cluster_size, n_export, replace=False)
            subset_df = pd.DataFrame(cluster_X[idx])
            subset_df["cluster_id"] = int(cluster_id)
            subset_frames.append(subset_df)
            log.debug("[profiling] cluster %d: exported %d rows", int(cluster_id), n_export)

    subsets_df = pd.concat(subset_frames, ignore_index=True) if subset_frames else None
    return profile, subsets_df


# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------


def _build_consolidated_summary(metrics: dict[str, dict]) -> dict:

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


    return summary