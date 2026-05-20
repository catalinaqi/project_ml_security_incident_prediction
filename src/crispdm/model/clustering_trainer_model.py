# src/crispdm/model/clustering_trainer_model.py
"""CRISP-DM Phase 4.3 – Model Training with hyperparameter tuning.

This module contains a pure function that trains clustering models
(KMeans, DBSCAN) according to configuration.  No file I/O or side effects.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from crispdm.common.logging_adapter_common import get_logger
from crispdm.configuration.model_registry_config import ModelRegistry, get_model_class

log = get_logger(__name__)

# Map of scoring metric names to sklearn functions
_SCORING_FUNCTIONS = {
    "silhouette": silhouette_score,
    "davies_bouldin": davies_bouldin_score,
    "calinski_harabasz": calinski_harabasz_score,
}


def _score_clustering(
    X: np.ndarray,
    labels: np.ndarray,
    metric_name: str,
) -> float:
    """Compute a clustering score using the requested metric.

    Parameters
    ----------
    X : np.ndarray
        Data matrix.
    labels : np.ndarray
        Cluster labels (may contain noise label -1 for DBSCAN).
    metric_name : str
        One of ``"silhouette"``, ``"davies_bouldin"``, ``"calinski_harabasz"``.

    Returns
    -------
    float
        The computed score.  Returns ``-1.0`` if evaluation fails (e.g. only one cluster).
    """
    # DBSCAN can assign only noise (-1).  Score only non-noise points.
    unique_labels = set(labels)
    # Exclude noise (-1) for DBSCAN? Better use all points, but silhouette may fail with single cluster.
    n_clusters = len(unique_labels - {-1})
    if n_clusters < 2:
        log.warning("[trainer] only %d cluster(s) found – cannot compute %s", n_clusters, metric_name)
        return -1.0

    scorer = _SCORING_FUNCTIONS.get(metric_name)
    if scorer is None:
        log.error("[trainer] unknown scoring metric '%s' – falling back to silhouette", metric_name)
        scorer = silhouette_score

    try:
        score = float(scorer(X, labels))
        log.debug("[trainer] %s = %.6f", metric_name, score)
        return score
    except Exception as e:
        log.warning("[trainer] scoring failed: %s – returning -1.0", e)
        return -1.0


def train_clustering_models(
    X_train: np.ndarray,
    tuning_cfg: dict[str, Any],
    fit_cfg: dict[str, Any],
    problem_type: str = "clustering",
    *,
    scoring: Optional[str] = None,
    grid_search_sample_size: Optional[int] = None,
    random_state: Optional[int] = None,
) -> dict[str, Any]:
    """Train clustering models with hyperparameter tuning from configuration.

    Parameters
    ----------
    X_train : np.ndarray
        Training data, shape ``(n_samples, n_features)``.
    tuning_cfg : dict[str, Any]
        The ``hyperparameter_tuning`` section of the step config:
        ``{ "strategy": ..., "scoring": ..., "grids": { algo: { param: [...] } } }``.
    fit_cfg : dict[str, Any]
        The ``fit`` section of the step config (e.g. ``{"fit_best_only": true}``).
    problem_type : str, optional
        Problem type string for model registry lookup (default ``"clustering"``).
    scoring : str or None, optional
        Override scoring metric (default from ``tuning_cfg["scoring"]``).
    grid_search_sample_size : int or None, optional
        Override subsample size for grid search (default from ``tuning_cfg["grid_search_sample_size"]``).
    random_state : int or None, optional
        Override random seed (default from ``tuning_cfg["random_state"]``).

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - ``"best_models"``: dict mapping algorithm name → fitted estimator.
        - ``"best_params"``: dict mapping algorithm name → best parameter dict.
        - ``"results"``: dict mapping algorithm name → list of (params, score) for all grid combos.
        - ``"cluster_labels"``: dict mapping algorithm name → array of labels on full ``X_train``.
        - ``"metadata"``: dict with general info (refit, metric, etc.).

    Raises
    ------
    ValueError
        If ``X_train`` is empty or no algorithms are enabled.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if X_train.ndim != 2 or X_train.shape[0] == 0:
        raise ValueError(f"[trainer] X_train must be 2D non-empty, got shape {X_train.shape}")

    # ------------------------------------------------------------------
    # Extract parameters (config first, then overrides)
    # ------------------------------------------------------------------
    scoring = scoring or tuning_cfg.get("scoring", "silhouette")
    grid_search_sample_size = grid_search_sample_size or tuning_cfg.get("grid_search_sample_size")
    random_state = random_state or tuning_cfg.get("random_state", None)
    refit = fit_cfg.get("fit_best_only", True)
    grids: dict[str, Any] = tuning_cfg.get("grids", {})

    log.info("[trainer] received grids keys: %s", list(grids.keys()))

    if not grids:
        log.warning("[trainer] no 'grids' defined in hyperparameter_tuning – nothing to train")
        return {
            "best_models": {},
            "best_params": {},
            "results": {},
            "cluster_labels": {},
            "metadata": {"scoring": scoring, "refit": refit, "grid_search_sample_size": grid_search_sample_size},
        }

    log.info(
        "[trainer] starting tuning: scoring=%s, refit=%s, grid_search_sample_size=%s, random_state=%s, grids=%s",
        scoring, refit, grid_search_sample_size, random_state, list(grids.keys()),
    )

    # ------------------------------------------------------------------
    # Subsample for grid search if requested
    # ------------------------------------------------------------------
    n_samples = X_train.shape[0]
    if grid_search_sample_size is not None and grid_search_sample_size < n_samples:
        rng = np.random.default_rng(seed=random_state)
        idx = rng.choice(n_samples, size=grid_search_sample_size, replace=False)
        X_grid = X_train[idx]
        log.info("[trainer] using subsample of %d rows for grid search (seed=%s)", grid_search_sample_size, random_state)
    else:
        X_grid = X_train
        log.debug("[trainer] using full training data for grid search (%d rows)", n_samples)

    # ------------------------------------------------------------------
    # Prepare result containers
    # ------------------------------------------------------------------
    best_models: dict[str, BaseEstimator] = {}
    best_params: dict[str, dict[str, Any]] = {}
    results: dict[str, list[dict[str, Any]]] = {}
    cluster_labels: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Iterate over algorithms defined in grids
    # ------------------------------------------------------------------
    for algo_name, algo_grid in grids.items():
        log.info("[trainer] processing algorithm '%s'", algo_name)

        # 1. Get model class from registry
        try:
            model_class = get_model_class(problem_type, algo_name)
            log.debug("[trainer] model class: %s", model_class.__name__)
        except (KeyError, ValueError) as e:
            log.error("[trainer] cannot get model class for '%s': %s – skipping", algo_name, e)
            continue

        # 2. Filter out non‑parameter keys (e.g., output, enabled, description)
        non_param_keys = {'output', 'enabled', 'description'}
        pure_params = {k: v for k, v in algo_grid.items() if k not in non_param_keys}

        log.debug("[trainer] algo=%s | pure_params keys: %s", algo_name, list(pure_params.keys()))

        # log.debug("[trainer] algo=%s | number of combos: %d", algo_name, len(combos))

        param_keys = list(pure_params.keys())
        log.debug("[trainer] algo=%s | param_keys: %s", algo_name, param_keys)

        if not param_keys:
            log.warning("[trainer] grid for '%s' has no valid hyperparameters – using defaults", algo_name)
            combos = [{}]
        else:
            from itertools import product
            #param_values = [pure_params[k] if isinstance(pure_params[k], list) else [pure_params[k]] for k in param_keys]
            param_values = [pure_params[k] for k in param_keys]
            combos = [dict(zip(param_keys, vals)) for vals in product(*param_values)]

        log.info("[trainer] '%s': %d hyperparameter combinations", algo_name, len(combos))

        best_score = -1.0
        best_combo_params = None
        algo_results: list[dict[str, Any]] = []

        # 3. Evaluate each combination on subsample (or full data)
        for combo in combos:
            # Build model instance
            try:
                model = model_class(**combo)
                log.debug("[trainer] trying params=%s", combo)
            except Exception as e:
                log.warning("[trainer] failed to instantiate %s with params %s: %s", algo_name, combo, e)
                continue

            # Fit on subsample
            try:
                model.fit(X_grid)
                pred_labels = model.labels_
            except Exception as e:
                log.warning("[trainer] fit failed for %s params %s: %s", algo_name, combo, e)
                continue

            # Compute score
            score = _score_clustering(X_grid, pred_labels, scoring)
            algo_results.append({
                "params": combo,
                "score": score,
                "n_clusters": int(len(set(pred_labels)) - (1 if -1 in pred_labels else 0)),
            })

            if score > best_score:
                best_score = score
                best_combo_params = combo.copy()

        # Store results for this algorithm
        results[algo_name] = algo_results
        best_params[algo_name] = best_combo_params

        # 4. Refit on full data if requested
        if refit and best_combo_params is not None:
            log.info("[trainer] refitting '%s' on full data with best params: %s", algo_name, best_combo_params)
            try:
                full_model = model_class(**best_combo_params)
                full_model.fit(X_train)
                best_models[algo_name] = full_model
                cluster_labels[algo_name] = full_model.labels_
                log.info("[trainer] '%s' refit done – %d clusters", algo_name, len(set(full_model.labels_)))
            except Exception as e:
                log.error("[trainer] refit failed for '%s': %s", algo_name, e)
        elif not refit and best_combo_params is not None:
            # Store the model trained on subsample (optional)
            try:
                sub_model = model_class(**best_combo_params)
                sub_model.fit(X_grid)
                best_models[algo_name] = sub_model
                cluster_labels[algo_name] = sub_model.labels_
                log.info("[trainer] storing subsample model for '%s'", algo_name)
            except Exception as e:
                log.error("[trainer] storing model for '%s' failed: %s", algo_name, e)
        else:
            log.warning("[trainer] no best params found for '%s' – skipping model storage", algo_name)

        # Log best score
        log.info("[trainer] '%s' best score (%s): %.6f with params: %s",
                 algo_name, scoring, best_score, best_combo_params)

    # ------------------------------------------------------------------
    # Build final metadata
    # ------------------------------------------------------------------
    metadata = {
        "scoring": scoring,
        "refit": refit,
        "grid_search_sample_size": grid_search_sample_size if grid_search_sample_size else "full",
        "random_state": random_state,
        "n_algorithms_processed": len(best_models),
        "note": "Hyperparameter tuning via grid search over config grids.",
    }

    log.info("[trainer] done – trained %d algorithm(s)", len(best_models))
    return {
        "best_models": best_models,
        "best_params": best_params,
        "results": results,
        "cluster_labels": cluster_labels,
        "metadata": metadata,
    }