# src/crispdm/model/clustering_trainer_model.py
"""CRISP-DM Phase 4.3 – Model Training with hyperparameter tuning.

This module contains a pure function that trains clustering models
(KMeans, DBSCAN) according to configuration.  No file I/O or side effects.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.model_selection import StratifiedShuffleSplit

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
        metrics_sample_size: int = 10000,
        random_state: Optional[int] = 7,
) -> float:
    """Compute a clustering score using the requested metric with safety bounds.

    Parameters
    ----------
    X : np.ndarray
        Data matrix.
    labels : np.ndarray
        Cluster labels (may contain noise label -1 for DBSCAN).
    metric_name : str
        One of ``"silhouette"``, ``"davies_bouldin"``, ``"calinski_harabasz"``.
    metrics_sample_size : int, default 10000
        Maximum rows to evaluate for expensive metrics (like Silhouette) to avoid O(N^2) hangs.
    random_state : int or None, default 7
        Seed for reproducible score downsampling.

    Returns
    -------
    float
        The computed score. Returns ``-1.05`` if evaluation fails.
    """
    # Step 1: Validate enough clusters exist to compute metrics
    unique_labels = set(labels)
    n_clusters = len(unique_labels - {-1})
    if n_clusters < 2:
        log.warning("[trainer] only %d cluster(s) found – cannot compute %s", n_clusters, metric_name)
        return -1.05

    scorer = _SCORING_FUNCTIONS.get(metric_name)
    if scorer is None:
        log.error("[trainer] unknown scoring metric '%s' – falling back to silhouette", metric_name)
        scorer = silhouette_score

    # Step 2: Performance safety check: Downsample if matrix exceeds config constraints
    n_samples = X.shape[0]
    if metric_name == "silhouette" and n_samples > metrics_sample_size:
        log.debug(
            "[trainer] downsampling silhouette evaluation from %d to %d rows for performance",
            n_samples,
            metrics_sample_size,
        )
        rng = np.random.default_rng(seed=random_state)
        sub_idx = rng.choice(n_samples, size=metrics_sample_size, replace=False)
        X_eval = X[sub_idx]
        labels_eval = labels[sub_idx]

        if len(set(labels_eval) - {-1}) < 2:
            log.warning("[trainer] downsampled slice has single cluster – falling back to full data matrix")
            X_eval = X
            labels_eval = labels
    else:
        X_eval = X
        labels_eval = labels

    try:
        score = float(scorer(X_eval, labels_eval))
        log.debug("[trainer] %s = %.6f", metric_name, score)
        return score
    except Exception as e:
        log.warning("[trainer] scoring failed: %s – returning -1.05", e)
        return -1.05


def train_clustering_models(
        X_train: np.ndarray,
        tuning_cfg: dict[str, Any],
        fit_cfg: dict[str, Any],
        problem_type: str = "clustering",
        *,
        scoring: Optional[str] = None,
        grid_search_sample_size: Optional[int] = None,
        metrics_sample_size: Optional[int] = None,
        random_state: Optional[int] = None,
        stratify_labels: Optional[np.ndarray] = None,
        max_training_rows: Optional[int] = None,
) -> dict[str, Any]:
    """Train clustering models with hyperparameter tuning from configuration.

    Parameters
    ----------
    X_train : np.ndarray
        Training data, shape ``(n_samples, n_features)``.
    tuning_cfg : dict[str, Any]
        The ``hyperparameter_tuning`` section of the step config.
    fit_cfg : dict[str, Any]
        The ``fit`` section of the step config.
    problem_type : str, optional
        Problem type string for model registry lookup (default ``"clustering"``).
    scoring : str or None, optional
        Override scoring metric.
    grid_search_sample_size : int or None, optional
        Override subsample size for grid search.
    metrics_sample_size : int or None, optional
        Override subsample size for silhouette evaluations.
    random_state : int or None, optional
        Override random seed.
    stratify_labels : np.ndarray or None, optional
        Target array used to enforce stratified sampling on high-imbalance datasets.

    Returns
    -------
    dict[str, Any]
        Dictionary containing trained estimators, parameters, array labels and grid metadata.
    """
    # ------------------------------------------------------------------
    # Step 3: Input verification and parameter extraction
    # ------------------------------------------------------------------
    if X_train.ndim != 2 or X_train.shape[0] == 0:
        raise ValueError(f"[trainer] X_train must be 2D non-empty, got shape {X_train.shape}")

    # Trace incoming dictionaries before parameter resolving
    log.debug("[trainer] raw tuning_cfg incoming keys: %s", list(tuning_cfg.keys()))
    log.debug("[trainer] raw fit_cfg incoming keys: %s", list(fit_cfg.keys()))

    scoring = scoring or tuning_cfg.get("scoring", "silhouette")
    grid_search_sample_size = grid_search_sample_size or tuning_cfg.get("grid_search_sample_size")
    metrics_sample_size = metrics_sample_size or tuning_cfg.get("metrics_sample_size", 10000)
    random_state = random_state or tuning_cfg.get("random_state", 7)
    refit = fit_cfg.get("fit_best_only", True)
    grids: dict[str, Any] = tuning_cfg.get("grids", {})
    sample_method = tuning_cfg.get("sample_method", "stratified")

    log.info("[trainer] received grids keys from pipeline configuration: %s", list(grids.keys()))

    if not grids:
        log.warning("[trainer] no 'grids' defined in hyperparameter_tuning – nothing to train")
        return {
            "best_models": {},
            "best_params": {},
            "results": {},
            "cluster_labels": {},
            "metadata": {
                "scoring": scoring,
                "refit": refit,
                "grid_search_sample_size": grid_search_sample_size
            },
        }

    log.info(
        "[trainer] starting tuning sequence: scoring='%s', refit=%s, sample_method='%s', grid_search_sample_size=%s, metrics_sample_size=%s, random_state=%s",
        scoring, refit, sample_method, grid_search_sample_size, metrics_sample_size, random_state,
    )

    # ------------------------------------------------------------------
    # Step 4: Executing Subsampling for Grid Search
    # ------------------------------------------------------------------
    n_samples = X_train.shape[0]
    if grid_search_sample_size is not None and grid_search_sample_size < n_samples:
        if sample_method == "stratified" and stratify_labels is not None:
            log.info(
                "[trainer] executing 'stratified' subsampling using target labels array to manage SOC imbalance ratios"
            )
            if len(stratify_labels) != n_samples:
                raise ValueError(
                    f"Length of stratify_labels ({len(stratify_labels)}) does not match X_train rows ({n_samples})"
                )

            sss = StratifiedShuffleSplit(n_splits=1, train_size=grid_search_sample_size, random_state=random_state)
            grid_idx, _ = next(sss.split(X_train, stratify_labels))
            X_grid = X_train[grid_idx]
        else:
            log.warning("[trainer] sample_method is 'stratified' but stratify_labels array not provided. Falling back to random selection.")
            rng = np.random.default_rng(seed=random_state)
            grid_idx = rng.choice(n_samples, size=grid_search_sample_size, replace=False)
            X_grid = X_train[grid_idx]

        log.info("[trainer] grid optimization matrix successfully instantiated with shape: %s", X_grid.shape)
    else:
        X_grid = X_train
        log.debug("[trainer] bypassed sub-sampling layer, using full training array for grid calculations (%d rows)", n_samples)

    # Prepare tracking containers
    best_models: dict[str, BaseEstimator] = {}
    best_params: dict[str, dict[str, Any]] = {}
    results: dict[str, list[dict[str, Any]]] = {}
    cluster_labels: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Step 5: Hyperparameter Search Loop over registered Algorithms
    # ------------------------------------------------------------------
    for algo_name, algo_grid in grids.items():
        log.info("[trainer] processing algorithm target: '%s'", algo_name)
        log.debug("[trainer] raw algo_grid params received for '%s': %s", algo_name, algo_grid)

        try:
            model_class = get_model_class(problem_type, algo_name)
            log.debug("[trainer] dynamic class factory discovery succeeded: %s", model_class.__name__)
        except (KeyError, ValueError) as e:
            log.error("[trainer] cannot resolve class mapping definition for '%s': %s – skipping algorithm", algo_name, e)
            continue

        # Filter infrastructure tokens from mathematical grid arrays
        non_param_keys = {'output', 'enabled', 'description'}
        pure_params = {k: v for k, v in algo_grid.items() if k not in non_param_keys}
        param_keys = list(pure_params.keys())

        log.debug("[trainer] algo=%s | pure hyperparameters extracted for combination grid: %s", algo_name, param_keys)

        if not param_keys:
            log.warning("[trainer] grid definition for '%s' has no valid hyperparameters – applying framework defaults", algo_name)
            combos = [{}]
        else:
            from itertools import product
            param_values = [pure_params[k] for k in param_keys]
            combos = [dict(zip(param_keys, vals)) for vals in product(*param_values)]

        log.info("[trainer] '%s' layer initialization: processing evaluation over %d combinations", algo_name, len(combos))

        # Initialized to negative infinity to catch worst-case scenarios cleanly
        best_score = -float('inf')
        best_combo_params = None
        algo_results: list[dict[str, Any]] = []

        # Parameter Evaluation Iteration over the isolated Subsample Matrix
        for combo in combos:
            try:
                model = model_class(**combo)
                log.debug("[trainer] evaluating hyperparameter instance signature: %s", combo)
            except Exception as e:
                log.warning("[trainer] failed to instantiate model layer for %s with combination %s: %s", algo_name, combo, e)
                continue

            try:
                model.fit(X_grid)
                pred_labels = model.labels_
            except Exception as e:
                log.warning("[trainer] mathematical fitting routine failed for %s with combo %s: %s", algo_name, combo, e)
                continue

            # Calculate scores feeding current execution configuration bounds
            score = _score_clustering(
                X_grid,
                pred_labels,
                scoring,
                metrics_sample_size=metrics_sample_size,
                random_state=random_state
            )

            algo_results.append({
                "params": combo,
                "score": score,
                "n_clusters": int(len(set(pred_labels)) - (1 if -1 in pred_labels else 0)),
            })

            if score > best_score:
                best_score = score
                best_combo_params = combo.copy()

        results[algo_name] = algo_results
        best_params[algo_name] = best_combo_params

        # Extraer de fit_cfg si no viene como argumento directo
        max_rows = max_training_rows or fit_cfg.get("max_training_rows")

        # ------------------------------------------------------------------
        # Step 6: Model Serialization Refit or Subsample Storage Routing
        # ------------------------------------------------------------------
        if refit and best_combo_params is not None:
            log.info("[trainer] refitting '%s' on full dataset array with chosen parameters: %s", algo_name, best_combo_params)
            # Lógica de recorte de seguridad para el Refit
            X_refit = X_train
            if max_rows and max_rows < X_train.shape[0]:
                log.info("[trainer] aplicando recorte de seguridad para refit: %d de %d filas", max_rows, X_train.shape[0])
                X_refit = X_train[:max_rows]

            log.info("[trainer] refitting '%s' on %d samples", algo_name, X_refit.shape[0])

            try:
                full_model = model_class(**best_combo_params)
                #full_model.fit(X_train)
                # SOLO entrenamos una vez con los datos decididos (X_refit)
                full_model.fit(X_refit)
                best_models[algo_name] = full_model
                cluster_labels[algo_name] = full_model.labels_
                log.info("[trainer] '%s' full data refit completed – registered %d active clusters", algo_name, len(set(full_model.labels_)))
            except Exception as e:
                log.error("[trainer] full dataset calculation routine failed for model '%s': %s", algo_name, e)
        elif not refit and best_combo_params is not None:
            log.info("[trainer] storing isolated subsample execution run for '%s' per configuration specifications", algo_name)
            try:
                sub_model = model_class(**best_combo_params)
                sub_model.fit(X_grid)
                best_models[algo_name] = sub_model
                cluster_labels[algo_name] = sub_model.labels_
                log.info("[trainer] '%s' subsample storage done – registered %d active clusters", algo_name, len(set(sub_model.labels_)))
            except Exception as e:
                log.error("[trainer] caching isolated execution matrix for '%s' failed: %s", algo_name, e)
        else:
            log.warning("[trainer] grid exploration routine returned zero viable paths for '%s' – skipping container tracking", algo_name)

        log.info("[trainer] '%s' optimization routine finalized. Scoring (%s): %.6f | Best configuration: %s",
                 algo_name, scoring, best_score, best_combo_params)

    # ------------------------------------------------------------------
    # Step 7: Build final metadata
    # ------------------------------------------------------------------
    metadata = {
        "scoring": scoring,
        "refit": refit,
        "sample_method": sample_method,
        "grid_search_sample_size": grid_search_sample_size if grid_search_sample_size else "full",
        "metrics_sample_size": metrics_sample_size,
        "random_state": random_state,
        "n_algorithms_processed": len(best_models),
        "note": "Hyperparameter tuning via configuration-driven stratified grid search mapping security profiles.",
    }

    log.info("[trainer] execution pipeline terminated successfully – processed %d model definitions", len(best_models))

    return {
        "best_models": best_models,
        "best_params": best_params,
        "results": results,
        "cluster_labels": cluster_labels,
        "metadata": metadata,
    }