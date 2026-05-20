# src/crispdm/model/knn_distance_analyzer_model.py
"""CRISP-DM Phase 4.2 – k-NN Distance Analysis for DBSCAN eps estimation.

This module provides a pure function that computes k-nearest neighbour
distances on a numpy array and returns statistics, a suggested eps value,
and a matplotlib figure.  No file I/O or side effects.
"""

from __future__ import annotations

from typing import Any

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

from crispdm.common.logging_adapter_common import get_logger

log = get_logger(__name__)


def knn_distance_analysis(
    X: np.ndarray,
    params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], matplotlib.figure.Figure]:
    """Compute k‑NN distances and return statistics + suggestion for ``eps``.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape ``(n_samples, n_features)``.
    params : dict[str, Any]
        Configuration dictionary expected to contain:

        - ``k`` (int): number of neighbours (default 5).
        - ``metric`` (list[str]): metric(s) to use (first element used, default ``["euclidean"]``).
        - ``sample_size`` (int): if less than ``n_samples``, a random subsample is drawn (default ``None`` → no subsample).
        - ``random_state`` (int): seed for reproducibility of subsample (default ``None``).
        - ``percentiles`` (list[float]): percentiles to compute on sorted distances (default ``[10, 25, 50, 75, 90, 95, 99]``).

    Returns
    -------
    stats : dict[str, Any]
        Contains computed percentiles and general metadata.
    eps_suggested : dict[str, Any]
        Contains the suggested ``eps`` value (based on the 95th percentile of
        the average k‑NN distance) and the full percentile list.
    fig : matplotlib.figure.Figure
        A figure with the sorted k‑NN distance curve (elbow plot).

    Raises
    ------
    ValueError
        If ``X`` is empty or ``k`` is invalid.
    """
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError(f"[knn_distance_analysis] X must be a 2D non-empty array, got shape {X.shape}")

    # ------------------------------------------------------------------
    # Extract parameters with defaults
    # ------------------------------------------------------------------
    k: int = int(params.get("k", 5))
    if k < 1:
        raise ValueError(f"[knn_distance_analysis] k must be >= 1, got {k}")

    metric_list: list[str] = params.get("metric", ["euclidean"])
    metric: str = metric_list[0] if metric_list else "euclidean"

    sample_size: int | None = params.get("sample_size")
    random_state: int | None = params.get("random_state")
    percentiles: list[float] = params.get("percentiles", [10, 25, 50, 75, 90, 95, 99])

    n_samples = X.shape[0]
    log.debug(
        "[knn_distance_analysis] started: n_samples=%d, k=%d, metric=%s, "
        "sample_size=%s, percentiles=%s",
        n_samples, k, metric, sample_size, percentiles,
    )

    # ------------------------------------------------------------------
    # Subsample if requested
    # ------------------------------------------------------------------
    if sample_size is not None and sample_size < n_samples:
        rng = np.random.default_rng(seed=random_state)
        indices = rng.choice(n_samples, size=sample_size, replace=False)
        X_used = X[indices]
        log.info(
            "[knn_distance_analysis] subsampled from %d to %d rows (seed=%s)",
            n_samples, sample_size, random_state,
        )
    else:
        X_used = X
        log.debug("[knn_distance_analysis] using full data (%d rows)", n_samples)

    # ------------------------------------------------------------------
    # Compute distances to k nearest neighbours
    # ------------------------------------------------------------------
    log.debug("[knn_distance_analysis] fitting NearestNeighbors(k=%d, metric='%s')", k, metric)
    neigh = NearestNeighbors(n_neighbors=k, metric=metric, n_jobs=-1)
    neigh.fit(X_used)

    # distances shape = (n_used, k)
    distances, _ = neigh.kneighbors(X_used, n_neighbors=k)
    # Average distance to the k neighbours for each point
    avg_distances = np.mean(distances, axis=1)
    # Sort for elbow curve
    sorted_distances = np.sort(avg_distances)

    log.debug("[knn_distance_analysis] distances computed, shape=%s", distances.shape)

    # ------------------------------------------------------------------
    # Compute percentiles
    # ------------------------------------------------------------------
    computed_percentiles: dict[str, float] = {}
    for p in percentiles:
        val = float(np.percentile(sorted_distances, p))
        computed_percentiles[str(p)] = round(val, 6)

    log.info("[knn_distance_analysis] percentiles computed: %s", computed_percentiles)

    # Suggested eps = 95th percentile (or customise)
    eps_suggested_val = computed_percentiles.get("95", float(np.percentile(sorted_distances, 95)))

    stats: dict[str, Any] = {
        "k": k,
        "metric": metric,
        "n_samples_used": X_used.shape[0],
        "distance_mean": float(np.mean(avg_distances)),
        "distance_std": float(np.std(avg_distances)),
        "distance_min": float(np.min(avg_distances)),
        "distance_max": float(np.max(avg_distances)),
        "percentiles": computed_percentiles,
        "note": "Average distance to k nearest neighbours.",
    }

    eps_suggested: dict[str, Any] = {
        "eps_suggested": round(eps_suggested_val, 6),
        "source_percentile": "95th percentile of average k-NN distance",
        "all_percentiles": computed_percentiles,
        "recommendation": (
            f"Consider eps ~ {eps_suggested_val:.4f} for DBSCAN. "
            "Tune with factor [0.5, 1.5] in hyperparameter search."
        ),
    }

    # ------------------------------------------------------------------
    # Build elbow plot figure
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sorted_distances, color="steelblue", linewidth=1.5)
    ax.axhline(y=eps_suggested_val, color="red", linestyle="--",
               label=f"Suggested eps = {eps_suggested_val:.4f} (P95)")
    ax.set_xlabel("Data points sorted by distance")
    ax.set_ylabel(f"Average distance to {k}-NN")
    ax.set_title(f"k-NN Distance Elbow Plot (k={k}, metric='{metric}')")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    log.debug("[knn_distance_analysis] plot generated")

    return stats, eps_suggested, fig