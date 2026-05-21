# src/crispdm/interpretation/business_alignment_evaluator_interpretation.py
"""Business alignment evaluation for clustering – Phase 5.2.

Provides functions to compute confusion matrices between cluster assignments
and ground truth labels, and to generate alignment visualisations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from crispdm.common.logging_adapter_common import get_logger

log = get_logger(__name__)


def compute_confusion_matrix(
    cluster_labels: np.ndarray,
    true_labels: np.ndarray,
    normalize: Optional[str] = None,
    collapse_top_n: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute a confusion matrix between cluster assignments and true labels.

    Parameters
    ----------
    cluster_labels : np.ndarray
        Cluster assignments (integer labels).
    true_labels : np.ndarray
        Ground truth class labels (integer encoded).
    normalize : str, optional
        One of ``"pred"``, ``"true"``, ``"all"``, or ``None``.
        If ``None``, raw counts are returned.
    collapse_top_n : int, optional
        If given and the number of clusters > `collapse_top_n`, only the
        top `collapse_top_n` clusters (by size) are kept; all others are
        aggregated into a special "other" cluster. Ignored for standard confusion.

    Returns
    -------
    Dict[str, Any]
        With keys:
        - ``"matrix"`` : nested list (rows = true classes, columns = clusters)
        - ``"row_names"`` : list of true class names (as strings)
        - ``"col_names"`` : list of cluster names (as strings)
        - ``"normalization"`` : used normalization or None
        - ``"accuracy"`` : overall accuracy (diagonal / total)
        - ``"n_clusters"`` : number of clusters (after collapsing if any)
        - ``"n_true_classes"`` : number of true classes
    """
    if len(cluster_labels) != len(true_labels):
        raise ValueError(
            f"Length mismatch: cluster_labels ({len(cluster_labels)}) vs "
            f"true_labels ({len(true_labels)})"
        )

    # --- Collapse clusters if requested ---
    if collapse_top_n is not None and collapse_top_n > 0:
        unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
        if len(unique_clusters) > collapse_top_n:
            # Keep top N clusters by size
            top_idx = np.argsort(counts)[::-1][:collapse_top_n]
            top_clusters = set(unique_clusters[top_idx])
            collapsed = np.where(
                np.isin(cluster_labels, list(top_clusters)),
                cluster_labels,
                -1,  # assign "other" label
            )
            # Re-label clusters sequentially to avoid gaps
            unique_new = np.sort(np.unique(collapsed))
            mapping = {old: i for i, old in enumerate(unique_new)}
            collapsed = np.array([mapping[x] for x in collapsed])
            cluster_labels = collapsed

    # --- Compute confusion matrix ---
    cm = confusion_matrix(
        true_labels,
        cluster_labels,
        normalize=normalize,
    )

    # Build human-readable names
    unique_true = np.unique(true_labels)
    unique_clust = np.unique(cluster_labels)

    row_names = [str(t) for t in unique_true]
    col_names = [str(c) for c in unique_clust]

    # Accuracy = sum of diagonal / total
    if normalize is None:
        accuracy = np.trace(cm) / cm.sum()
    else:
        # Normalized matrix: diagonal is already a proportion; accuracy not meaningful
        accuracy = None

    return {
        "matrix": cm.tolist(),
        "row_names": row_names,
        "col_names": col_names,
        "normalization": normalize,
        "accuracy": accuracy,
        "n_clusters": len(unique_clust),
        "n_true_classes": len(unique_true),
    }


def generate_alignment_plot(
    cluster_labels_dict: Dict[str, np.ndarray],
    true_labels: np.ndarray,
    targets: List[str],
    plot_type: str = "stacked_bar",
) -> plt.Figure:
    """Generate a stacked bar chart showing cluster composition per ground truth class.

    For each target model, a subplot is created showing the distribution
    of cluster assignments within each true class.

    Parameters
    ----------
    cluster_labels_dict : Dict[str, np.ndarray]
        Mapping from model name (e.g., ``"kmeans_n2"``) to cluster label array.
    true_labels : np.ndarray
        Ground truth labels (integer encoded).
    targets : List[str]
        Which models to include (e.g., ``["kmeans_n2", "kmeans_n3"]``).
    plot_type : str
        Currently only ``"stacked_bar"`` is supported.

    Returns
    -------
    plt.Figure
        Matplotlib figure with one subplot per target model.
    """
    n_models = len(targets)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 4), squeeze=False)
    axes = axes.flatten()

    unique_true = np.sort(np.unique(true_labels))

    for ax, model_name in zip(axes, targets):
        labels = cluster_labels_dict.get(model_name)
        if labels is None:
            ax.text(0.5, 0.5, f"No data for {model_name}", ha="center", va="center")
            ax.set_title(model_name)
            continue

        # Build a contingency table
        df = pd.DataFrame({"true": true_labels, "cluster": labels})
        contingency = pd.crosstab(df["true"], df["cluster"], normalize="index")

        # Reindex to ensure all true classes are present (even if empty)
        contingency = contingency.reindex(index=unique_true, fill_value=0)

        # Plot stacked bar
        contingency.plot.bar(
            stacked=True,
            ax=ax,
            legend=False,
            colormap="tab10",
        )
        ax.set_title(f"{model_name} – Cluster vs IncidentGrade")
        ax.set_xlabel("IncidentGrade")
        ax.set_ylabel("Proportion")
        ax.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")

    fig.tight_layout()
    return fig