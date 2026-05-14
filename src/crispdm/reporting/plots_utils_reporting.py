# src/crispdm/reporting/plots_utils_reporting.py
from __future__ import annotations

# ---------------------------------------------------------------------------
# SECTION 1 – Standard-library imports
# ---------------------------------------------------------------------------
from collections.abc import Sequence

# ---------------------------------------------------------------------------
# SECTION 2 – Third-party imports
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# SECTION 3 – Internal imports
# ---------------------------------------------------------------------------
from crispdm.common.logging_adapter_common import get_logger

# ---------------------------------------------------------------------------
# SECTION 4 – Module-level logger
# ---------------------------------------------------------------------------
log = get_logger(__name__)

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# Plot-factory utilities producing Matplotlib Figure objects in consistent
# visual style across all modelling tracks.
# =============================================================================

# =============================================================================
# SECTION 5 – Data-understanding figures  (Stage 2)
# =============================================================================


def plot_missingness_top(df: pd.DataFrame, *, title: str, top_n: int) -> plt.Figure:
    """Return horizontal bar chart of top-N columns by missingness %."""
    # Step 1: Compute per-column missingness and select top_n
    miss = (df.isna().mean() * 100).sort_values(ascending=False).head(top_n)
    miss = miss.iloc[::-1]

    # Step 2: Build figure via OO API
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(miss.index.astype(str), miss.values)
    ax.set_title(title)
    ax.set_xlabel("missing %")

    log.debug("[plots] plot_missingness_top top_n=%d", top_n)
    return fig


def plot_target_distribution(s: pd.Series, *, title: str, top_n: int = 30) -> plt.Figure:
    """Return bar chart of target value counts."""
    # Step 1: Compute value counts
    vc = s.value_counts(dropna=False).head(top_n)

    # Step 2: Build figure via OO API
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(vc.index.astype(str), vc.values)
    ax.set_title(title)
    ax.set_xticklabels(vc.index.astype(str), rotation=45, ha="right")

    log.debug("[plots] plot_target_distribution top_n=%d", top_n)
    return fig


def plot_numeric_hist(df: pd.DataFrame, col: str, *, title: str, bins: int = 30) -> plt.Figure:
    """Return histogram for single numeric column."""
    # Step 1: Extract non-null values
    values = df[col].dropna().to_numpy()

    # Step 2: Build figure via OO API
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(col)
    ax.set_ylabel("count")

    log.debug("[plots] plot_numeric_hist col=%s bins=%d", col, bins)
    return fig


def plot_categorical_distribution(
        df: pd.DataFrame, columns: list[str], *, title: str, sample_rows: int | None = None
) -> plt.Figure:
    """Return countplot for multiple categorical columns."""
    # Step 1: Sample if requested
    data = df.sample(n=min(sample_rows, len(df)), random_state=42) if sample_rows else df

    # Step 2: Build figure with subplots
    n_cols = len(columns)
    fig, axes = plt.subplots(nrows=n_cols, ncols=1, figsize=(10, 4 * n_cols))
    if n_cols == 1:
        axes = [axes]

    # Step 3: Plot each column
    for ax, col in zip(axes, columns):
        if col not in data.columns:
            continue
        vc = data[col].value_counts().head(20)
        ax.bar(vc.index.astype(str), vc.values)
        ax.set_title(f"{col} distribution")
        ax.set_xticklabels(vc.index.astype(str), rotation=45, ha="right")
        ax.set_ylabel("count")

    fig.suptitle(title, y=1.0)
    fig.tight_layout()

    log.debug("[plots] plot_categorical_distribution cols=%d", len(columns))
    return fig


def plot_temporal_overview(
        df: pd.DataFrame, time_column: str, *, title: str, resample_rule: str = "D", sample_rows: int | None = None
) -> plt.Figure:
    """Return time series plot of daily counts."""
    # Step 1: Sample if requested
    data = df.sample(n=min(sample_rows, len(df)), random_state=42) if sample_rows else df

    # Step 2: Ensure datetime column
    if time_column not in data.columns:
        log.warning("[plots] time_column=%s not found", time_column)
        return plt.figure()

    ts = pd.to_datetime(data[time_column], errors="coerce")
    ts_clean = ts.dropna()

    # Step 3: Resample and count
    counts = ts_clean.to_frame(name="ts").groupby(pd.Grouper(key="ts", freq=resample_rule)).size()

    # Step 4: Build figure via OO API
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(counts.index, counts.values)
    ax.set_title(title)
    ax.set_xlabel("date")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)

    log.debug("[plots] plot_temporal_overview resample=%s points=%d", resample_rule, len(counts))
    return fig


def plot_target_by_category(
        df: pd.DataFrame, target_column: str, group_by: str, *, title: str, sample_rows: int | None = None
) -> plt.Figure:
    """Return stacked bar chart of target distribution by category."""
    # Step 1: Sample if requested
    data = df.sample(n=min(sample_rows, len(df)), random_state=42) if sample_rows else df

    # Step 2: Validate columns
    if target_column not in data.columns or group_by not in data.columns:
        log.warning("[plots] missing columns target=%s group=%s", target_column, group_by)
        return plt.figure()

    # Step 3: Build crosstab
    ct = pd.crosstab(data[group_by], data[target_column])

    # Step 4: Build figure via OO API
    fig, ax = plt.subplots(figsize=(12, 6))
    ct.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(group_by)
    ax.set_ylabel("count")
    ax.legend(title=target_column, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    log.debug("[plots] plot_target_by_category target=%s group=%s", target_column, group_by)
    return fig


# =============================================================================
# SECTION 6 – Regression figures  (Stages 4 – 5)
# =============================================================================


def plot_residuals(y_true: NDArray[np.floating], y_pred: NDArray[np.floating], *, title: str) -> plt.Figure:
    """Return residuals scatter plot."""
    # Step 1: Compute residuals
    resid = y_true - y_pred

    # Step 2: Build figure via OO API
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, resid, s=10)
    ax.axhline(0, color="red", linewidth=0.8, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("y_pred")
    ax.set_ylabel("residual")

    log.debug("[plots] plot_residuals n=%d", len(y_true))
    return fig


# =============================================================================
# SECTION 7 – Classification figures  (Stages 4 – 5)
# =============================================================================


def plot_confusion_matrix(cm: NDArray[np.integer], labels: Sequence[str], *, title: str) -> plt.Figure:
    """Return annotated confusion matrix heatmap."""
    # Step 1: Build figure via OO API
    fig, ax = plt.subplots(figsize=(6, 5))

    # Step 2: Render heatmap
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)

    # Step 3: Label axes
    ticks = range(len(labels))
    ax.set_xticks(list(ticks))
    ax.set_xticklabels(list(labels), rotation=45, ha="right")
    ax.set_yticks(list(ticks))
    ax.set_yticklabels(list(labels))
    fig.colorbar(im, ax=ax)

    # Step 4: Annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    log.debug("[plots] plot_confusion_matrix classes=%d", len(labels))
    return fig


# =============================================================================
# SECTION 8 – Clustering figures  (Stage 4)
# =============================================================================


def plot_cluster_sizes(labels: NDArray[np.integer] | Sequence[int], *, title: str) -> plt.Figure:
    """Return bar chart of sample counts per cluster."""
    # Step 1: Count per cluster
    s = pd.Series(labels).value_counts().sort_index()

    # Step 2: Build figure via OO API
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(s.index.astype(str), s.values)
    ax.set_title(title)
    ax.set_xlabel("cluster")
    ax.set_ylabel("count")

    log.debug("[plots] plot_cluster_sizes n_clusters=%d", len(s))
    return fig