# src/crispdm/reporting/artifacts_service_reporting.py
from __future__ import annotations

# ---------------------------------------------------------------------------
# SECTION 1 – Standard-library imports
# ---------------------------------------------------------------------------
import json
import pickle
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# SECTION 2 – Third-party imports
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# SECTION 3 – Internal imports
# ---------------------------------------------------------------------------
from crispdm.configuration.enum_registry_config import PhaseDir, StageSubDir
from crispdm.common.logging_adapter_common import get_logger
from crispdm.common.path_service_common import resolve_path

# ---------------------------------------------------------------------------
# SECTION 4 – Module-level logger
# ---------------------------------------------------------------------------
log = get_logger(__name__)

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# Persistence utilities for terminal pipeline artifacts.
# Handles writing every human- or registry-facing output produced by phases
# 2–5 to the correct sub-directory of a run folder created by
# context_utils_core.make_run_dir().
#
# Separating terminal artifact persistence from intermediate data persistence
# (persist_utils_data.py) enforces the Single Responsibility Principle:
# persist_utils_data writes DataFrames consumed by the next stage;
# this module writes outputs consumed by humans, MLflow, or audit logs.
#
# Option A terminal artifacts:
#   Stage 2 → phase2_data_understanding/reports/profiling_report.json
#   Stage 2 → phase2_data_understanding/figures/*.png
#   Stage 2 → phase2_data_understanding/tables_png/*.png
#   Stage 3 → phase3_data_preparation/reports/preparation_report.json
#   Stage 3 → phase3_data_preparation/figures/*.png
#   Stage 4 → models/model.pkl           (clustering + classification track)
#   Stage 4 → phase4_data_modeling/reports/modeling_report.json
#   Stage 5 → metrics.json
#   Stage 5 → stage5_evaluation/reports/evaluation_report.json
#   Stage 5 → stage5_evaluation/figures/*.png
#
# Option B terminal artifacts:
#   Stage 2 → phase2_data_understanding/reports/profiling_report.json
#   Stage 2 → phase2_data_understanding/figures/*.png
#   Stage 3 → phase3_data_preparation/reports/preparation_report.json
#   Stage 4 → models/model.pkl           (all four modelling tracks)
#   Stage 4 → phase4_data_modeling/reports/modeling_report.json
#   Stage 5 → metrics.json
#   Stage 5 → stage5_evaluation/figures/*.png
#   Stage 5 → stage5_evaluation/tables_png/*.png
#
# Program flow:
# -----------------------------------------------------------------------------
# - stage/stage2_* → calls save_figure(), save_table_png(), save_stage_report()
#                    after profiling (options A and B share this)
# - stage/stage3_* → calls save_figure(), save_stage_report()
#                    for preparation summaries
# - stage/stage4_* → calls save_model_pickle() for trained estimators
#                    calls save_stage_report() for modeling summaries
# - stage/stage5_* → calls save_metrics(), save_figure(), save_stage_report()
#                    for evaluation outputs  (no parquet writes in Stage 5)
#
# Design patterns
# -----------------------------------------------------------------------------
# - GoF: none
# - Enterprise / Architectural:
#   - Artifact Repository (filesystem-based): centralises all terminal write
#     I/O so paths, filenames, and DPI settings are never scattered across
#     stage modules.  No stage calls fig.savefig() or pickle.dump() directly.
#   - Convention over Configuration: directory layout is driven by PhaseDir
#     and StageSubDir enums; filenames are supplied by the caller from YAML
#     configuration, never hardcoded inside this module.
# =============================================================================

# =============================================================================
# SECTION 5 – JSON / metrics helpers
# =============================================================================


def save_json(
        path: Path,
        payload: dict[str, Any],
) -> Path:
    """Serialise *payload* to a UTF-8 JSON file at *path*.

    Parameters
    ----------
    path:
        Destination file path.  Parent directory must already exist.
    payload:
        Arbitrary mapping to serialise; ``datetime`` and ``Path`` objects
        are coerced to strings via ``default=str``.

    Returns
    -------
    Path
        The path that was written, for call-chaining or audit logging.
    """
    # Step 1 – Serialise payload to an indented JSON string
    content = json.dumps(payload, indent=2, default=str)

    # Step 2 - Ensure the parent directory exists
    # parents=True creates missing folders; exist_ok=True prevents errors if it exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Step 2 – Write encoded content to disk
    path.write_text(content, encoding="utf-8")

    # Step 3 – Emit info-level confirmation
    log.info("[artifacts] json saved: %s", path)
    return path


def save_metrics(
        run_dir: Path,
        metrics: dict[str, Any],
) -> Path:
    """Persist pipeline metrics to ``<run_dir>/metrics.json``.

    Writes the canonical ``metrics.json`` file at the root of the current
    run directory.  The file is created (or overwritten) on each call, so
    callers should accumulate all metrics in a single dict before invoking
    this function.

    Parameters
    ----------
    run_dir:
        Root of the current pipeline run as returned by
        ``context_utils_core.make_run_dir()``.
    metrics:
        Flat or nested mapping of metric names to scalar values.

    Returns
    -------
    Path
        Absolute path of the written ``metrics.json`` file.
    """
    # Step 1 – Resolve the canonical metrics file path at the run root
    out_path = run_dir / StageSubDir.METRICS.value

    # Step 2 – Delegate serialisation to save_json
    return save_json(out_path, metrics)


# =============================================================================
# SECTION 6 – Stage report helper
# =============================================================================


def save_stage_report(
        run_dir: Path,
        stage_name: str,
        payload: dict[str, Any],
        filename: str,
) -> Path:
    """Write a stage-level summary report as JSON.

    The destination directory ``<run_dir>/<stage_name>/reports/`` is
    expected to already exist (created by ``make_run_dir()``).
    No ``mkdir()`` call is issued here.

    Parameters
    ----------
    run_dir:
        Root of the current pipeline run.
    stage_name:
        CRISP-DM stage sub-directory name (e.g. ``PhaseDir.PHASE2.value``).
    payload:
        Mapping of summary statistics or metadata for the stage.
    filename:
        Name of file into "output_artifacts"

    Returns
    -------
    Path
        Absolute path of the written ``stage_report.json``.
    """
    # Step 1 – Build destination path using the StageSubDir.REPORTS convention
    report_path = run_dir / stage_name / StageSubDir.REPORTS.value / filename

    # Step 2 – Delegate serialisation to save_json
    return save_json(report_path, payload)


# =============================================================================
# SECTION 7 – Table-as-PNG helper
# =============================================================================


def save_table_png(
        df: pd.DataFrame,
        *,
        out_path: Path,
        title: Optional[str] = None,
        max_rows: int = 30,
        dpi: int,
) -> Path:
    """Save a pandas DataFrame as a PNG image using a Matplotlib table.

    Keeps the project *"every artefact visible as PNG"* convention so that
    run results are inspectable without a notebook or CSV viewer.

    Parameters
    ----------
    df:
        DataFrame to render.  Must not be ``None``.
    out_path:
        Destination ``.png`` file path.
    title:
        Optional title rendered above the table.
    max_rows:
        Maximum number of rows to include; surplus rows are silently dropped.
    dpi:
        Dots-per-inch resolution for the saved image.  No default is
        provided; callers must pass an explicit value to prevent silent
        resolution mismatches across pipeline runs.

    Returns
    -------
    Path
        The path that was written.

    Raises
    ------
    ValueError
        If *df* is ``None``.
    """
    # Step 2: Resolve destination path to absolute location.
    resolved: Path = resolve_path(out_path)

    # Step 3: Create parent directories if they do not exist.
    resolved.parent.mkdir(parents=True, exist_ok=True)
    log.debug("[save_parquet] ensured parent dir=%s", resolved.parent)

    # -----

    # Step 1 – Guard: reject None DataFrame input explicitly
    if df is None:
        raise ValueError("df must not be None")

    # Step 2 – Truncate to max_rows to keep the PNG readable
    df2 = df.copy()
    if len(df2) > max_rows:
        df2 = df2.head(max_rows)

    # Step 3 – Create figure and axis; hide the default axis frame
    fig, ax = plt.subplots(figsize=(12, 0.4 * (len(df2) + 2)))
    ax.axis("off")

    # Step 4 – Optionally render a title above the table
    if title:
        ax.set_title(title)

    # Step 5 – Render the DataFrame as a Matplotlib table widget
    tbl = ax.table(
        cellText=df2.values,
        colLabels=list(df2.columns),
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.2)

    # Step 6 – Apply tight layout, write PNG at caller-specified resolution
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    # Step 7 – Emit info-level confirmation
    log.info("[artifacts] table png saved: %s", out_path)
    return out_path


# =============================================================================
# SECTION 8 – Figure helper
# =============================================================================


def save_figure(
        fig: plt.Figure,
        *,
        out_path: Path,
        dpi: int,
) -> Path:
    """Save a Matplotlib figure and close it to avoid memory leaks.

    Parameters
    ----------
    fig:
        Matplotlib ``Figure`` instance to persist.
    out_path:
        Destination ``.png`` file path.
    dpi:
        Dots-per-inch resolution for the saved image.  No default is
        provided; callers must pass an explicit value.

    Returns
    -------
    Path
        The path that was written.
    """
    # Step 1 – Apply tight layout to minimise surrounding whitespace
    fig.tight_layout()

    # Step 2 – Render and write the figure at caller-specified resolution
    fig.savefig(out_path, dpi=dpi)

    # Step 3 – Release figure from memory (critical inside notebook loops)
    plt.close(fig)

    # Step 4 – Emit info-level confirmation
    log.info("[artifacts] figure saved: %s", out_path)
    return out_path


# =============================================================================
# SECTION 9 – Model serialisation helper
# =============================================================================


def save_model_pickle(
        run_dir: Path,
        model: Any,
        filename: str = "model.pkl",
) -> Path:
    """Persist a scikit-learn-compatible model via pickle.

    Writes the serialised model to ``<run_dir>/models/<filename>``.
    The ``models/`` sub-directory is created if absent (defensive guard
    in case ``make_run_dir()`` was not called by the pipeline runner).

    Parameters
    ----------
    run_dir:
        Root of the current pipeline run.
    model:
        Any picklable object (typically a fitted ``Pipeline`` or estimator).
    filename:
        Target filename inside ``models/``; defaults to ``"model.pkl"``.

    Returns
    -------
    Path
        Absolute path of the written pickle file.
    """
    # Step 1 – Resolve destination path under the PhaseDir.MODELS sub-directory
    out_path = run_dir / PhaseDir.MODELS.value / filename

    # Step 2 – Ensure the models directory exists (defensive mkdir guard)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 3 – Serialise model to bytes and write to disk
    out_path.write_bytes(pickle.dumps(model))

    # Step 4 – Emit info-level confirmation
    log.info("[artifacts] model saved: %s", out_path)
    return out_path
