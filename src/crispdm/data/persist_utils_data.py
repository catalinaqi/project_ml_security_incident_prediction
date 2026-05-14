# src/crispdm/data/persist_utils_data.py
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from crispdm.common.logging_adapter_common import get_logger
from crispdm.common.path_service_common import resolve_path

# Initialize logger.
log = get_logger(__name__)

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# Persistence utilities for intermediate pipeline artifacts.
# Handles writing data and objects produced between phases to disk so the
# next stage can consume them without re-running prior phases.
#
# Separating persistence from loading (load_utils_data.py) enforces the
# Single Responsibility Principle: load_utils_data reads, this module writes.
#
# Option A intermediate artifacts:
#   Stage 2 → sample_200k_with_source.parquet  (train + test, with source col)
#   Stage 3 → train_prepared_150k.parquet      (train rows, transformed)
#   Stage 3 → test_prepared_50k.parquet        (test rows, transformed)
#   Stage 3 → train_incident_grade.npy         (train labels for validation)
#   Stage 3 → test_incident_grade.npy          (test labels for validation)
#   Stage 3 → transformers_pipeline.pkl        (fitted imputer/scaler/encoder)
#   Stage 3 → 15× JSON reports                (feature selection, cleaning, etc.)
#
# Option B intermediate artifacts:
#   Stage 2 → train_sample_200k.parquet        (train only, no source col)
#   Stage 3 → train_prepared_200k.parquet      (train rows, transformed)
#   Stage 3 → train_incident_grade.npy         (train labels for validation)
#   Stage 3 → transformers_pipeline.pkl        (fitted imputer/scaler/encoder)
#   Stage 3 → 15× JSON reports                (feature selection, cleaning, etc.)
#
# Program flow:
# -----------------------------------------------------------------------------
# - stage/stage2_* → calls save_parquet() after sampling train+test (A)
#                    or sampling train only (B)
# - stage/stage3_* → calls save_parquet() for prepared train and test parquets
#                    calls save_numpy() for IncidentGrade labels (NEW)
#                    calls save_json() for reports with naming convention (NEW)
#                    calls save_pickle() for the fitted transformers pipeline
# - stage/stage4_* → calls save_pickle() for trained models
# - stage/stage5_* → reads artifacts (no writes in Stage 5)
#
# Design patterns
# -----------------------------------------------------------------------------
# - GoF: none
# - Enterprise/Architectural:
#   - Data Access Layer (thin): isolates all write I/O from business logic.
#     No stage touches df.to_parquet() or pickle.dump() directly.
#   - Repository (partial): centralises artifact persistence so paths and
#     compression settings are never scattered across stage modules.
#
# NEW in Phase 3:
# -----------------------------------------------------------------------------
# - save_json(): Structured reports with naming convention
#   {step}.{method}.{technique}.{filename}.json
# - save_numpy(): NumPy arrays for IncidentGrade labels (post-hoc validation)
# =============================================================================


# =============================================================================
# SECTION 1 - PARQUET PERSISTENCE
# =============================================================================


def save_parquet(
        df: pd.DataFrame,
        path: str | Path,
        *,
        compression: str,
) -> Path:
    """
    Persist a DataFrame as a Parquet file with the specified compression.

    Called in Stages 2 and 3 to write intermediate datasets consumed by the
    next pipeline stage. ``snappy`` offers the best balance between
    compression ratio and read/write speed for pipeline artifacts.

    Intermediate artifact paths per option:

    Option A:
        Stage 2: ``sample_200k_with_source.parquet``
        Stage 3: ``train_prepared_150k.parquet``
        Stage 3: ``test_prepared_50k.parquet``

    Option B:
        Stage 2: ``train_sample_200k.parquet``
        Stage 3: ``train_prepared_200k.parquet``

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to persist.
    path : str | Path
        Relative or absolute destination path for the ``.parquet`` file.
        Resolved against the project root via ``resolve_path()``.
        Parent directories are created automatically if they do not exist.
    compression : str
        Parquet compression codec.
        Options: ``"snappy"`` | ``"gzip"`` | ``"brotli"`` | ``"zstd"`` | ``"none"``.
        Default: ``"snappy"`` (fast, industry-standard for pipeline artifacts).

    Returns
    -------
    Path
        Resolved absolute path where the file was written.

    Raises
    ------
    ValueError
        If ``df`` is empty (writing an empty parquet is likely a pipeline bug).
    Exception
        Any ``df.to_parquet()`` exception is logged then re-raised.

    Examples
    --------
    >>> path = save_parquet(
    ...     df_sample, "out/runs/clustering/.../sample_200k_with_source.parquet"
    ... )
    >>> print(path.stat().st_size / 1024**2, "MB")
    """
    # Step 1: Guard against persisting an empty DataFrame -- likely a bug.
    if df.empty:
        log.error(
            "[save_parquet] DataFrame is empty -- refusing to write "
            "an empty parquet to path=%s. "
            "Check the pipeline step that produced this DataFrame.",
            path,
        )
        raise ValueError(
            f"Cannot save an empty DataFrame to '{path}'. "
            f"An empty DataFrame at this stage indicates a pipeline bug."
        )

    # Step 2: Resolve destination path to absolute location.
    resolved: Path = resolve_path(path)
    log.debug(
        "[save_parquet] resolved path=%s compression=%s rows=%d cols=%d",
        resolved,
        compression,
        len(df),
        df.shape[1],
    )

    # Step 3: Create parent directories if they do not exist.
    resolved.parent.mkdir(parents=True, exist_ok=True)
    log.debug("[save_parquet] ensured parent dir=%s", resolved.parent)

    # Step 4: Write parquet to disk.
    try:
        log.info(
            "[save_parquet] writing rows=%d cols=%d compression=%s path=%s",
            len(df),
            df.shape[1],
            compression,
            resolved,
        )
        df.to_parquet(resolved, compression=cast(Any, compression), index=False)

    except Exception:
        log.exception("[save_parquet] to_parquet failed path=%s", resolved)
        raise

    # Step 5: Log written file size for storage awareness.
    size_mb: float = resolved.stat().st_size / (1024**2)
    log.info(
        "[save_parquet] written size_mb=%.2f compression=%s path=%s",
        size_mb,
        compression,
        resolved,
    )
    return resolved


# =============================================================================
# SECTION 2 - PICKLE PERSISTENCE
# =============================================================================


def save_pickle(
        obj: Any,
        path: str | Path,
) -> Path:
    """
    Persist a Python object to disk using pickle serialisation.

    Used in Stage 3 to serialise the fitted transformers pipeline
    (imputer + scaler + encoder) for on-the-fly application in Stage 5.
    Used in Stage 4 to serialise trained clustering models (KMeans, DBSCAN).

    Intermediate artifact paths per option:

    Both options Stage 3:
        ``transformers_pipeline.pkl`` — fitted sklearn Pipeline object.

    Both options Stage 4:
        ``kmeans_best.pkl``  — best KMeans model after grid search.
        ``dbscan_best.pkl``  — best DBSCAN model after grid search.

    Parameters
    ----------
    obj : Any
        Python object to serialise. Typically a fitted sklearn Pipeline,
        a trained clustering model, or a metadata dict.
    path : str | Path
        Relative or absolute destination path for the ``.pkl`` file.
        Resolved against the project root via ``resolve_path()``.
        Parent directories are created automatically if they do not exist.

    Returns
    -------
    Path
        Resolved absolute path where the file was written.

    Raises
    ------
    ValueError
        If ``obj`` is ``None`` (persisting None is likely a pipeline bug).
    Exception
        Any ``pickle.dump()`` exception is logged then re-raised.

    Examples
    --------
    >>> path = save_pickle(
    ...     transformers_pipeline, "out/runs/.../transformers_pipeline.pkl"
    ... )
    >>> path = save_pickle(kmeans_model, "out/runs/.../models/kmeans_best.pkl")
    """
    # Step 1: Guard against persisting None -- likely a pipeline bug.
    if obj is None:
        log.error(
            "[save_pickle] object is None -- refusing to serialise None to path=%s. "
            "Check the pipeline step that produced this object.",
            path,
        )
        raise ValueError(
            f"Cannot pickle None to '{path}'. "
            f"A None object at this stage indicates a pipeline bug."
        )

    # Step 2: Resolve destination path to absolute location.
    resolved: Path = resolve_path(path)
    log.debug(
        "[save_pickle] resolved path=%s object_type=%s",
        resolved,
        type(obj).__name__,
    )

    # Step 3: Create parent directories if they do not exist.
    resolved.parent.mkdir(parents=True, exist_ok=True)
    log.debug("[save_pickle] ensured parent dir=%s", resolved.parent)

    # Step 4: Serialise object to disk.
    try:
        log.info(
            "[save_pickle] serialising object_type=%s path=%s",
            type(obj).__name__,
            resolved,
        )
        with resolved.open("wb") as fh:
            pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)

    except Exception:
        log.exception("[save_pickle] pickle.dump failed path=%s", resolved)
        raise

    # Step 5: Log written file size.
    size_mb: float = resolved.stat().st_size / (1024**2)
    log.info(
        "[save_pickle] written size_mb=%.3f object_type=%s path=%s",
        size_mb,
        type(obj).__name__,
        resolved,
    )
    return resolved


# =============================================================================
# SECTION 3 - PICKLE LOADING
# =============================================================================


def load_pickle(
        path: str | Path,
) -> Any:
    """
    Load a pickled Python object from disk.

    Used in Stage 5 to deserialise the fitted transformers pipeline written
    by Stage 3 (``transformers_pipeline.pkl``) for on-the-fly application to
    each 100k-row test chunk during full evaluation.

    Also used in Stage 5 to load trained models written by Stage 4
    (``kmeans_best.pkl``, ``dbscan_best.pkl``) for cluster assignment.

    Security note: only load pickle files from trusted pipeline artifact
    directories. Never load pickles from external or user-supplied sources.

    Covers: Option A and Option B -- both use transformers_pipeline.pkl
    in Stage 5, though the transformers were fitted on different data:
      Option A: fitted on train_rows (~150k) from the combined sample.
      Option B: fitted on train_rows (200k) from the train-only sample.

    Parameters
    ----------
    path : str | Path
        Relative or absolute path to the ``.pkl`` file.
        Resolved against the project root via ``resolve_path()``.

    Returns
    -------
    Any
        Deserialised Python object.

    Raises
    ------
    FileNotFoundError
        If the resolved path does not exist.
    Exception
        Any ``pickle.load()`` exception is logged then re-raised.

    Examples
    --------
    >>> transformers = load_pickle("out/runs/.../transformers_pipeline.pkl")
    >>> kmeans = load_pickle("out/runs/.../models/kmeans_best.pkl")
    """
    # Step 1: Resolve path to absolute location.
    resolved: Path = resolve_path(path)
    log.debug("[load_pickle] resolved path=%s", resolved)

    # Step 2: Validate file existence before attempting to load.
    if not resolved.exists():
        log.error("[load_pickle] pickle file not found path=%s", resolved)
        raise FileNotFoundError(f"Pickle file not found: {resolved}")

    # Step 3: Log file size before loading for memory awareness.
    size_mb: float = resolved.stat().st_size / (1024**2)
    log.info(
        "[load_pickle] loading size_mb=%.3f path=%s",
        size_mb,
        resolved,
    )

    # Step 4: Deserialise object from disk.
    try:
        with resolved.open("rb") as fh:
            obj: Any = pickle.load(fh)  # noqa: S301 (trusted pipeline artifacts only)

    except Exception:
        log.exception("[load_pickle] pickle.load failed path=%s", resolved)
        raise

    # Step 5: Log loaded object type for audit trail.
    log.info(
        "[load_pickle] loaded object_type=%s path=%s",
        type(obj).__name__,
        resolved,
    )
    return obj


# =============================================================================
# SECTION 4 - JSON PERSISTENCE
# =============================================================================


def save_json(
        obj: Any,
        path: str | Path,
        *,
        indent: int = 2,
) -> Path:
    """
    Persist a Python dict or list as a formatted JSON file.

    Used in Stage 3 to save structured reports, feature lists, and metadata
    with the naming convention ``{step}.{method}.{technique}.{filename}``.

    Stage 3 JSON artifacts per step:

    Step 3.1 (Data Selection):
        ``3.1.manual.exclusions.selected_features.json``
        ``3.1.manual.exclusions.dropped_features.json``
        ``3.1.feature_selection.report.feature_selection_report.json``

    Step 3.2 (Data Cleaning):
        ``3.2.sentinel.removal.sentinel_removal_report.json``
        ``3.2.cleaning.imputation.cleaning_report.json``
        ``3.2.missing.before.missing_values_before.json``
        ``3.2.missing.after.missing_values_after.json``

    Step 3.3 (Data Transformation):
        ``3.3.scaling.robust.scaling_report.json``
        ``3.3.encoding.mappings.encoding_mappings.json``
        ``3.3.transformation.summary.transformation_summary.json``
        ``3.3.feature_engineering.aggregations.feature_engineering_report.json``

    Parameters
    ----------
    obj : Any
        Python object to serialise (typically dict or list).
        Must be JSON-serialisable (no numpy arrays, pandas objects, or
        custom classes without a custom encoder).
    path : str | Path
        Relative or absolute destination path for the ``.json`` file.
        Resolved against the project root via ``resolve_path()``.
        Parent directories are created automatically if they do not exist.
    indent : int, optional
        Number of spaces for pretty-printing. Default: 2.
        Use ``indent=None`` for compact single-line output.

    Returns
    -------
    Path
        Resolved absolute path where the file was written.

    Raises
    ------
    ValueError
        If ``obj`` is ``None`` (persisting None is likely a pipeline bug).
    TypeError
        If ``obj`` contains non-JSON-serialisable types.
    Exception
        Any ``json.dump()`` exception is logged then re-raised.

    Examples
    --------
    >>> report = {"dropped": ["Id", "IncidentId"], "kept": 42}
    >>> path = save_json(
    ...     report, "out/runs/.../3.1.manual.exclusions.dropped_features.json"
    ... )
    """
    # Step 1: Guard against persisting None -- likely a pipeline bug.
    if obj is None:
        log.error(
            "[save_json] object is None -- refusing to serialise None to path=%s. "
            "Check the pipeline step that produced this object.",
            path,
        )
        raise ValueError(
            f"Cannot save None to '{path}'. "
            f"A None object at this stage indicates a pipeline bug."
        )

    # Step 2: Resolve destination path to absolute location.
    resolved: Path = resolve_path(path)
    log.debug(
        "[save_json] resolved path=%s object_type=%s indent=%s",
        resolved,
        type(obj).__name__,
        indent,
    )

    # Step 3: Create parent directories if they do not exist.
    resolved.parent.mkdir(parents=True, exist_ok=True)
    log.debug("[save_json] ensured parent dir=%s", resolved.parent)

    # Step 4: Serialise object to disk with pretty-printing.
    try:
        log.info(
            "[save_json] writing object_type=%s path=%s",
            type(obj).__name__,
            resolved,
        )
        with resolved.open("w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=indent, ensure_ascii=False, sort_keys=False)

    except TypeError as err:
        log.error(
            "[save_json] object contains non-JSON-serialisable types path=%s error=%s",
            resolved,
            err,
        )
        raise

    except Exception:
        log.exception("[save_json] json.dump failed path=%s", resolved)
        raise

    # Step 5: Log written file size.
    size_kb: float = resolved.stat().st_size / 1024
    log.info(
        "[save_json] written size_kb=%.2f object_type=%s path=%s",
        size_kb,
        type(obj).__name__,
        resolved,
    )
    return resolved


# =============================================================================
# SECTION 5 - NUMPY PERSISTENCE
# =============================================================================


def save_numpy(
        arr: Any,
        path: str | Path,
) -> Path:
    """
    Persist a NumPy array to disk as a ``.npy`` file.

    Used in Stage 3 to save the ``IncidentGrade`` target labels separately
    from the prepared feature matrices for post-hoc validation (ARI, NMI)
    in Stage 4 and Stage 5.

    Stage 3 NumPy artifacts:

    Both options:
        ``train_incident_grade.npy`` — train labels for validation.
        ``test_incident_grade.npy``  — test labels for validation.

    The ``IncidentGrade`` column is dropped from the feature DataFrames
    before saving ``train_prepared_150k.parquet`` and
    ``test_prepared_50k.parquet`` because clustering is unsupervised.
    Labels are kept in separate ``.npy`` files so Stage 5 can compute
    external validation metrics (Adjusted Rand Index, Normalised Mutual
    Information) after cluster assignment.

    Parameters
    ----------
    arr : np.ndarray
        NumPy array to persist. Typically a 1D array of string labels
        (e.g. ``["TP", "BP", "FP", ...]``).
    path : str | Path
        Relative or absolute destination path for the ``.npy`` file.
        Resolved against the project root via ``resolve_path()``.
        Parent directories are created automatically if they do not exist.

    Returns
    -------
    Path
        Resolved absolute path where the file was written.

    Raises
    ------
    ValueError
        If ``arr`` is ``None`` or not a NumPy array.
    Exception
        Any ``np.save()`` exception is logged then re-raised.

    Examples
    --------
    >>> import numpy as np
    >>> labels = np.array(["TP", "BP", "FP", "TP"])
    >>> path = save_numpy(labels, "out/runs/.../train_incident_grade.npy")
    """
    # Step 1: Guard against None and validate type.
    if arr is None:
        log.error(
            "[save_numpy] array is None -- refusing to save None to path=%s. "
            "Check the pipeline step that produced this array.",
            path,
        )
        raise ValueError(
            f"Cannot save None to '{path}'. "
            f"A None array at this stage indicates a pipeline bug."
        )

    if not isinstance(arr, np.ndarray):
        log.error(
            "[save_numpy] object is not a numpy array type=%s path=%s",
            type(arr).__name__,
            path,
        )
        raise ValueError(
            f"save_numpy() requires a numpy.ndarray, got {type(arr).__name__}."
        )

    # Step 2: Resolve destination path to absolute location.
    resolved: Path = resolve_path(path)
    log.debug(
        "[save_numpy] resolved path=%s shape=%s dtype=%s",
        resolved,
        arr.shape,
        arr.dtype,
    )

    # Step 3: Create parent directories if they do not exist.
    resolved.parent.mkdir(parents=True, exist_ok=True)
    log.debug("[save_numpy] ensured parent dir=%s", resolved.parent)

    # Step 4: Save array to disk in NumPy binary format.
    try:
        log.info(
            "[save_numpy] saving shape=%s dtype=%s path=%s",
            arr.shape,
            arr.dtype,
            resolved,
        )
        np.save(resolved, arr, allow_pickle=False)

    except Exception:
        log.exception("[save_numpy] np.save failed path=%s", resolved)
        raise

    # Step 5: Log written file size.
    size_kb: float = resolved.stat().st_size / 1024
    log.info(
        "[save_numpy] written size_kb=%.2f shape=%s dtype=%s path=%s",
        size_kb,
        arr.shape,
        arr.dtype,
        resolved,
    )
    return resolved
