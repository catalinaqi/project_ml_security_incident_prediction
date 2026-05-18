# src/crispdm/data/load_loader_data.py
from __future__ import annotations

from pathlib import Path
from typing import Any,  Optional
import pickle
import pandas as pd
from crispdm.common.logging_adapter_common import get_logger
from crispdm.common.path_service_common import resolve_path

log = get_logger(__name__)


def load_parquet(
        path: str | Path,
        *,
        columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Load a Parquet file produced by a previous pipeline stage.

    Used in Stages 3, 4, and 5 (Option A) to read intermediate Parquet files.
    The input_source, input_source_sample, and input_source_full fields of
    ReadStrategyContract provide the path values callers pass here.

    Covers:
      Option A Stage 3: reads sample_200k_with_source.parquet
      Option B Stage 3: reads train_sample_200k.parquet
      Option A Stage 4: reads train_prepared_150k.parquet
      Option B Stage 4: reads train_prepared_200k.parquet
      Option A Stage 5: reads test_prepared_50k.parquet (quick eval)

    Parameters
    ----------
    path : str | Path
        Relative or absolute path to the .parquet file.
    columns : Optional[list[str]]
        Subset of columns to load. None loads all columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with all rows and the requested columns.

    Raises
    ------
    FileNotFoundError
        If the resolved path does not exist.
    """
    # Step 1: Resolve path to absolute location.
    resolved = resolve_path(path)
    log.debug("[load_parquet] path=%s columns=%s", resolved, columns)

    # Step 2: Validate file existence before attempting to read.
    if not resolved.exists():
        log.error("[load_parquet] parquet file not found path=%s", resolved)
        raise FileNotFoundError(f"Parquet file not found: {resolved}")

    # Step 3: Load parquet — columns=None reads all columns.
    try:
        log.info("[load_parquet] loading path=%s columns=%s", resolved, columns)
        df: pd.DataFrame = pd.read_parquet(resolved, columns=columns)

    except Exception:
        log.exception("[load_parquet] read_parquet failed path=%s", resolved)
        raise

    # Step 4: Log result dimensions for audit trail.
    log.info(
        "[load_parquet] loaded rows=%d cols=%d path=%s", len(df), df.shape[1], resolved
    )
    return df

def load_pickle(
        path: str | Path,
) -> Any:
    """
    Load a pickled Python object from disk.

    Re-exported from :mod:`crispdm.data.persist_utils_data` for backward
    compatibility. Callers are encouraged to import from ``persist_utils_data``
    directly in new code.

    Used in Stage 5 to deserialise the fitted transformers pipeline written
    by Stage 3 (``transformers_pipeline.pkl``) for on-the-fly application to
    each 100k-row test chunk during full evaluation.

    Also used in Stage 5 to load trained models written by Stage 4
    (``kmeans_best.pkl``, ``dbscan_best.pkl``) for cluster assignment.

    Security note: only load pickle files from trusted pipeline artifact
    directories. Never load pickles from external or user-supplied sources.

    Covers: Option A and Option B -- both use ``transformers_pipeline.pkl``
    in Stage 5, though the transformers were fitted on different data:
      Option A: fitted on train_rows (~150k) from the combined sample.
      Option B: fitted on train_rows (200k) from the train-only sample.

    Parameters
    ----------
    path : str | Path
        Relative or absolute path to the ``.pkl`` file.
        Resolved against the project root via :func:`resolve_path`.

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

