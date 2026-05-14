# src/crispdm/data/load_utils_data.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple

import pandas as pd

from crispdm.configuration.enum_registry_config import ReadMode, normalize_read_mode
from crispdm.common.logging_adapter_common import get_logger
from crispdm.common.path_service_common import resolve_path

# Initialize logger.
log = get_logger(__name__)

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# Data loading utilities for the CRISP-DM pipeline.
# Provides a unified, strategy-driven interface for loading CSV and Parquet
# files across all pipeline phases (2-5) and for both pipeline options.
#
# Option A (drift-aware):
#   Stage 2 — loads train + test combined, adds 'source' column, samples 200k.
#   Stage 3 — reads sample parquet, splits by 'source', fits transformers on
#             train rows (~150k), applies to both train and test rows.
#   Stage 4 — reads train_prepared_150k.parquet (train only).
#   Stage 5 — quick eval on test_prepared_50k.parquet +
#             full eval on GUIDE_Test.csv (chunked, 100k/iter).
#
# Option B (train-only):
#   Stage 2 — loads train CSV only, samples 200k (no source column).
#   Stage 3 — reads train_sample_200k.parquet, fits and applies on train (200k).
#   Stage 4 — reads train_prepared_200k.parquet (train only).
#   Stage 5 — full eval on GUIDE_Test.csv (chunked, 100k/iter). No quick eval.
#
# The read_strategy block in each pipeline YAML is the single source of truth
# that drives all loading decisions. No code changes are needed when switching
# between Option A and Option B, or between clustering / classification /
# regression / timeseries pipelines — only YAML values change.
#
# YAML structure per stage:
# -----------------------------------------------------------------------------
#   Stage 2: has dataset_input (source_type, train_path, test_path)
#            with read_strategy nested inside dataset_input.
#            → Requires DataSourceConfig (wraps ReadStrategyContract).
#
#   Stages 3-5: have ONLY read_strategy at stage level (no dataset_input).
#               Input paths are inside read_strategy via input_source,
#               input_source_sample, input_source_full.
#               → Require ReadStrategyContract only.
#
# Program flow:
# -----------------------------------------------------------------------------
# - YAML pipeline configuration  → read_strategy block (+ dataset_input for Stage 2)
# - pipeline/* or stage/* → builds ReadStrategyContract from YAML dict
# - Stage 2 only          → builds DataSourceConfig wrapping the contract
# - stage/*               → calls load_* public functions with the contract
# - load_* functions      → dispatch to private CSV / Parquet primitives
# - primitives            → return DataFrame or Generator[DataFrame]
#
# Design patterns
# -----------------------------------------------------------------------------
# - GoF -> Gang of Four:
#   - Strategy: load_by_strategy() dispatches to the correct CSV primitive
#               based on ReadStrategyContract.mode (sample | chunked | full).
#               Each mode maps to a different private function, selected at
#               runtime without conditionals in the caller.
#   - Factory Method: ReadStrategyContract.from_dict() and
#               DataSourceConfig.from_dict() create immutable contracts
#               from raw YAML dicts without exposing constructors directly.
# - DDD -> Domain-Driven Design:
#   - Value Object: ReadStrategyContract and DataSourceConfig are frozen
#               dataclasses -- immutable, equality by value, no identity.
#               They represent the YAML contract as a typed Python object.
# - Enterprise/Architectural:
#   - Data Access Layer (thin):
#       -   Isolates all CSV/Parquet I/O from business logic in stage/* and pipeline/*.
#         No stage touches pd.read_csv directly -- all I/O goes through this module.
#   - Contract / DTO:
#       -   ReadStrategyContract maps the YAML read_strategy block 1-to-1 and is the
#           single source of truth for loading behaviour across all phases, options,
#           and pipeline types.
# =============================================================================


# =============================================================================
# SECTION 1 — DATACLASSES (contracts)
# =============================================================================


@dataclass(frozen=True)
class ReadStrategyContract:
    """
    Immutable contract parsed from the ``read_strategy`` block of a pipeline YAML.

    Every field maps 1-to-1 to a YAML key. The YAML is the single source of
    truth — no defaults are applied in code. Missing required fields raise
    ``KeyError`` immediately so misconfiguration fails at startup, not silently
    mid-run.

    Used for all pipeline types (clustering, classification, regression,
    timeseries) and for both Option A and Option B across all phases.

    Stage-by-stage value summary
    ----------------------------
    .. code-block:: text

        Field                    Stage2-A  Stage2-B  Stage3     Stage4     Stage5
        -------------------------------------------------------------------------
        mode                     sample    sample    sample     sample     chunked
        sample_rows              200_000   200_000   200_000    150_000    null
        sample_method            random    random    random     random     null
        random_state             7         7         7          7          null
        chunksize                100_000   100_000   100_000    100_000    100_000
        combine_before_sampling  True      False     False      False      False
        add_source_column        True      False     False      False      False
        input_source             null      null      *.parquet  *.parquet  null
        input_source_sample      null      null      null       null       A:*.pq
        input_source_full        null      null      null       null       *.csv

    Parameters
    ----------
    mode : ReadMode
        Reading strategy: ``sample`` | ``chunked`` | ``full``.
    sample_rows : int
        Number of rows when ``mode="sample"``.
    sample_method : str
        Sampling algorithm: ``random`` | ``head`` | ``tail``.
    chunksize : int
        Rows per iteration when ``mode="chunked"``.
    random_state : int
        Reproducibility seed for random sampling.
    combine_before_sampling : bool
        **Option A — Stage 2 only.**
        ``True``: train and test CSVs are concatenated before sampling.
        ``False`` (Option B): only train CSV is loaded.
    add_source_column : bool
        **Option A — Stage 2 only.**
        ``True``: ``'source'`` column (``'train'``/``'test'``) added before
        concat to enable drift detection (Stage 2) and row filtering (Stage 3).
        ``False`` (Option B): no column added.
    input_source : Optional[str]
        **Stages 3 and 4.**
        Path to intermediate Parquet from previous stage.
        ``None`` for Stage 2 (reads raw CSV) and Stage 5.
    input_source_sample : Optional[str]
        **Stage 5 — Option A only.**
        Pre-prepared test Parquet for quick evaluation (seconds).
        ``None`` for Option B (no quick eval path).
    input_source_full : Optional[str]
        **Stage 5 — both options.**
        Full raw test CSV for chunked evaluation (minutes).
    """

    # Core reading parameters — all phases, both options
    mode: ReadMode
    sample_rows: int
    sample_method: str
    chunksize: int
    random_state: int

    # Combination and source-tracking — Stage 2, Option A specific
    combine_before_sampling: bool  # Option A=True  | Option B=False
    add_source_column: bool  # Option A=True  | Option B=False

    # Intermediate input paths — Stages 3-5
    input_source: Optional[str]  # Stage 3-4 parquet
    input_source_sample: Optional[str]  # Stage 5 quick eval (Option A only)
    input_source_full: Optional[str]  # Stage 5 full eval (both options)

    @classmethod
    def from_dict(
            cls,
            raw: dict[str, Any],
    ) -> "ReadStrategyContract":  # noqa: UP006
        """
        Build a ``ReadStrategyContract`` from a raw YAML ``read_strategy`` dict.

        The YAML is the single source of truth — all required fields must be
        present. Optional fields may be absent or ``null`` in the YAML.
        Raises ``KeyError`` for missing required fields.

        Covers: Option A and Option B — same factory for every stage.

        Parameters
        ----------
        raw : dict
            Dictionary from the YAML ``read_strategy`` block of a stage.

        Returns
        -------
        ReadStrategyContract
            Fully populated, immutable contract.

        Raises
        ------
        KeyError
            If any required field is absent from *raw*.
        ValueError
            If ``mode`` is not a recognised ``ReadMode`` value.
        """
        # Step 1: Log raw keys for traceability before any parsing.
        log.debug(
            "[ReadStrategyContract.from_dict] raw keys=%s",
            list(raw.keys()),
        )

        # Step 2: Parse mode — raises ValueError for unrecognised values.
        mode: ReadMode = normalize_read_mode(raw["mode"])

        # Step 3: Extract required integer fields.
        # Raises KeyError if any field is absent — YAML must declare them all.
        sample_rows: int = int(raw["sample_rows"])
        chunksize: int = int(raw["chunksize"])
        random_state: int = int(raw["random_state"])

        # Step 4: Extract required string field.
        sample_method: str = str(raw["sample_method"])

        # Step 5: Extract required boolean flags.
        # These are the structural difference between Option A and Option B.
        combine_before_sampling: bool = bool(raw["combine_before_sampling"])
        add_source_column: bool = bool(raw["add_source_column"])

        # Step 6: Extract optional path fields — None when absent or null.
        # Stages 3-4 set input_source; Stage 5 sets input_source_sample/full.
        input_source: Optional[str] = raw.get("input_source") or None
        input_source_sample: Optional[str] = raw.get("input_source_sample") or None
        input_source_full: Optional[str] = raw.get("input_source_full") or None

        # Step 7: Log fully resolved values to aid YAML misconfiguration debugging.
        log.info(
            "[ReadStrategyContract.from_dict] resolved — "
            "mode=%s sample_rows=%d sample_method=%s chunksize=%d "
            "random_state=%d combine=%s add_source=%s "
            "input_source=%s input_source_sample=%s input_source_full=%s",
            mode.value,
            sample_rows,
            sample_method,
            chunksize,
            random_state,
            combine_before_sampling,
            add_source_column,
            input_source,
            input_source_sample,
            input_source_full,
        )

        # Step 8: Build and return the immutable contract.
        return cls(
            mode=mode,
            sample_rows=sample_rows,
            sample_method=sample_method,
            chunksize=chunksize,
            random_state=random_state,
            combine_before_sampling=combine_before_sampling,
            add_source_column=add_source_column,
            input_source=input_source,
            input_source_sample=input_source_sample,
            input_source_full=input_source_full,
        )


@dataclass(frozen=True)
class DataSourceConfig:
    """
    Immutable configuration parsed from the ``dataset_input`` block of Stage 2.

    **Only Stage 2 has a ``dataset_input`` block.**
    Stages 3-5 declare their input paths inside ``read_strategy``
    (``input_source``, ``input_source_sample``, ``input_source_full``)
    and use ``ReadStrategyContract`` directly — no ``DataSourceConfig`` needed.

    Encapsulates *what* to load (source type and CSV paths) together with
    *how* to load it (``ReadStrategyContract``). Passed as a single unit
    to Stage 2 loading functions.

    Parameters
    ----------
    source_type : str
        Dataset origin for Stage 2:

        - ``"csv_combined"`` **(Option A)**: train and test CSVs are loaded
          together, combined, ``'source'`` column added, 200k rows sampled
          from both to enable drift detection in Stage 2 and source-based
          filtering in Stage 3.
        - ``"csv_separate"`` **(Option B)**: only ``train_path`` is loaded
          and sampled. Test is not seen until Stage 5 (chunked evaluation).

    train_path : str
        Path to the training CSV. Required for both options.
    test_path : Optional[str]
        Path to the test CSV.
        Required for Option A (``csv_combined``).
        ``None`` for Option B — test is deferred to Stage 5.
    strategy : ReadStrategyContract
        Reading contract from the ``read_strategy`` block nested inside
        ``dataset_input`` (Stage 2 structure).
    """

    source_type: str  # "csv_combined" (A) | "csv_separate" (B)
    train_path: str  # required for both options
    test_path: Optional[str]  # Option A=present | Option B=None
    strategy: ReadStrategyContract

    @classmethod
    def from_dict(
            cls,
            dataset_input: dict[str, Any],
    ) -> "DataSourceConfig":  # noqa: UP006
        """
        Build a ``DataSourceConfig`` from the raw YAML ``dataset_input`` dict.

        The ``read_strategy`` block must be nested inside ``dataset_input``
        (Stage 2 structure). Raises ``KeyError`` for missing required fields
        and ``ValueError`` for unsupported ``source_type`` values.

        Covers: Option A (``csv_combined``) and Option B (``csv_separate``).

        Parameters
        ----------
        dataset_input : dict
            Dictionary from the YAML ``dataset_input`` block of Stage 2,
            containing ``source_type``, ``train_path``, optionally
            ``test_path``, and the nested ``read_strategy`` dict.

        Returns
        -------
        DataSourceConfig
            Fully populated, immutable source configuration.

        Raises
        ------
        KeyError
            If ``source_type``, ``train_path``, ``read_strategy``, or any
            required ``read_strategy`` field is absent.
        ValueError
            If ``source_type`` is not ``"csv_combined"`` or ``"csv_separate"``.
        """
        # Step 1: Log raw keys for traceability.
        log.debug(
            "[DataSourceConfig.from_dict] raw keys=%s",
            list(dataset_input.keys()),
        )

        # Step 2: Extract and validate source_type.
        # Parquet and other formats are not valid here — Stage 2 always reads CSV.
        source_type: str = str(dataset_input["source_type"])
        _valid: set[str] = {"csv_combined", "csv_separate"}
        if source_type not in _valid:
            log.error(
                "[DataSourceConfig.from_dict] unsupported source_type='%s' "
                "— valid values for Stage 2: %s. "
                "Parquet inputs for Stages 3-5 are declared via "
                "read_strategy.input_source, not here.",
                source_type,
                _valid,
            )
            raise ValueError(
                f"Unsupported source_type='{source_type}'. "
                f"Stage 2 only supports {_valid}. "
                f"Parquet inputs (Stages 3-5) go in read_strategy.input_source."
            )

        # Step 3: Extract train_path — required for both options.
        train_path: str = str(dataset_input["train_path"])

        # Step 4: Extract test_path — required for Option A, absent for Option B.
        test_path: Optional[str] = dataset_input.get("test_path") or None

        # Step 5: Validate Option A consistency.
        if source_type == "csv_combined" and test_path is None:
            log.error(
                "[DataSourceConfig.from_dict] source_type='csv_combined' "
                "(Option A) requires test_path — it is null or absent. "
                "Stage 2 drift detection cannot proceed."
            )
            raise KeyError(
                "source_type='csv_combined' (Option A) requires 'test_path' "
                "in dataset_input but it is null or absent."
            )

        # Step 6: Log Option B intent — test is intentionally deferred.
        if source_type == "csv_separate":
            log.info(
                "[DataSourceConfig.from_dict] source_type='csv_separate' "
                "(Option B) — test data deferred to Stage 5 chunked evaluation. "
                "No drift detection performed in Stage 2."
            )

        # Step 7: Build ReadStrategyContract from the nested read_strategy block.
        strategy: ReadStrategyContract = ReadStrategyContract.from_dict(
            dataset_input["read_strategy"]
        )

        # Step 8: Log final resolved configuration.
        log.info(
            "[DataSourceConfig.from_dict] resolved — "
            "source_type=%s train_path=%s test_path=%s "
            "mode=%s combine=%s add_source=%s",
            source_type,
            train_path,
            test_path,
            strategy.mode.value,
            strategy.combine_before_sampling,
            strategy.add_source_column,
        )

        # Step 9: Build and return the immutable configuration.
        return cls(
            source_type=source_type,
            train_path=train_path,
            test_path=test_path,
            strategy=strategy,
        )


# =============================================================================
# SECTION 2 — PRIVATE CSV PRIMITIVES
# =============================================================================
# All functions in this section are internal to this module (prefix _).
# They are thin I/O wrappers: no business logic, no strategy decisions.
# Strategy decisions (which function to call) are made by the public
# dispatcher load_by_strategy() in Section 3.
#
# Called exclusively by load_by_strategy() and load_csv_combined().
#
# Coverage:
#   _load_csv_head()          → both options, all phases (head sampling)
#   _load_csv_random_sample() → both options, all phases (random sampling)
#   _load_csv_chunks()        → both options, Stage 5 (chunked evaluation)
#   _load_csv_full()          → both options (use with caution on large files)
#   _add_source_column()      → Option A only, Stage 2
#   _sample_combined()        → Option A only, Stage 2
# =============================================================================


def _load_csv_head(
        path: str,
        *,
        csv_params: Optional[Dict[str, Any]],
        sample_rows: int,
) -> pd.DataFrame:
    """
    Load the first N rows of a CSV file in row order (deterministic).

    Does not load the entire file — uses ``nrows`` to stop reading after
    ``sample_rows`` rows. Suitable when the dataset has no temporal or
    category bias in row ordering, or when exact reproducibility without
    a random seed is required.

    Not a statistical sample: always returns rows 0..N-1.
    For a representative random sample use ``_load_csv_random_sample()``.

    Covers: Option A and Option B — called when ``sample_method="head"``.

    Parameters
    ----------
    path : str
        Relative or absolute path to the CSV file.
        Resolved against the project root via ``resolve_path()``.
    csv_params : Optional[Dict[str, Any]]
        Extra keyword arguments forwarded to ``pd.read_csv()``
        (e.g. ``sep``, ``encoding``, ``low_memory``).
        Pass ``None`` to use pandas defaults.
    sample_rows : int
        Number of rows to read from the start of the file.

    Returns
    -------
    pd.DataFrame
        DataFrame with at most ``sample_rows`` rows and all columns.

    Raises
    ------
    FileNotFoundError
        If the resolved path does not exist.
    pd.errors.ParserError
        If the CSV cannot be parsed with the given parameters.
    Exception
        Any other ``pd.read_csv()`` exception is logged then re-raised.
    """
    # Step 1: Resolve path to absolute location using project root.
    resolved = resolve_path(path)
    log.debug("[_load_csv_head] resolved path=%s sample_rows=%d", resolved, sample_rows)

    # Step 2: Merge csv_params with defaults (empty dict if None).
    params: Dict[str, Any] = dict(csv_params or {})
    log.debug(
        "[_load_csv_head] csv_params=%s",
        json.dumps({k: str(v) for k, v in params.items()}, ensure_ascii=False),
    )

    # Step 3: Load first N rows — granular error handling for clear diagnostics.
    try:
        log.info(
            "[_load_csv_head] loading first %d rows from path=%s",
            sample_rows,
            resolved,
        )
        df: pd.DataFrame = pd.read_csv(resolved, nrows=sample_rows, **params)

    except FileNotFoundError:
        log.error("[_load_csv_head] file not found path=%s", resolved)
        raise

    except pd.errors.ParserError:
        log.error(
            "[_load_csv_head] CSV parse error path=%s params=%s",
            resolved,
            params,
        )
        raise

    except Exception:
        log.exception(
            "[_load_csv_head] unexpected error path=%s sample_rows=%d",
            resolved,
            sample_rows,
        )
        raise

    # Step 4: Log result dimensions for quick sanity check.
    log.info(
        "[_load_csv_head] loaded rows=%d cols=%d path=%s",
        len(df),
        df.shape[1],
        resolved,
    )
    return df


def _load_csv_random_sample(
        path: str,
        *,
        csv_params: Optional[Dict[str, Any]],
        sample_rows: int,
        chunksize: int,
        random_state: int,
) -> pd.DataFrame:
    """
    Load a representative random sample from a large CSV without full memory load.

    Strategy: read the file in chunks of ``chunksize`` rows, take a
    proportional sub-sample from each chunk, concatenate, then down-sample
    to exactly ``sample_rows``. Memory peak = one chunk at a time
    (≈ ``chunksize`` rows), never the full file.

    Why chunked and not full-load + sample:
        Loading a 2 GB file to then drop 99 % of rows wastes 2 GB of RAM.
        The chunked approach keeps peak memory proportional to ``chunksize``.

    Covers: Option A and Option B — called when ``sample_method="random"``.

    Parameters
    ----------
    path : str
        Relative or absolute path to the CSV file.
    csv_params : Optional[Dict[str, Any]]
        Extra keyword arguments forwarded to ``pd.read_csv()``.
    sample_rows : int
        Target number of rows in the returned DataFrame.
    chunksize : int
        Number of rows read per chunk iteration.
    random_state : int
        Seed for the random sampler — ensures reproducibility across runs.

    Returns
    -------
    pd.DataFrame
        DataFrame with exactly ``sample_rows`` rows (or fewer if the file
        has fewer rows than requested) and all columns.

    Raises
    ------
    FileNotFoundError
        If the resolved path does not exist.
    Exception
        Any ``pd.read_csv()`` exception is logged then re-raised.
    """
    # Step 1: Resolve path to absolute location.
    resolved = resolve_path(path)
    log.debug(
        "[_load_csv_random_sample] resolved path=%s sample_rows=%d "
        "chunksize=%d random_state=%d",
        resolved,
        sample_rows,
        chunksize,
        random_state,
    )

    # Step 2: Merge csv_params with defaults.
    params: Dict[str, Any] = dict(csv_params or {})
    log.debug(
        "[_load_csv_random_sample] csv_params=%s",
        json.dumps({k: str(v) for k, v in params.items()}, ensure_ascii=False),
    )

    # Step 3: Validate file existence before starting the chunked iteration.
    if not resolved.exists():
        log.error("[_load_csv_random_sample] file not found path=%s", resolved)
        raise FileNotFoundError(f"CSV file not found: {resolved}")

    # Step 4: Iterate chunks and collect a proportional sub-sample from each.
    samples: list[pd.DataFrame] = []
    rows_seen: int = 0

    log.info(
        "[_load_csv_random_sample] starting chunked sampling "
        "path=%s chunksize=%d target=%d",
        resolved,
        chunksize,
        sample_rows,
    )

    try:
        for chunk in pd.read_csv(resolved, chunksize=chunksize, **params):
            rows_seen += len(chunk)

            # Take at most sample_rows rows from each chunk proportionally.
            n: int = min(len(chunk), sample_rows)
            samples.append(chunk.sample(n=n, random_state=random_state))

    except Exception:
        log.exception(
            "[_load_csv_random_sample] error during chunked read path=%s rows_seen=%d",
            resolved,
            rows_seen,
        )
        raise

    log.debug(
        "[_load_csv_random_sample] chunked pass complete "
        "rows_seen=%d chunks_collected=%d",
        rows_seen,
        len(samples),
    )

    # Step 5: Concatenate all chunk sub-samples into one DataFrame.
    df: pd.DataFrame = pd.concat(samples, ignore_index=True)

    # Step 6: Down-sample to exactly sample_rows if over-sampled.
    # This happens when the file has many small chunks each contributing N rows.
    if len(df) > sample_rows:
        log.debug(
            "[_load_csv_random_sample] down-sampling concat rows=%d → target=%d",
            len(df),
            sample_rows,
        )
        df = df.sample(n=sample_rows, random_state=random_state).reset_index(drop=True)

    # Step 7: Log final result dimensions.
    log.info(
        "[_load_csv_random_sample] done rows=%d cols=%d path=%s",
        len(df),
        df.shape[1],
        resolved,
    )
    return df


def _load_csv_chunks(
        path: str,
        *,
        csv_params: Optional[Dict[str, Any]],
        chunksize: int,
) -> Generator[pd.DataFrame, None, None]:
    """
    Yield a CSV file in fixed-size chunks without loading it fully into memory.

    Returns a generator; the file is read lazily one chunk at a time.
    The caller is responsible for processing each yielded DataFrame before
    requesting the next, keeping memory usage bounded to ``chunksize`` rows
    at any moment.

    Primary use case:
        Stage 5 full evaluation — GUIDE_Test.csv (6M rows, 1 GB) is read
        100k rows at a time; transformers are applied on-the-fly per chunk.
        Both Option A and Option B use this path for Stage 5 full evaluation.

    Covers: Option A and Option B — Stage 5 full evaluation (``mode="chunked"``).

    Parameters
    ----------
    path : str
        Relative or absolute path to the CSV file.
    csv_params : Optional[Dict[str, Any]]
        Extra keyword arguments forwarded to ``pd.read_csv()``.
    chunksize : int
        Number of rows per yielded chunk.

    Yields
    ------
    pd.DataFrame
        DataFrame containing at most ``chunksize`` rows. The last chunk may
        contain fewer rows if the file size is not a multiple of ``chunksize``.

    Raises
    ------
    FileNotFoundError
        If the resolved path does not exist (raised before iteration starts).
    Exception
        Any ``pd.read_csv()`` initialisation exception is logged then re-raised.
    """
    # Step 1: Resolve path to absolute location.
    resolved = resolve_path(path)
    log.debug(
        "[_load_csv_chunks] resolved path=%s chunksize=%d",
        resolved,
        chunksize,
    )

    # Step 2: Merge csv_params with defaults.
    params: Dict[str, Any] = dict(csv_params or {})
    log.debug(
        "[_load_csv_chunks] csv_params=%s",
        json.dumps({k: str(v) for k, v in params.items()}, ensure_ascii=False),
    )

    # Step 3: Validate file existence before initialising the reader.
    if not resolved.exists():
        log.error("[_load_csv_chunks] file not found path=%s", resolved)
        raise FileNotFoundError(f"CSV file not found: {resolved}")

    # Step 4: Initialise the pandas TextFileReader (lazy — no rows read yet).
    try:
        log.info(
            "[_load_csv_chunks] initialising chunked reader path=%s chunksize=%d",
            resolved,
            chunksize,
        )
        reader = pd.read_csv(resolved, chunksize=chunksize, **params)

    except Exception:
        log.exception(
            "[_load_csv_chunks] failed to initialise reader path=%s",
            resolved,
        )
        raise

    # Step 5: Yield chunks lazily — memory peak stays at one chunk at a time.
    chunk_index: int = 0
    for chunk in reader:
        log.debug(
            "[_load_csv_chunks] yielding chunk=%d rows=%d",
            chunk_index,
            len(chunk),
        )
        yield chunk
        chunk_index += 1

    log.info(
        "[_load_csv_chunks] iteration complete total_chunks=%d path=%s",
        chunk_index,
        resolved,
    )


def _load_csv_full(
        path: str,
        *,
        csv_params: Optional[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Load an entire CSV file into memory as a single DataFrame.

    Issues a warning for files larger than 500 MB because full loading
    of large files may exhaust available RAM. For large files prefer
    ``_load_csv_random_sample()`` (profiling) or ``_load_csv_chunks()``
    (evaluation).

    Covers: Option A and Option B — available for all phases, but rarely
    used in this pipeline due to GUIDE dataset size (2 GB train, 1 GB test).

    Parameters
    ----------
    path : str
        Relative or absolute path to the CSV file.
    csv_params : Optional[Dict[str, Any]]
        Extra keyword arguments forwarded to ``pd.read_csv()``.

    Returns
    -------
    pd.DataFrame
        DataFrame containing all rows and columns from the CSV file.

    Raises
    ------
    FileNotFoundError
        If the resolved path does not exist.
    Exception
        Any ``pd.read_csv()`` exception is logged then re-raised.
    """
    # Step 1: Resolve path to absolute location.
    resolved = resolve_path(path)
    log.debug("[_load_csv_full] resolved path=%s", resolved)

    # Step 2: Merge csv_params with defaults.
    params: Dict[str, Any] = dict(csv_params or {})
    log.debug(
        "[_load_csv_full] csv_params=%s",
        json.dumps({k: str(v) for k, v in params.items()}, ensure_ascii=False),
    )

    # Step 3: Validate file existence.
    if not resolved.exists():
        log.error("[_load_csv_full] file not found path=%s", resolved)
        raise FileNotFoundError(f"CSV file not found: {resolved}")

    # Step 4: Warn on large files — full load may exhaust RAM.
    file_size_mb: float = resolved.stat().st_size / (1024**2)
    if file_size_mb > 500:
        log.warning(
            "[_load_csv_full] LARGE FILE: %.0f MB — full load may require "
            "~%.0f MB RAM. Consider mode='sample' for profiling or "
            "mode='chunked' for Stage 5 evaluation.",
            file_size_mb,
            file_size_mb * 3.5,
            )

    # Step 5: Load entire file into memory.
    try:
        log.info(
            "[_load_csv_full] loading full file path=%s size_mb=%.1f",
            resolved,
            file_size_mb,
        )
        df: pd.DataFrame = pd.read_csv(resolved, **params)

    except Exception:
        log.exception("[_load_csv_full] read_csv failed path=%s", resolved)
        raise

    # Step 6: Log result dimensions.
    log.info(
        "[_load_csv_full] loaded rows=%d cols=%d path=%s",
        len(df),
        df.shape[1],
        resolved,
    )
    return df


# =============================================================================
# SECTION 3 — PRIVATE HELPERS (Option A Stage 2 specific)
# =============================================================================


def _add_source_column(
        df: pd.DataFrame,
        source_value: str,
) -> pd.DataFrame:
    """
    Add a ``'source'`` column to a DataFrame to mark its dataset origin.

    **Option A — Stage 2 only.**
    Called once for the train DataFrame (``source_value="train"``) and once
    for the test DataFrame (``source_value="test"``) before concatenation.
    The resulting ``'source'`` column enables:

    - Drift detection in Stage 2 (compare distributions by source).
    - Row filtering in Stage 3 (separate df_train ≈ 150k and df_test ≈ 50k).

    Not called in Option B (no source tracking needed).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to annotate. Modified in place is avoided —
        a copy is returned to keep the function side-effect-free.
    source_value : str
        Label to assign to every row: ``"train"`` or ``"test"``.

    Returns
    -------
    pd.DataFrame
        New DataFrame identical to *df* with an additional ``'source'``
        column containing ``source_value`` for every row.

    Raises
    ------
    ValueError
        If ``source_value`` is not ``"train"`` or ``"test"``.
    """
    # Step 1: Validate source_value — only 'train' and 'test' are allowed.
    _valid_sources = {"train", "test"}
    if source_value not in _valid_sources:
        log.error(
            "[_add_source_column] invalid source_value='%s' — expected %s",
            source_value,
            _valid_sources,
        )
        raise ValueError(
            f"source_value='{source_value}' is not valid. "
            f"Expected one of {_valid_sources}."
        )

    # Step 2: Warn if 'source' column already exists — prevents silent overwrite.
    if "source" in df.columns:
        log.warning(
            "[_add_source_column] 'source' column already exists — "
            "overwriting with source_value='%s' rows=%d",
            source_value,
            len(df),
        )

    # Step 3: Assign the source column and return a new DataFrame.
    log.debug(
        "[_add_source_column] adding source='%s' to rows=%d",
        source_value,
        len(df),
    )
    result: pd.DataFrame = df.assign(source=source_value)

    log.info(
        "[_add_source_column] done source='%s' rows=%d cols=%d",
        source_value,
        len(result),
        result.shape[1],
    )
    return result


def _sample_combined(
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        *,
        sample_rows: int,
        random_state: int,
) -> pd.DataFrame:
    """
    Combine train and test DataFrames and draw a stratified random sample.

    **Option A — Stage 2 only.**
    Implements the common of the Option A data strategy:

    1. Both DataFrames must already have a ``'source'`` column
       (added by ``_add_source_column()`` before this call).
    2. Concatenates train + test into one combined DataFrame (≈ 19M rows
       for GUIDE: 13M train + 6M test).
    3. Draws ``sample_rows`` random rows from the combined DataFrame so that
       both datasets are proportionally represented (≈ 150k train, ≈ 50k test
       for a 200k sample from a 13M/6M split).

    The proportional representation is a natural consequence of random sampling
    from the combined DataFrame — no explicit stratification is applied.

    Not called in Option B (train is sampled independently).

    Parameters
    ----------
    df_train : pd.DataFrame
        Train DataFrame with ``'source'`` column already set to ``"train"``.
    df_test : pd.DataFrame
        Test DataFrame with ``'source'`` column already set to ``"test"``.
    sample_rows : int
        Target number of rows in the returned sample (e.g. 200_000).
    random_state : int
        Seed for the random sampler — ensures reproducibility.

    Returns
    -------
    pd.DataFrame
        Combined and sampled DataFrame with ``sample_rows`` rows (or fewer if
        the combined dataset is smaller than requested) and a ``'source'``
        column identifying each row's origin.

    Raises
    ------
    ValueError
        If ``'source'`` column is missing from either input DataFrame,
        indicating ``_add_source_column()`` was not called beforehand.
    """
    # Step 1: Validate that both DataFrames have the 'source' column.
    for name, df in [("df_train", df_train), ("df_test", df_test)]:
        if "source" not in df.columns:
            log.error(
                "[_sample_combined] '%s' is missing 'source' column — "
                "call _add_source_column() before _sample_combined().",
                name,
            )
            raise ValueError(
                f"'{name}' is missing the 'source' column. "
                f"Call _add_source_column() on both DataFrames first."
            )

    # Step 2: Log input sizes for memory and proportion awareness.
    log.info(
        "[_sample_combined] combining train rows=%d and test rows=%d "
        "(combined=%d) → target sample=%d random_state=%d",
        len(df_train),
        len(df_test),
        len(df_train) + len(df_test),
        sample_rows,
        random_state,
        )

    # Step 3: Concatenate train and test into one combined DataFrame.
    # ignore_index=True resets the integer index after concat.
    df_combined: pd.DataFrame = pd.concat(
        [df_train, df_test],
        ignore_index=True,
    )
    log.debug(
        "[_sample_combined] concat complete combined rows=%d cols=%d",
        len(df_combined),
        df_combined.shape[1],
    )

    # Step 4: Draw random sample from the combined DataFrame.
    # If the combined size is smaller than sample_rows, return all rows.
    if len(df_combined) <= sample_rows:
        log.warning(
            "[_sample_combined] combined rows=%d <= sample_rows=%d — "
            "returning all rows without sampling.",
            len(df_combined),
            sample_rows,
        )
        return df_combined.reset_index(drop=True)

    df_sample: pd.DataFrame = df_combined.sample(
        n=sample_rows, random_state=random_state
    ).reset_index(drop=True)

    # Step 5: Log resulting source distribution for drift-detection audit trail.
    source_counts = df_sample["source"].value_counts().to_dict()
    log.info(
        "[_sample_combined] sample complete rows=%d source_distribution=%s",
        len(df_sample),
        source_counts,
    )

    return df_sample


# =============================================================================
# SECTION 4 - PUBLIC PARQUET READING (I)
# Only reading lives here. Writing and pickle operations are in
# persist_utils_data.py (Single Responsibility Principle).
#
# Imported at the top of this module and re-exported via noqa:F401
# so existing callers can still do:
#   from crispdm.data.load_utils_data import save_parquet   # still works
#   from crispdm.data.persist_utils_data import save_parquet # preferred
#
# Persistence functions available via import above:
#   save_parquet(df, path, compression)  → Stages 2-3 intermediate parquets
#   save_pickle(obj, path)               → Stages 3-4 transformers + models
#   load_pickle(path)                    → Stage 5 load transformers + models
# =============================================================================


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


# =============================================================================
# SECTION 5 - PUBLIC DISPATCHER
# =============================================================================


def load_by_strategy(
        path: str | Path,
        *,
        csv_params: Optional[Dict[str, Any]] = None,
        strategy: ReadStrategyContract,
) -> Tuple[
    Optional[pd.DataFrame],
    Optional[Generator[pd.DataFrame, None, None]],
    ReadStrategyContract,
]:
    """
    Load a CSV file according to the reading contract from the pipeline YAML.

    Central dispatcher that reads strategy.mode and strategy.sample_method
    to select the correct primitive. Returns a consistent tuple regardless
    of mode so callers have a uniform interface.

    Covers: Option A and Option B -- same dispatcher for all phases.

    Parameters
    ----------
    path : str | Path
        Relative or absolute path to the CSV file.
    csv_params : Optional[Dict[str, Any]]
        Extra kwargs forwarded to pd.read_csv().
    strategy : ReadStrategyContract
        Fully resolved reading contract.

    Returns
    -------
    Tuple[Optional[pd.DataFrame], Optional[Generator], ReadStrategyContract]
        (DataFrame, None, strategy) for mode=sample or full.
        (None, Generator, strategy) for mode=chunked.

    Raises
    ------
    ValueError
        If strategy.mode is not a recognised ReadMode value.
    FileNotFoundError
        If the resolved path does not exist.
    """
    log.info(
        "[load_by_strategy] path=%s mode=%s sample_method=%s "
        "sample_rows=%d chunksize=%d",
        path,
        strategy.mode.value,
        strategy.sample_method,
        strategy.sample_rows,
        strategy.chunksize,
    )

    # Step 1: Dispatch to random sample primitive.
    if strategy.mode == ReadMode.SAMPLE and strategy.sample_method == "random":
        log.info("[load_by_strategy] -> _load_csv_random_sample")
        df = _load_csv_random_sample(
            str(path),
            csv_params=csv_params,
            sample_rows=strategy.sample_rows,
            chunksize=strategy.chunksize,
            random_state=strategy.random_state,
        )
        return df, None, strategy

    # Step 2: Dispatch to head sample primitive.
    if strategy.mode == ReadMode.SAMPLE and strategy.sample_method in {"head", "tail"}:
        if strategy.sample_method == "tail":
            log.warning(
                "[load_by_strategy] sample_method='tail' not supported -- "
                "falling back to head sampling."
            )
        log.info("[load_by_strategy] -> _load_csv_head")
        df = _load_csv_head(
            str(path), csv_params=csv_params, sample_rows=strategy.sample_rows
        )
        return df, None, strategy

    # Step 3: Dispatch to chunked generator primitive.
    if strategy.mode == ReadMode.CHUNKED:
        log.info(
            "[load_by_strategy] -> _load_csv_chunks chunksize=%d", strategy.chunksize
        )
        generator = _load_csv_chunks(
            str(path), csv_params=csv_params, chunksize=strategy.chunksize
        )
        return None, generator, strategy

    # Step 4: Dispatch to full load primitive.
    if strategy.mode == ReadMode.FULL:
        log.warning("[load_by_strategy] mode=full -- ensure RAM > file size x 3.5")
        df = _load_csv_full(str(path), csv_params=csv_params)
        return df, None, strategy

    # Step 5: Unrecognised mode -- raise explicitly.
    log.error("[load_by_strategy] unrecognised mode=%s path=%s", strategy.mode, path)
    raise ValueError(f"Unrecognised ReadMode: {strategy.mode}")


def load_with_origin(
        path: str | Path,
        *,
        csv_params: Optional[Dict[str, Any]] = None,
        strategy: ReadStrategyContract,
) -> Tuple[pd.DataFrame, Path]:
    """
    Load a CSV file and return both the DataFrame and its resolved path.

    Convenience wrapper around load_by_strategy() for callers that also
    need the resolved path for logging or artifact registration.
    Only valid for mode=sample or mode=full.

    Covers: Option A and Option B.

    Parameters
    ----------
    path : str | Path
        Relative or absolute path to the CSV file.
    csv_params : Optional[Dict[str, Any]]
        Extra kwargs forwarded to pd.read_csv().
    strategy : ReadStrategyContract
        Fully resolved reading contract.

    Returns
    -------
    Tuple[pd.DataFrame, Path]
        DataFrame and resolved absolute path of the source CSV.

    Raises
    ------
    ValueError
        If strategy.mode is chunked (returns generator, not DataFrame).
    """
    # Step 1: Guard against chunked mode.
    if strategy.mode == ReadMode.CHUNKED:
        log.error(
            "[load_with_origin] mode=chunked incompatible -- "
            "use load_by_strategy() and iterate the returned generator."
        )
        raise ValueError(
            "load_with_origin() does not support mode=chunked. "
            "Use load_by_strategy() and iterate the returned generator."
        )

    # Step 2: Resolve path for return value.
    resolved: Path = resolve_path(path)
    log.debug("[load_with_origin] resolved path=%s", resolved)

    # Step 3: Delegate to dispatcher.
    log.info("[load_with_origin] delegating to load_by_strategy path=%s", resolved)
    df, _, _ = load_by_strategy(path, csv_params=csv_params, strategy=strategy)

    # Step 4: Guarantee non-None DataFrame.
    if df is None:
        df = pd.DataFrame()
        log.warning(
            "[load_with_origin] dispatcher returned None -- returning empty DataFrame"
        )

    # Step 5: Log and return.
    log.info(
        "[load_with_origin] done rows=%d cols=%d path=%s",
        len(df),
        df.shape[1],
        resolved,
    )
    return df, resolved


# =============================================================================
# SECTION 6 - PUBLIC LOADERS (stage and option specific)
# =============================================================================


def load_train_only(
        path: str | Path,
        *,
        csv_params: Optional[Dict[str, Any]] = None,
        strategy: ReadStrategyContract,
) -> Tuple[pd.DataFrame, Path]:
    """
    Load only the training CSV file according to the reading contract.

    Option B Stage 2. Test dataset not loaded until Stage 5.
    Produces the input to save_parquet(train_sample_200k.parquet).

    Also used internally by load_csv_combined() for Option A.

    Parameters
    ----------
    path : str | Path
        Relative or absolute path to the training CSV.
    csv_params : Optional[Dict[str, Any]]
        Extra kwargs forwarded to pd.read_csv().
    strategy : ReadStrategyContract
        Reading contract with combine_before_sampling=False
        and add_source_column=False for Option B.

    Returns
    -------
    Tuple[pd.DataFrame, Path]
        Sampled train DataFrame and resolved absolute path.

    Raises
    ------
    FileNotFoundError
        If the training CSV does not exist.
    """
    # Step 1: Log intent with option context.
    log.info(
        "[load_train_only] Option B Stage 2 -- test deferred to Stage 5 path=%s", path
    )

    # Step 2: Delegate to load_with_origin.
    df, resolved = load_with_origin(path, csv_params=csv_params, strategy=strategy)

    # Step 3: Log result.
    log.info(
        "[load_train_only] done rows=%d cols=%d path=%s (no source column)",
        len(df),
        df.shape[1],
        resolved,
    )
    return df, resolved


def load_test_only(
        path: str | Path,
        *,
        csv_params: Optional[Dict[str, Any]] = None,
        strategy: ReadStrategyContract,
) -> Tuple[
    Optional[pd.DataFrame],
    Optional[Generator[pd.DataFrame, None, None]],
    Path,
]:
    """
    Load the test CSV file according to the reading contract.

    Stage 5 both options (full chunked evaluation). GUIDE_Test.csv (6M rows)
    read in 100k-row chunks. Transformers from Stage 3 applied on-the-fly.

    Returns a generator when strategy.mode=chunked (Stage 5 standard)
    or a DataFrame when strategy.mode=sample (rarely needed).

    Parameters
    ----------
    path : str | Path
        Relative or absolute path to the test CSV.
    csv_params : Optional[Dict[str, Any]]
        Extra kwargs forwarded to pd.read_csv().
    strategy : ReadStrategyContract
        Reading contract with mode=chunked for Stage 5.

    Returns
    -------
    Tuple[Optional[pd.DataFrame], Optional[Generator], Path]
        (None, Generator, path) when mode=chunked (Stage 5 standard).
        (DataFrame, None, path) when mode=sample or full.

    Raises
    ------
    FileNotFoundError
        If the test CSV does not exist.
    """
    # Step 1: Resolve path for return value.
    resolved: Path = resolve_path(path)
    log.info(
        "[load_test_only] Stage 5 full evaluation mode=%s chunksize=%d path=%s",
        strategy.mode.value,
        strategy.chunksize,
        resolved,
    )

    # Step 2: Dispatch through load_by_strategy.
    df, generator, _ = load_by_strategy(path, csv_params=csv_params, strategy=strategy)

    # Step 3: Log result.
    if generator is not None:
        log.info(
            "[load_test_only] chunked generator ready chunksize=%d path=%s",
            strategy.chunksize,
            resolved,
        )
    else:
        log.info(
            "[load_test_only] loaded rows=%d cols=%d path=%s",
            len(df) if df is not None else 0,
            df.shape[1] if df is not None else 0,
            resolved,
        )

    return df, generator, resolved


def load_csv_combined(
        train_path: str | Path,
        test_path: str | Path,
        *,
        csv_params: Optional[Dict[str, Any]] = None,
        strategy: ReadStrategyContract,
) -> pd.DataFrame:
    """
    Load train and test CSVs, add source tracking, combine, and sample.

    Option A Stage 2 only. Implements the full Option A ingestion flow:
      1. Load train CSV (chunked random sample).
      2. Load test CSV (chunked random sample).
      3. Add source column to each (train / test).
      4. Combine into one DataFrame.
      5. Sample strategy.sample_rows rows randomly.

    Output feeds save_parquet(sample_200k_with_source.parquet).
    Not called in Option B -- use load_train_only() instead.

    Parameters
    ----------
    train_path : str | Path
        Relative or absolute path to the training CSV (~2 GB, 13M rows).
    test_path : str | Path
        Relative or absolute path to the test CSV (~1 GB, 6M rows).
    csv_params : Optional[Dict[str, Any]]
        Extra kwargs forwarded to pd.read_csv() for both files.
    strategy : ReadStrategyContract
        Reading contract with combine_before_sampling=True
        and add_source_column=True (Option A markers).

    Returns
    -------
    pd.DataFrame
        Combined and sampled DataFrame with strategy.sample_rows rows
        and a source column.

    Raises
    ------
    ValueError
        If strategy flags indicate Option B configuration.
    FileNotFoundError
        If either CSV does not exist.
    """
    # Step 1: Validate strategy flags -- must be Option A configuration.
    if not strategy.combine_before_sampling:
        log.error(
            "[load_csv_combined] strategy.combine_before_sampling=False -- "
            "Option B contract. Use load_train_only() for Option B."
        )
        raise ValueError(
            "load_csv_combined() requires strategy.combine_before_sampling=True "
            "(Option A)."
        )

    if not strategy.add_source_column:
        log.error(
            "[load_csv_combined] strategy.add_source_column=False -- "
            "source column required for Stage 2 drift detection."
        )
        raise ValueError(
            "load_csv_combined() requires strategy.add_source_column=True (Option A)."
        )

    log.info(
        "[load_csv_combined] Option A Stage 2 -- "
        "train=%s test=%s sample_rows=%d random_state=%d",
        train_path,
        test_path,
        strategy.sample_rows,
        strategy.random_state,
    )

    # Step 2: Load train CSV using random sample.
    # log.info("[load_csv_combined] loading train CSV path=%s", train_path)
    # df_train_raw = _load_csv_random_sample(
    #     str(train_path),
    #     csv_params=csv_params,
    #     sample_rows=strategy.sample_rows,
    #     chunksize=strategy.chunksize,
    #     random_state=strategy.random_state,
    # )
    # log.info(
    #     "[load_csv_combined] train loaded rows=%d cols=%d",
    #     len(df_train_raw),
    #     df_train_raw.shape[1],
    # )
    log.info("[load_csv_combined] loading train CSV path=%s", train_path)
    df_train_raw, _, _ = load_by_strategy(
        str(train_path),
        csv_params=csv_params,
        strategy=strategy,
    )
    log.info(
        "[load_csv_combined] train loaded rows=%d cols=%d",
        len(df_train_raw),
        df_train_raw.shape[1],
    )
    # Step 3: Load test CSV using random sample.
    # log.info("[load_csv_combined] loading test CSV path=%s", test_path)
    # df_test_raw = _load_csv_random_sample(
    #     str(test_path),
    #     csv_params=csv_params,
    #     sample_rows=strategy.sample_rows,
    #     chunksize=strategy.chunksize,
    #     random_state=strategy.random_state,
    # )
    # log.info(
    #     "[load_csv_combined] test loaded rows=%d cols=%d",
    #     len(df_test_raw),
    #     df_test_raw.shape[1],
    # )
    log.info("[load_csv_combined] loading test CSV path=%s", test_path)
    df_test_raw, _, _ = load_by_strategy(
        str(test_path),
        csv_params=csv_params,
        strategy=strategy,
    )
    log.info(
        "[load_csv_combined] test loaded rows=%d cols=%d",
        len(df_test_raw),
        df_test_raw.shape[1],
    )
    # Step 4: Add source column to each DataFrame.
    log.info("[load_csv_combined] adding source column to train and test")
    df_train = _add_source_column(df_train_raw, "train")
    df_test = _add_source_column(df_test_raw, "test")

    # Step 5: Combine and sample to exactly strategy.sample_rows rows.
    log.info(
        "[load_csv_combined] combining and sampling to %d rows", strategy.sample_rows
    )
    df_combined = _sample_combined(
        df_train,
        df_test,
        sample_rows=strategy.sample_rows,
        random_state=strategy.random_state,
    )

    # Step 6: Log final output dimensions and source distribution.
    source_counts = df_combined["source"].value_counts().to_dict()
    log.info(
        "[load_csv_combined] done rows=%d cols=%d source_distribution=%s",
        len(df_combined),
        df_combined.shape[1],
        source_counts,
    )
    return df_combined


def filter_by_source(
        df: pd.DataFrame,
        source_value: str,
) -> pd.DataFrame:
    """
    Filter a DataFrame to rows belonging to a specific dataset source.

    Option A Stage 3 only. Called after load_parquet(sample_200k_with_source.parquet)
    to split the combined 200k sample into:
      df_train ~150k rows (train) -- used to fit() transformers.
      df_test  ~50k rows (test)  -- used to transform() only (no fitting).

    This is the mechanism that prevents data leakage in Option A.
    Not called in Option B (no source column in the parquet).

    Parameters
    ----------
    df : pd.DataFrame
        Combined DataFrame with a source column.
        Typically loaded from sample_200k_with_source.parquet.
    source_value : str
        Value to filter on: train or test.

    Returns
    -------
    pd.DataFrame
        Rows where df[source] == source_value with reset index.

    Raises
    ------
    KeyError
        If source column is absent -- parquet was produced by Option B
        or _add_source_column() was not called during Stage 2.
    ValueError
        If source_value is not train or test.
    """
    # Step 1: Validate source column exists -- must be Option A parquet.
    if "source" not in df.columns:
        log.error(
            "[filter_by_source] 'source' column not found -- "
            "ensure parquet was produced by Option A Stage 2 "
            "(combine_before_sampling=True, add_source_column=True). "
            "Option B parquets do not have a source column."
        )
        raise KeyError(
            "'source' column not found. "
            "filter_by_source() only works with Option A parquets."
        )

    # Step 2: Validate source_value.
    _valid = {"train", "test"}
    if source_value not in _valid:
        log.error(
            "[filter_by_source] invalid source_value='%s' -- expected %s",
            source_value,
            _valid,
        )
        raise ValueError(f"source_value='{source_value}' not valid. Expected {_valid}.")

    # Step 3: Log available sources for misconfiguration diagnosis.
    present = df["source"].unique().tolist()
    log.debug(
        "[filter_by_source] filtering source='%s' available=%s total_rows=%d",
        source_value,
        present,
        len(df),
    )

    # Step 4: Apply boolean mask and reset index.
    df_filtered: pd.DataFrame = df.loc[df["source"] == source_value].reset_index(
        drop=True
    )

    # Step 5: Warn if filter returns zero rows.
    if len(df_filtered) == 0:
        log.warning(
            "[filter_by_source] 0 rows returned for source='%s' -- "
            "available: %s. Verify Stage 2 source labels.",
            source_value,
            present,
        )

    # Step 6: Log result with retention percentage.
    log.info(
        "[filter_by_source] source='%s' rows=%d cols=%d "
        "(from total=%d, retained=%.1f%%)",
        source_value,
        len(df_filtered),
        df_filtered.shape[1],
        len(df),
        100 * len(df_filtered) / len(df) if len(df) > 0 else 0.0,
    )
    return df_filtered


# =============================================================================
# SECTION 4.1 - PUBLIC PICKLE READING (re-exported for backward compatibility)
# =============================================================================


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
    # Delegate to the canonical implementation in persist_utils_data.
    from crispdm.data.persist_utils_data import load_pickle as _load_pickle

    return _load_pickle(path)
