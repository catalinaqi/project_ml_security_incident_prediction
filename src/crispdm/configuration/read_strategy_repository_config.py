
from __future__ import annotations


from dataclasses import dataclass
from typing import Any, Dict, Generator, Optional, Tuple

from crispdm.configuration.enum_registry_config import ReadMode, normalize_read_mode
from crispdm.common.logging_adapter_common import get_logger

log = get_logger(__name__)

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
    stratify_column: str
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

        stratify_column: str = str(raw["stratify_column"])

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
            "mode=%s sample_rows=%d sample_method=%s stratify_column=%s chunksize=%d "
            "random_state=%d combine=%s add_source=%s "
            "input_source=%s input_source_sample=%s input_source_full=%s",
            mode.value,
            sample_rows,
            sample_method,
            stratify_column,
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
            stratify_column=stratify_column,
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

