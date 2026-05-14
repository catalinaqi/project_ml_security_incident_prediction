# src/crispdm/configuration/enum_regsitry_config.py
from __future__ import annotations
"""
=============================================================================
Why this module exists
-----------------------------------------------------------------------------
Central registry of "configuration enums" and their normalization helpers.
YAML files contain raw strings — this module converts them into controlled,
typed enum values so the rest of the pipeline never operates on bare strings.

Covered pipeline types: clustering · classification · regression · timeseries
Covered options:        Option A (drift-aware) · Option B (train-only)

Enum inventory:
-----------------------------------------------------------------------------
  ProblemType          — ML task family (clustering/classification/…)
  CsvSourceType        — data source format (csv only for this project)
  ReadMode             — CSV loading strategy (full/sample/chunked)
  LogLevel             — logging verbosity (DEBUG → … → CRITICAL)
  FeatureSelectionMode — feature selection strategy (auto/include/exclude)

Program flow:
-----------------------------------------------------------------------------
- YAML configuration (string values)
    -> load_loader_config.load_and_resolve()  -> raw dict
    -> validate_validator_config.validate_config_dict() (uses normalize_*)
    -> schema_dto_config.ProjectConfig.from_dict()      (uses normalize_*)
    -> typed DTOs with enum fields

Design principles:
-----------------------------------------------------------------------------
- All enums inherit (str, Enum) so instances compare equal to their string
  equivalents (e.g. ProblemType.CLUSTERING == "clustering" → True).
  This lets callers pass either a ProblemType or the plain string "clustering"
  without adapter code, and allows direct YAML-value comparison.

- __str__ returns self.value (the YAML-compatible string) so enum instances
  format correctly in log messages, f-strings, and Path concatenations.
  No log.debug inside __str__ — that causes log noise and recursive logging
  on every format call (every log.info("task=%s", task) would trigger a
  log.debug inside task.__str__()).

- normalize_*() functions are the single entry point for string → enum
  conversion. They are called by both the validator and the DTO factory so
  the conversion logic is never duplicated.

Design patterns
-----------------------------------------------------------------------------
- GoF -> Gang of Four: none.
- Enterprise/Architectural:
  - Typed Configuration Boundary: invalid strings are rejected at the
      boundary (normalize_*) so internal code only ever sees valid enums.
  - Defensive Parsing (fail-fast): normalize_* raises ValueError immediately
      on unrecognised values — misconfiguration surfaces at startup.
=============================================================================
"""


# =============================================================================
# SECTION 1 – Standard-library imports
# =============================================================================
from enum import Enum

# =============================================================================
# SECTION 2 – Third-party imports
# =============================================================================
# (none required – all functionality uses stdlib logging + optional extensions)

# =============================================================================
# SECTION 3 – Internal imports
# =============================================================================
from crispdm.common.logging_adapter_common import get_logger

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Level logger
# ──────────────────────────────────────────────────────────────────────────────
log = get_logger(__name__)

# =============================================================================
# SECTION 1 — ENUMS
# =============================================================================

# =============================================================================
# SECTION 5 — Constants
# ============================================================================

# =============================================================================
# SECTION 6 — Type variable
# =============================================================================
# (none required – )

# =============================================================================
# SECTION 7 — Class
# =============================================================================
class ProblemType(str, Enum):
    """
    Top-level ML problem family — drives pipeline branch selection.

    Inherits from ``str`` so ``ProblemType.CLUSTERING == "clustering"`` is
    ``True``, enabling direct comparison with YAML string values.

    Attributes
    ----------
    CLUSTERING : str
        Unsupervised grouping — no target column required.
    CLASSIFICATION : str
        Supervised discrete-label prediction — requires ``target_col``.
    REGRESSION : str
        Supervised continuous-value prediction — requires ``target_col``.
    TIMESERIES : str
        Temporal sequence modelling — requires ``time_col``.
    """

    CLUSTERING = "clustering"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    TIMESERIES = "timeseries"

    def __str__(self) -> str:
        """Return the YAML-compatible string value."""
        return self.value


class ReadMode(str, Enum):
    """
    CSV loading strategy for large files (train ~2 GB, test ~1 GB).

    Inherits from ``str`` for direct YAML string comparison.
    The YAML ``read_strategy.mode`` key maps to one of these values.

    Attributes
    ----------
    FULL : str
        Load the entire CSV into memory.  Only safe for small files or
        environments with sufficient RAM.
    SAMPLE : str
        Load a random / head / tail subset of rows.  Used in Stages 2–4
        to keep memory bounded while preserving statistical properties.
    CHUNKED : str
        Process the CSV in fixed-size iteration chunks.  Used in Stage 5
        for full test-set evaluation without loading 1 GB at once.
    """

    FULL = "full"
    SAMPLE = "sample"
    CHUNKED = "chunked"

    def __str__(self) -> str:
        """Return the YAML-compatible string value."""
        return self.value


class PhaseDir(str, Enum):
    """
    CRISP-DM stage directory names under ``out/runs/{task}/{dataset}/{run_id}/``.

    Inherits from ``str`` so ``PhaseDir.PHASE2 == "phase2_data_understanding"`` is
    ``True``, enabling direct use in ``Path`` expressions without calling
    ``.value`` explicitly.

    Single source of truth for stage directory names — the string literal
    ``"phase2_data_understanding"`` never appears outside this definition.

    Attributes
    ----------
    MODELS : str
        Serialised model artefacts (joblib, pickle) — not a CRISP-DM stage.
    STAGE2 : str
        Stage 2 — Data Understanding outputs.
    STAGE3 : str
        Stage 3 — Data Preparation outputs.
    STAGE4 : str
        Stage 4 — Modelling outputs.
    STAGE5 : str
        Stage 5 — Evaluation outputs.
    """

    PHASE2 = "phase2_data_understanding"
    PHASE3 = "phase3_data_preparation"
    PHASE4 = "phase4_data_modeling"
    PHASE5 = "phase5_evaluation_and_interpretation"

    def __str__(self) -> str:
        return self.value

class StageSubDir(str, Enum):
    """
    Artifact sub-directory names inside each CRISP-DM stage directory.

    Inherits from ``str`` so ``StageSubDir.FIGURES == "figures"`` is
    ``True``, enabling direct use in ``Path`` expressions without calling
    ``.value`` explicitly.

    Created by ``make_run_dir()`` at run init — controlled by the
    ``output_policy`` block of each stage YAML config:

    * ``output_policy.save_all_as_png: true``         → creates ``figures/``
    * ``output_policy.save_all_tables_as_png: true``  → creates ``tables_png/``

    Attributes
    ----------
    FIGURES : str
        Sub-directory for all plot/chart PNG artefacts.
    TABLES_PNG : str
        Sub-directory for DataFrames rendered as PNG images.
    """

    FIGURES = "figures"
    TABLES_PNG = "tables_png"
    REPORTS = "reports"
    METRICS = "metrics"


class StepsPhase(str, Enum):  # noqa: D101
    STEP_2_1 = "step_2_1_data_acquisition"
    STEP_2_2 = "step_2_2_data_description"
    STEP_2_3 = "step_2_3_data_quality_assessment"
    STEP_2_4 = "step_2_4_data_exploration"
    STEP_3_1 = "step_3_1_data_selection"
    STEP_3_2 = "step_3_2_data_cleaning"
    STEP_3_3 = "step_3_3_data_transformation"
    STEP_3_4 = "step_3_4_data_integration"
    STEP_3_5 = "step_3_5_data_formatting"
    STEP_4_1 = "step_4_1_algorithm_selection"
    STEP_4_2 = "step_4_2_model_training"
    STEP_4_3 = "step_4_3_test_design"
    STEP_4_4 = "step_4_4_model_evaluation"
    STEP_5_1 = "step_5_1_interpretation"
    STEP_5_2 = "step_5_2_business_evaluation"
    STEP_5_3 = "step_5_3_process_audit"
    STEP_5_4 = "step_5_4_decision_making"

    def __str__(self) -> str:
        return self.value

# =============================================================================
# SECTION 8 — Private functions
# =============================================================================
# (none required – )

# =============================================================================
# SECTION 9 — Public functions
# ============================================================================
def normalize_problem_type(
        value: str | ProblemType,
) -> ProblemType:
    """
    Normalize a raw string or existing enum into a ``ProblemType`` member.

    Called by the validator and the DTO factory so conversion logic is
    never duplicated.  Accepts the already-enum case cheaply (no string
    operations).

    Parameters
    ----------
    value : str | ProblemType
        Raw YAML string (e.g. ``"clustering"``) or an existing
        ``ProblemType`` member.

    Returns
    -------
    ProblemType
        The corresponding enum member.

    Raises
    ------
    ValueError
        If *value* is not a recognised ``ProblemType`` string.
    """
    # Step 1: Pass-through if already a valid enum — avoids redundant work.
    if isinstance(value, ProblemType):
        log.debug("[normalize_problem_type] already enum value=%s", value.value)
        return value

    # Step 2: Normalise to lowercase stripped string to tolerate YAML casing.
    normalised: str = (value or "").strip().lower()

    # Step 3: Parse into enum — raises ValueError on unrecognised values.
    try:
        result = ProblemType(normalised)
    except ValueError:
        valid = [m.value for m in ProblemType]
        log.error("[normalize_problem_type] invalid value=%r valid=%s", value, valid)
        raise ValueError(f"Unknown ProblemType={value!r}. Valid values: {valid}")

    # Step 4: Log resolved value and return.
    log.debug("[normalize_problem_type] resolved value=%r -> %s", value, result.value)
    return result


def normalize_read_mode(value: str | ReadMode) -> ReadMode:
    """
    Normalize a raw string or existing enum into a ``ReadMode`` member.

    Parameters
    ----------
    value : str | ReadMode
        Raw YAML string (e.g. ``"sample"``) or an existing ``ReadMode``
        member.

    Returns
    -------
    ReadMode
        The corresponding enum member.

    Raises
    ------
    ValueError
        If *value* is not a recognised ``ReadMode`` string.
    """
    # Step 1: Pass-through if already a valid enum — avoids redundant work.
    if isinstance(value, ReadMode):
        log.debug("[normalize_read_mode] already enum value=%s", value.value)
        return value

    # Step 2: Normalise to lowercase stripped string to tolerate YAML casing.
    normalised: str = (value or "").strip().lower()

    # Step 3: Parse into enum — raises ValueError on unrecognised values.
    try:
        result = ReadMode(normalised)
    except ValueError:
        valid = [m.value for m in ReadMode]
        log.error("[normalize_read_mode] invalid value=%r valid=%s", value, valid)
        raise ValueError(f"Unknown ReadMode={value!r}. Valid values: {valid}")

    # Step 4: Log resolved value and return.
    log.debug("[normalize_read_mode] resolved value=%r -> %s", value, result.value)
    return result

