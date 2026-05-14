# src/crispdm/common/enum_registry_common.py
"""
=============================================================================
Why this module exists
-----------------------------------------------------------------------------
Defines infrastructure-level enumerations shared across the crispdm package.
"Infrastructure-level" means: no dependency on pipeline configuration, no dependency
on any other crispdm module. These enums map 1-to-1 with Python stdlib
concepts (logging levels) or cross-cutting runtime contracts.

Keeping them in ``common/`` — not in ``configuration/`` — breaks the import cycle
that would arise if ``logging_utils_core`` imported from ``configuration/``:

Public surface:
-----------------------------------------------------------------------------
  LogLevel               → enum used by runtime.log_level (YAML)
  normalize_log_level()  → validates + normalises raw YAML string → LogLevel

Program flow:
-----------------------------------------------------------------------------
  YAML: runtime.log_level: "DEBUG"
      │
      ▼  (build_run_config → schema_dto_config)
  ctx.configuration.runtime.log_level  →  str "DEBUG"
      │
      ▼  (init_run_api → setup_run_logging)
  normalize_log_level("DEBUG")  →  LogLevel.DEBUG
      │
      ▼  (setup_run_logging Step 6)
  getattr(logging, LogLevel.DEBUG.value)  →  10
  root.setLevel(10)

  If YAML has a typo (e.g. "DEBG"):
  normalize_log_level("DEBG")
      → LogLevel("DEBG") raises ValueError
      → pipeline aborts in init_run_api before any artifact is created

Design patterns:
-----------------------------------------------------------------------------
- GoF: none.
- Enterprise / Architectural:
  - Value Object: LogLevel is immutable, identity-less, purely descriptive.
  - Anti-corruption layer: normalize_log_level isolates the rest of the
    codebase from raw YAML strings — callers always receive a typed enum.
=============================================================================
"""
from __future__ import annotations
# =============================================================================
# SECTION 1 – Standard-library imports
# =============================================================================
from enum import Enum

# =============================================================================
# SECTION 2 – Third-party imports
# =============================================================================
# (none required – )

# =============================================================================
# SECTION 3 – Internal imports
# =============================================================================
# (none required – )

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Level logger
# ──────────────────────────────────────────────────────────────────────────────
# (none required – )

# =============================================================================
# SECTION 5 — Constants
# =============================================================================
# (none required – )

# =============================================================================
# SECTION 6 — Type variable
# =============================================================================
# (none required – )

# =============================================================================
# SECTION 7 — Class
# =============================================================================
class LogLevel(str, Enum):
    """Logging verbosity level for a pipeline run.

    Inherits from ``str`` so that instances compare equal to their string
    values, enabling direct YAML comparison and use with
    ``logging.getLevelName()``.  Maps 1-to-1 with Python's five standard
    logging levels.

    The YAML key ``runtime.log_level`` must be one of these values.

    Attributes
    ----------
    DEBUG : str
        Most verbose — emits all operations.  Use for troubleshooting.
    INFO : str
        Major milestones only — stage start/end, key metrics.
    WARNING : str
        Non-blocking issues that may affect results.
    ERROR : str
        Blocking failures that prevent stage completion.
    CRITICAL : str
        Unrecoverable failures that abort the entire pipeline run.
    """

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    def __str__(self) -> str:
        """Return the YAML-compatible string value (e.g. ``"DEBUG"``).

        Returns
        -------
        str
            The enum member's string value.
        """
        return self.value


# =============================================================================
# SECTION 8 — Private functions
# =============================================================================
# (none required – )

# =============================================================================
# SECTION 9 — Public functions
# =============================================================================
def normalize_log_level(value: str | LogLevel,) -> LogLevel:
    """Normalise a raw string or existing enum into a ``LogLevel`` member.

    Accepts the YAML string exactly as written (any case) and returns the
    canonical ``LogLevel`` enum.  Raises ``ValueError`` immediately if the
    value is unrecognised — no silent fallback.

    Parameters
    ----------
    value : str | LogLevel
        Raw YAML string (e.g. ``"debug"``, ``"INFO"``) or an existing
        ``LogLevel`` member.

    Returns
    -------
    LogLevel
        The corresponding enum member.

    Raises
    ------
    ValueError
        If *value* does not match any ``LogLevel`` member after normalisation.
        The error message lists all valid values.
    """
    # Step 1: Pass-through if already a valid enum — avoids redundant work.
    if isinstance(value, LogLevel):
        return value

    # Step 2: Strip whitespace and convert to uppercase — YAML authors may
    #         write lowercase or mixed-case (e.g. "debug", "Debug").
    normalised: str = (value or "").strip().upper()

    # Step 3: Parse into enum — raises ValueError on unrecognised values
    #         with a message that lists all accepted members.
    try:
        return LogLevel(normalised)
    except ValueError:
        valid = [m.value for m in LogLevel]
        raise ValueError(
            f"[normalize_log_level] Unknown LogLevel={value!r}. "
            f"Valid values: {valid}. "
            f"Check runtime.log_level in the pipeline YAML."
        )
