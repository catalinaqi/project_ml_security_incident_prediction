# src/crispdm/common/logging_adapter_common.py
"""
=============================================================================
Why this module exists
-----------------------------------------------------------------------------
Logging is a cross-cutting concern (observability). Every module in the
crispdm package needs a logger, but NONE of them should deal with handlers,
formatters, or file creation. That responsibility lives here.

Public surface (what the rest of the codebase uses):
-----------------------------------------------------------------------------
  get_logger(module_name)                    → called by every module
  config_run_logging(output_root,             → called once by the Facade
                    run_name, log_level)       (init_run_api / preview)

Private helpers (internal to this module only):
-----------------------------------------------------------------------------
  _safe_name(name)                           → filesystem-safe slug
  _build_log_file(output_root, run_name)     → resolves the log Path

Program flow:
-----------------------------------------------------------------------------
  execution_facade_api.py.init_run_api()
    └─► config_run_logging(                   ← single call, single owner
              output_root = ctx.configuration.runtime.output_root,
              run_name    = "run_{task}_{pipeline}_{run_id}",
              log_level   = ctx.configuration.runtime.log_level,   ← from YAML
        )
              ├─► normalize_log_level()      ← validates + typed enum
              ├─► _build_log_file()          ← resolves Path, no I/O
              ├─► creates output/logs/ dir
              ├─► attaches FileHandler + StreamHandler to namespace root
              └─► returns Path(log_file)     ← stored in ctx.log_file

  Every other module:
    log = get_logger(__name__)               ← no handler logic, ever
    log.info("...")

Key design decisions:
-----------------------------------------------------------------------------
- log_level always comes from YAML (runtime.log_level). There is no
  hardcoded default — if the value is missing or invalid the call fails
  loudly (ValueError) instead of silently falling back to DEBUG.
- config_run_logging is idempotent: calling it a second time for the SAME
  log file is a no-op. Calling it for a NEW file replaces all handlers
  so each run gets a clean log. This relies on _CONFIGURED_FOR (module-
  level singleton). Known limitation: not safe for parallel runs in the
  same process (e.g. pytest -n auto). Use separate processes for that.
- Format strings are module constants — one place to change layout.
- log_level validation is delegated to normalize_log_level() in
  common/enums_utils_core.py — single source of truth for LogLevel.

Design patterns:
-----------------------------------------------------------------------------
- GoF: none (Python logging is a library facility).
- Enterprise / Architectural:
  - Cross-cutting concern / Observability.
  - Singleton-ish initialisation: configure handlers once per run.
  - Facade orchestrates logging init; modules only consume get_logger().
=============================================================================
"""


from __future__ import annotations
# =============================================================================
# SECTION 1 – Standard-library imports
# =============================================================================
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

# =============================================================================
# SECTION 2 – Third-party imports
# =============================================================================
# (none required – all functionality uses stdlib logging + optional extensions)

# =============================================================================
# SECTION 3 – Internal imports
# =============================================================================
from crispdm.common.enum_registry_common import LogLevel, normalize_log_level

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Level logger
# ──────────────────────────────────────────────────────────────────────────────
# (none required – )

# =============================================================================
# SECTION 5 — Constants
# =============================================================================
# All literals are defined here. No string/format literals appear in function
# bodies — change layout or directory name in exactly one place.
# ---------------------------------------------------------------------------

# Namespace prefix shared by every logger in the project.
_LOGGER_NAMESPACE: str = "LOGGER_NAMESPACE_ML"

# Sub-directory name inside output_root where log files are written.
_LOGS_DIR_NAME: str = "logs"

# strftime pattern used to stamp log file names.
_TIMESTAMP_FMT: str = "%Y%m%d_%H%M%S"

# Console handler format — compact, readable in notebook output.
_CONSOLE_FMT: str = "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s"
_CONSOLE_DATEFMT: str = "%H:%M:%S"

# File handler format — tab-separated, includes file:line for grep/debug.
_FILE_FMT: str = (
    "%(asctime)s\t%(levelname)s\t%(name)s\t%(filename)s:%(lineno)d\t%(message)s"
)
_FILE_DATEFMT: str = "%Y-%m-%d %H:%M:%S"

# Tracks the log file active for the current process (idempotency guard).
_CONFIGURED_FOR: Optional[Path] = None

# =============================================================================
# SECTION 6 — Type variable
# =============================================================================
# (none required – )

# =============================================================================
# SECTION 7 — Class
# =============================================================================
# (none required – )

# =============================================================================
# SECTION 8 — Private functions
# =============================================================================
def _safe_name(name: str) -> str:
    """Convert a run or pipeline name to a filesystem-safe slug.

    Keeps ASCII letters, digits, underscores, and hyphens.  All other
    characters (spaces, dots, slashes, etc.) are collapsed to a single
    hyphen.  The result is always non-empty.

    Parameters
    ----------
    name : str
        Raw run or pipeline name (e.g. ``"run_clustering_option_a_20260422"``).

    Returns
    -------
    str
        Lowercase slug safe for use as a filename component.
        Falls back to ``"run"`` if the input is blank after stripping.
    """
    # Step 1: Normalise whitespace and convert to lowercase.
    name = (name or "").strip().lower()

    # Step 2: Replace any run of disallowed characters with a single hyphen.
    name = re.sub(r"[^a-z0-9_\-]+", "-", name)

    # Step 3: Collapse consecutive hyphens and strip leading/trailing ones.
    name = re.sub(r"-{2,}", "-", name).strip("-")

    # Step 4: Guarantee a non-empty result.
    return name or "run"


def _build_log_file(
    output_root: Path | str,
    run_name: str,
    timestamp: Optional[str] = None,
) -> Path:
    """Resolve the log file path for a single run without creating it.

    The file and its parent directory are created later inside
    ``config_run_logging``.  Keeping path resolution separate makes the
    function independently testable with a fixed timestamp.

    Parameters
    ----------
    output_root : Path | str
        Root output directory (e.g. ``"out"``).  The ``logs/`` sub-directory
        is appended automatically using ``_LOGS_DIR_NAME``.
    run_name : str
        Human-readable run identifier used as the filename stem
        (e.g. ``"run_clustering_option_a_20260422_185654"``).
    timestamp : str or None, optional
        Pre-computed ``_TIMESTAMP_FMT`` string.  When ``None`` the current
        wall-clock time is used.  Inject a fixed value in tests to get a
        deterministic path.

    Returns
    -------
    Path
        Full path of the form
        ``<output_root>/logs/<safe_run_name>_<timestamp>.log``.

    Examples
    --------
    >>> _build_log_file("out", "my_run", timestamp="20260422_185654")
    PosixPath('out/logs/my_run_20260422_185654.log')
    """
    # Step 1: Resolve output root and append the logs sub-directory constant.
    logs_dir: Path = Path(output_root) / _LOGS_DIR_NAME

    # Step 2: Stamp the filename — use provided timestamp or current time.
    ts: str = timestamp or datetime.now().strftime(_TIMESTAMP_FMT)

    # Step 3: Assemble the final path with the sanitised run name.
    return logs_dir / f"{_safe_name(run_name)}_{ts}.log"


# =============================================================================
# SECTION 9 — Public functions
# =============================================================================
def get_logger(module_name: str) -> logging.Logger:
    """Return a namespaced logger for a module.

    Every module in the package should obtain its logger through this
    function — never through ``logging.getLogger`` directly — so that all
    records share the ``_LOGGER_NAMESPACE`` prefix and are captured by the
    handler configured in ``config_run_logging``.

    ``config_run_logging`` should be called by the Facade before any module
    emits records; until then records propagate to the Python root logger.

    Parameters
    ----------
    module_name : str
        Fully-qualified module name, typically ``__name__``.

    Returns
    -------
    logging.Logger
        Logger bound to ``<_LOGGER_NAMESPACE>.<module_name>``.
    """
    # Step 1: Return namespaced child logger — no handler side-effects.
    return logging.getLogger(f"{_LOGGER_NAMESPACE}.{module_name}")


def config_run_logging(
    output_root: Path | str,
    log_level: str,
    run_name: str,
) -> Path:
    """Configure file and console logging scoped to a single pipeline run.

    This is the **only** logging setup function the rest of the codebase
    should call.  It validates the level via ``normalize_log_level``,
    resolves the log file path, creates the ``logs/`` directory, and
    attaches handlers to the namespace root logger.

    Called **once** per run from ``init_run_api`` in the Facade layer.
    The returned ``Path`` is stored in ``ctx.log_file`` for audit purposes.

    Parameters
    ----------
    output_root : Path | str
        Root output directory from ``ProjectConfig.runtime.output_root``
        (YAML key ``runtime.output_root``).
    run_name : str
        Human-readable run identifier used as the log file stem
        (e.g. ``"run_clustering_option_a_20260422_185654"``).
    log_level : str
        Logging verbosity from ``ProjectConfig.runtime.log_level``
        (YAML key ``runtime.log_level``).
        Accepted: ``DEBUG | INFO | WARNING | ERROR | CRITICAL``.

    Returns
    -------
    Path
        Absolute path to the log file that was opened.

    Raises
    ------
    ValueError
        If ``log_level`` is not a recognised ``LogLevel`` member.
        Raised by ``normalize_log_level`` — no silent fallback.

    Notes
    -----
    **Idempotency**: calling this function a second time with the *same*
    ``log_file`` path is a no-op.  Calling it with a *different* path (new
    run in the same process) replaces all existing handlers so each run
    writes to its own clean file.

    **Parallel-run limitation**: ``_CONFIGURED_FOR`` is a module-level
    singleton.  Two concurrent runs in the same process (e.g. ``pytest -n
    auto``) will overwrite each other's handler state.  Use separate
    OS processes for parallel pipeline execution.
    """
    global _CONFIGURED_FOR

    # Step 1: Validate and normalise log_level from YAML via LogLevel enum.
    #         normalize_log_level raises ValueError if the value is not a
    #         recognised member — no silent fallback to DEBUG.
    level: LogLevel = normalize_log_level(log_level)

    # Step 2: Resolve the log file path via private helper (no I/O yet).
    log_file: Path = _build_log_file(output_root, run_name=run_name)

    # Step 3: Idempotency guard — same file already active, nothing to do.
    root: logging.Logger = logging.getLogger(_LOGGER_NAMESPACE)
    if _CONFIGURED_FOR == log_file and root.handlers:
        return log_file

    # Step 4: New run in the same process — remove stale handlers cleanly.
    if root.handlers:
        for handler in list(root.handlers):
            root.removeHandler(handler)

    # Step 5: Create the logs directory (parents=True handles first-run).
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Step 6: Set namespace root level and stop propagation to root logger.
    #         level.value → "DEBUG" → getattr(logging, "DEBUG") → 10
    numeric_level: int = getattr(logging, level.value)
    root.setLevel(numeric_level)
    root.propagate = False

    # Step 7: Build formatters from module constants — no inline literals.
    console_formatter = logging.Formatter(fmt=_CONSOLE_FMT, datefmt=_CONSOLE_DATEFMT)
    file_formatter = logging.Formatter(fmt=_FILE_FMT, datefmt=_FILE_DATEFMT)

    # Step 8: Attach console handler (StreamHandler) for notebook/terminal.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(console_formatter)

    # Step 9: Attach file handler — append mode, UTF-8, path from Step 2.
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(file_formatter)

    root.addHandler(console_handler)
    root.addHandler(file_handler)

    # Step 10: Record active file for idempotency on future calls.
    _CONFIGURED_FOR = log_file
    root.debug(
        "[config_run_logging] handlers attached log_file=%s level=%s",
        log_file,
        level.value,
    )

    return log_file
