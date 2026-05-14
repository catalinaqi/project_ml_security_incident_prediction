# src/crispdm/common/path_service_common.py
"""
=============================================================================
Why this module exists
-----------------------------------------------------------------------------
Centralized path resolution utilities for the CRISP-DM pipeline.
Provides a single source of truth for locating the project root and
resolving relative paths used across all pipeline phases and notebooks.

Without this module, each layer (data, stage, pipeline, notebook) would
independently implement project root detection — leading to duplication,
inconsistency, and hard-to-debug path errors across execution contexts
(notebook, CLI, CI/CD).

Covers: Option A and Option B pipelines (path resolution is strategy-agnostic).

Program flow:
-----------------------------------------------------------------------------
- Notebook          → imports find_project_root() to set PROJECT_ROOT
- load_utils_data   → imports resolve_path() to resolve CSV/Parquet paths
- phase/*           → imports resolve_path() to resolve artifact output paths
- pipeline/*        → imports resolve_path() to resolve YAML configuration paths

Design patterns
-----------------------------------------------------------------------------
- GoF: none
- Enterprise/Architectural:
  - Utility Layer:
      - stateless functions with no side effects
  - Single Source of Truth:
      - one place for all path resolution logic
=============================================================================
"""
from __future__ import annotations
# =============================================================================
# SECTION 1 – Standard-library imports
# =============================================================================
from pathlib import Path
from typing import Optional

# =============================================================================
# SECTION 2 – Third-party imports
# =============================================================================
# (none required – )

# =============================================================================
# SECTION 3 – Internal imports
# =============================================================================
from crispdm.common.logging_adapter_common import get_logger

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Level logger
# ──────────────────────────────────────────────────────────────────────────────
log = get_logger(__name__)

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
# (none required – )

# =============================================================================
# SECTION 8 — Private functions
# =============================================================================
# (none required – )

# =============================================================================
# SECTION 9 — Public functions
# =============================================================================
def find_project_root(start: Optional[Path] = None) -> Path:
    """
    Locate the project root by traversing the directory tree upward.

    Searches for standard project root markers (``pyproject.toml`` or ``.git``)
    starting from *start* and walking toward the filesystem root. Raises
    ``RuntimeError`` if no marker is found, making misconfiguration explicit
    rather than silently falling back to an incorrect directory.

    Covers: Option A and Option B (path resolution is pipeline-agnostic).

    Parameters
    ----------
    start : Optional[Path]
        Directory from which to begin the upward search.
        If ``None``, the current working directory (``Path.cwd()``) is used.

    Returns
    -------
    Path
        Absolute path of the first directory containing ``pyproject.toml``
        or ``.git``.

    Raises
    ------
    RuntimeError
        If neither marker is found anywhere in the directory hierarchy.

    Examples
    --------
    >>> root = find_project_root()
    >>> configuration = root / "configuration" / "pipelines" / "clustering_pipeline_config.yml"
    """
    # Step 1: Resolve starting directory to an absolute path.
    start_dir: Path = (start or Path.cwd()).resolve()
    log.debug("[find_project_root] search starting at dir=%s", start_dir)

    # Step 2: Build upward traversal chain: [start_dir, parent, grandparent, ..., /].
    search_chain = [start_dir, *start_dir.parents]

    # Step 3: Walk upward and return the first directory that contains a marker.
    for candidate in search_chain:
        has_pyproject = (candidate / "pyproject.toml").exists()
        has_git = (candidate / ".git").exists()

        if has_pyproject or has_git:
            log.debug(
                "[find_project_root] project root found at dir=%s "
                "(pyproject=%s git=%s)",
                candidate,
                has_pyproject,
                has_git,
            )
            return candidate

    # Step 4: No marker found — raise explicitly so misconfiguration is visible.
    log.error(
        "[find_project_root] no project root marker found "
        "starting from dir=%s — ensure pyproject.toml or .git exists",
        start_dir,
    )
    raise RuntimeError(
        f"Project root not found (pyproject.toml / .git) "
        f"starting from '{start_dir}'. "
        f"Ensure the project has a pyproject.toml or .git at its root."
    )


def resolve_path(
        path: str | Path,
        root: Optional[Path] = None,
) -> Path:
    """
    Resolve a file path to an absolute path anchored at the project root.

    If *path* is already absolute, it is returned as-is without calling
    ``find_project_root()``. If *path* is relative, it is joined to *root*
    (or the detected project root when *root* is ``None``).

    Used internally by ``load_utils_data`` to resolve CSV and Parquet paths
    declared in YAML ``read_strategy`` fields (``train_path``, ``test_path``,
    ``input_source``, ``input_source_full``, ``input_source_sample``).

    Covers: Option A and Option B (called for every path regardless of strategy).

    Parameters
    ----------
    path : str | Path
        Relative or absolute file path to resolve.
    root : Optional[Path]
        Explicit project root to use as the anchor for relative paths.
        If ``None``, ``find_project_root()`` is called automatically.

    Returns
    -------
    Path
        Absolute, resolved ``Path`` object.

    Raises
    ------
    RuntimeError
        Propagated from ``find_project_root()`` if the project root cannot
        be located and *root* is ``None``.

    Examples
    --------
    >>> resolved = resolve_path("data/raw/train/GUIDE_Train.csv")
    >>> resolved.is_absolute()
    True
    """
    # Step 1: Normalise input to a Path object.
    p = Path(path)
    log.debug("[resolve_path] resolving path=%s (absolute=%s)", p, p.is_absolute())

    # Step 2: Return absolute paths unchanged — no root resolution needed.
    if p.is_absolute():
        log.debug("[resolve_path] path is absolute, returning as-is path=%s", p)
        return p

    # Step 3: Determine the anchor root (explicit or auto-detected).
    anchor: Path = root if root is not None else find_project_root()
    log.debug("[resolve_path] anchor root=%s", anchor)

    # Step 4: Join relative path to the anchor root and resolve symlinks.
    resolved = (anchor / p).resolve()
    log.debug("[resolve_path] resolved path=%s", resolved)

    # Step 5: Warn if the resolved path does not exist — helps catch YAML typos early.
    if not resolved.exists():
        log.warning(
            "[resolve_path] resolved path does not exist — "
            "verify YAML configuration path=%s (resolved=%s)",
            path,
            resolved,
        )

    return resolved


