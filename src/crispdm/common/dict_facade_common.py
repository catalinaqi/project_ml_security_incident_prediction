# src/crispdm/common/helpers_adapter_core.py
"""
=============================================================================
ADD TO:
Append these two functions at the end of the existing file.

Why here and not in configuration/?
----------------------------
enabled() and dget() are pure Python dict utilities — they import nothing,
know nothing about YAML schemas, and have no domain dependency.
The common/ layer is the correct home for cross-cutting stateless helpers.
Every layer (configuration, data, stage, pipeline) imports from common without
creating circular dependencies.
=============================================================================
"""
from __future__ import annotations
# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1 – Standard-library imports
# ──────────────────────────────────────────────────────────────────────────────
from typing import Any, Mapping, TypeVar, cast, overload

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2 – Third-party imports
# ──────────────────────────────────────────────────────────────────────────────
# (none required – )

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3 – Internal imports
# ──────────────────────────────────────────────────────────────────────────────
# (none required – )

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Level logger
# ──────────────────────────────────────────────────────────────────────────────
# (none required – )

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Constants
# ──────────────────────────────────────────────────────────────────────────────
# (none required – )

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Type variable
# ──────────────────────────────────────────────────────────────────────────────
_T = TypeVar("_T")

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Class
# ──────────────────────────────────────────────────────────────────────────────
# (none required – )

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 8 — Private functions
# ──────────────────────────────────────────────────────────────────────────────
# (none required – )

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 9 — Public functions
# ──────────────────────────────────────────────────────────────────────────────
def enabled(node: object, *, default: bool = True) -> bool:
    """
    Return the ``enabled`` flag from a YAML configuration node.

    Every step / technique block in the pipeline YAML may carry an
    ``enabled`` key.  This helper centralises the check so callers avoid
    repeated ``isinstance`` guards across all stage runners.

    Parameters
    ----------
    node : object
        Dict-like YAML node.  Non-dict values fall back to *default*.
    default : bool, optional
        Fallback when the node is not a dict or the ``"enabled"`` key is
        absent.  Keyword-only to prevent positional misuse.

    Returns
    -------
    bool
        ``True`` if the node is enabled; ``False`` otherwise.

    Examples
    --------
    >>> enabled({"enabled": True})
    True
    >>> enabled({"enabled": False}, default=True)
    False
    >>> enabled(None)
    True
    >>> enabled(None, default=False)
    False
    >>> enabled({})  # key absent → enabled
    True
    """
    # Step 1: Non-dict nodes cannot carry an enabled flag — return default.
    if not isinstance(node, dict):
        return default

    # Step 2: Resolve the flag; fall back to default when key is absent.
    return bool(node.get("enabled", default))


# ──────────────────────────────────────────────────────────────────────────────
# dget — type-safe dict accessor  (3 type signatures required by mypy-strict)
# ──────────────────────────────────────────────────────────────────────────────


@overload
def dget(d: Mapping[str, Any], key: str, default: _T) -> _T:
    "Overload 1 — using default: returns default's exact type."
    ...


@overload
def dget(d: Mapping[str, Any], key: str) -> Any:
    "Overload 2 — without default: returns Any (may be None)."
    ...


def dget(
        d: Mapping[str, Any],
        key: str,
        default: _T | None = None,
) -> _T | Any:
    """Get a value from a dict, treating an explicit ``None`` as absent.

    Args:
        d:       Source mapping (typically a YAML-loaded ``dict[str, Any]``).
        key:     Key to look up.
        default: Fallback value returned when key is missing or mapped to
                 ``None``.  When provided, the return type is narrowed to
                 ``type(default)`` by the overload signatures above.

    Returns
    -------
        The mapped value cast to ``_T``, or ``default`` if absent/None.

    Example:
        >>> cfg = {"compression": "snappy"}
        >>> dget(cfg, "compression", default="")
        'snappy'
        >>> dget(cfg, "missing_key", default="fallback")
        'fallback'
    """
    # Step 1: Use .get() without a sentinel to isolate the None case.
    v = d.get(key)

    # Step 2: If None (key absent or YAML null), return the caller's default.
    if v is None:
        return default

    # Step 3: Cast convinces mypy the value matches the _T of `default`.
    return cast(_T, v)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Private functions
# ──────────────────────────────────────────────────────────────────────────────
# (none required – )