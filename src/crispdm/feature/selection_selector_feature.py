# src/crispdm/feature/selection_selector_feature.py
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from crispdm.common.dict_facade_common import dget
from crispdm.common.logging_adapter_common import get_logger

log = get_logger(__name__)


def bigint_cleanup(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    """Apply sentinel removal on columns with bigint thresholds.

    Config structure expected:
        {
            "enabled": bool,
            "params": {
                "thresholds": {col_name: threshold, ...},
                "actions": {
                    "create_binary_flags": bool,
                    "nullify_above_threshold": bool,
                    "keep_real_values": bool,
                    "include_real_in_output": bool
                }
            }
        }

    Behavior:
        - nullify_above_threshold: replaces values > threshold with NaN.
        - create_binary_flags: creates <col>_is_missing binary column.
        - keep_real_values: preserves any existing <col>_real columns.
        - include_real_in_output: creates <col>_real columns containing the
          original values BEFORE nullification.
    """
    log.info("[3.1] bigint_cleanup called with %d columns", df.shape[1])
    log.debug("[3.1] bigint_cleanup config received: %s", config)

    enabled = config.get("enabled", True)
    if not enabled:
        log.info("[3.1] bigint_cleanup disabled – skipping")
        return df, {"applied": False}

    params = config.get("params", {})
    thresholds = params.get("thresholds", {})

    if not thresholds:
        log.warning("[3.1] bigint_cleanup: no thresholds defined in config=%s – returning empty report", config)
        return df, {"applied": True, "error": "no thresholds defined"}

    actions = params.get("actions", {})
    create_binary_flags = actions.get("create_binary_flags", True)
    nullify_above_threshold = actions.get("nullify_above_threshold", True)
    keep_real_values = actions.get("keep_real_values", True)
    include_real_in_output = actions.get("include_real_in_output", False)

    log.debug("[3.1] bigint_cleanup thresholds: %s | create_binary_flags=%s | nullify=%s | keep_real=%s | include_real=%s",
              thresholds, create_binary_flags, nullify_above_threshold, keep_real_values, include_real_in_output)

    result_df = df.copy()
    nullified = []
    flags = []
    real_columns_created = []

    for col, threshold in thresholds.items():
        if col not in result_df.columns:
            log.warning("[3.1] bigint_cleanup: column '%s' not found in dataframe – skipping", col)
            continue

        if nullify_above_threshold and not col.endswith("_real"):
            mask = result_df[col] > threshold
            if mask.any():
                # --- Optionally create _real column with original values ---
                if include_real_in_output:
                    real_col = f"{col}_real"
                    if real_col not in result_df.columns:
                        result_df[real_col] = result_df[col].copy()
                        real_columns_created.append(real_col)
                        log.info("[3.1] bigint_cleanup: created real column '%s' with original values", real_col)
                    else:
                        log.debug("[3.1] bigint_cleanup: real column '%s' already exists – preserving", real_col)

                # --- Nullify values above threshold ---
                result_df.loc[mask, col] = np.nan
                nullified.append(col)
                log.debug("[3.1] bigint_cleanup: nullified %d values in column '%s'", mask.sum(), col)

                # --- Create binary missing flag ---
                if create_binary_flags:
                    flag_col = f"{col}_is_missing"
                    result_df[flag_col] = mask.astype(int)
                    flags.append(flag_col)
                    log.debug("[3.1] bigint_cleanup: created flag '%s'", flag_col)
            else:
                log.debug("[3.1] bigint_cleanup: column '%s' has no values above threshold %s", col, threshold)

        # --- Preserve existing _real columns (from Phase 2) ---
        if keep_real_values:
            real_col = f"{col}_real"
            if real_col in result_df.columns:
                log.debug("[3.1] bigint_cleanup: preserving real column '%s'", real_col)

    log.info("[3.1] bigint_cleanup finished: nullified=%d, flags=%d, real_created=%d",
             len(nullified), len(flags), len(real_columns_created))
    return result_df, {
        "columns_nullified": nullified,
        "flags_added": flags,
        "real_columns_created": real_columns_created,
    }


def manual_include_exclude(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    """Apply manual include/exclude column filtering.

    Config structure expected:
        {
            "params": {
                "include": [col_names],
                "exclude": [col_names]
            }
        }
    Note: If 'include' is empty, all columns are kept before exclusions.
    """
    params = config.get("params", {})
    include = params.get("include", [])
    exclude = params.get("exclude", [])

    log.info("[3.1] manual_include_exclude called with include=%d, exclude=%d", len(include), len(exclude))
    log.debug("[3.1] manual_include_exclude config: include=%s, exclude=%s", include, exclude)

    original = list(df.columns)
    excluded = []

    if include:
        # Warn about requested columns that don't exist
        missing_include = [c for c in include if c not in original]
        if missing_include:
            log.warning("[3.1] manual_include_exclude: include columns not found in dataframe: %s", missing_include)
        keep = [c for c in include if c in original]
        excluded = [c for c in original if c not in keep]
        result_df = df[keep].copy()
        log.debug("[3.1] manual_include_exclude: selected %d columns (dropped %d)", len(keep), len(excluded))
    else:
        result_df = df.copy()

    if exclude:
        # Warn about requested columns to drop that don't exist
        missing_exclude = [c for c in exclude if c not in result_df.columns]
        if missing_exclude:
            log.warning("[3.1] manual_include_exclude: exclude columns not found in dataframe: %s", missing_exclude)
        to_drop = [c for c in exclude if c in result_df.columns]
        if to_drop:
            result_df = result_df.drop(columns=to_drop)
            excluded.extend(to_drop)
            log.debug("[3.1] manual_include_exclude: dropped explicit exclude columns: %s", to_drop)

    log.info("[3.1] manual_include_exclude finished: removed %d columns total", len(excluded))
    return result_df, {"original_columns": original, "excluded_columns": excluded}


def drop_technical_columns(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    """Drop technical columns not useful for modeling.

    Config structure expected:
        {
            "params": {
                "manual_exclude": [col_names],
                "keep_for_aggregation": [col_names]
            }
        }
    Columns in both manual_exclude and keep_for_aggregation are preserved.
    """
    log.info("[3.1] drop_technical_columns called with %d columns", df.shape[1])

    params = config.get("params", {})
    manual_exclude = params.get("manual_exclude", [])
    keep_for_aggregation = params.get("keep_for_aggregation", [])

    log.debug("[3.1] drop_technical_columns: manual_exclude=%s, keep_for_aggregation=%s", manual_exclude, keep_for_aggregation)

    drop_set = set(manual_exclude) - set(keep_for_aggregation)
    to_drop = [c for c in drop_set if c in df.columns]

    # Warn about columns in manual_exclude that don't exist
    missing = [c for c in manual_exclude if c not in df.columns]
    if missing:
        log.warning("[3.1] drop_technical_columns: columns in manual_exclude not found in dataframe: %s", missing)

    # Warn about columns in keep_for_aggregation that don't exist
    missing_keep = [c for c in keep_for_aggregation if c not in df.columns]
    if missing_keep:
        log.warning("[3.1] drop_technical_columns: columns in keep_for_aggregation not found: %s", missing_keep)

    result_df = df.drop(columns=to_drop) if to_drop else df.copy()
    log.info("[3.1] drop_technical_columns dropped %d columns", len(to_drop))
    log.debug("[3.1] drop_technical_columns remaining columns: %d", result_df.shape[1])
    return result_df, {"dropped_columns": to_drop}


def remove_constant_features(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    """Remove columns with only one unique value (or <= threshold).

    Config structure expected:
        {
            "params": {
                "threshold_unique": int
            }
        }
    Default threshold is 1 (removes columns with 0 or 1 unique values).
    """
    log.info("[3.1] remove_constant_features called with %d columns", df.shape[1])

    params = config.get("params", {})
    threshold = params.get("threshold_unique", 1)
    log.debug("[3.1] remove_constant_features: threshold_unique=%d", threshold)

    to_drop = [c for c in df.columns if df[c].nunique(dropna=False) <= threshold]

    if to_drop:
        log.debug("[3.1] remove_constant_features: constant columns detected: %s", to_drop)
    else:
        log.debug("[3.1] remove_constant_features: no constant columns found")

    result_df = df.drop(columns=to_drop) if to_drop else df.copy()
    log.info("[3.1] remove_constant_features removed %d columns", len(to_drop))
    return result_df, {"removed_columns": to_drop}


def remove_duplicate_features(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    """Remove columns that are exact duplicates of another column.

    Config structure expected:
        {
            "params": {
                "strategy": str  # currently only "exact" is supported
            }
        }
    The strategy parameter is logged for traceability but only exact matching is implemented.
    """
    log.info("[3.1] remove_duplicate_features called with %d columns", df.shape[1])

    params = config.get("params", {})
    strategy = params.get("strategy", "exact")
    log.debug("[3.1] remove_duplicate_features: strategy='%s'", strategy)

    if strategy != "exact":
        log.warning("[3.1] remove_duplicate_features: unsupported strategy '%s' – falling back to 'exact'", strategy)

    columns = list(df.columns)
    removed = []
    seen = {}

    for col in columns:
        signature = tuple(df[col].fillna("_NaN_").tolist())
        if signature in seen:
            removed.append(col)
            log.debug("[3.1] remove_duplicate_features: column '%s' is duplicate of '%s'", col, seen[signature])
        else:
            seen[signature] = col

    result_df = df.drop(columns=removed) if removed else df.copy()

    log.info("[3.1] remove_duplicate_features removed %d duplicate columns", len(removed))
    if removed:
        log.debug("[3.1] remove_duplicate_features: removed columns: %s", removed)
    return result_df, {"removed_columns": removed, "strategy_used": strategy}

