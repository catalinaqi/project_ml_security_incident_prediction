# src/crispdm/feature/engineering_transformer_feature.py
"""
Transformation functions for Phase 3.3 — Data Transformation.

All functions accept a Pandas DataFrame and a params dict (from config),
and return (df_transformed, report_dict).
For scaling/encoding with fit/transform, they accept both train and test.
"""
# ============================================================================
# Transformation functions for Phase 3.3 — Data Transformation
# ============================================================================

from __future__ import annotations

from typing import Any, Optional
import pandas as pd
from sklearn.preprocessing import StandardScaler

from crispdm.common.logging_adapter_common import get_logger

log = get_logger(__name__)


# ------------------------------------------------------------------
# 1) Group‑by aggregations
# ------------------------------------------------------------------
def groupby_aggregations(
        df: pd.DataFrame,
        tech_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aggregate per IncidentId and join back."""
    if not tech_cfg.get("enabled", True):
        return df, {"applied": False}
    params = tech_cfg.get("params", {})

    groupby_col = params.get("groupby_column", "IncidentId")
    aggs = params.get("aggregations", {})

    if groupby_col not in df.columns:
        msg = f"Group‑by column '{groupby_col}' not found in DataFrame"
        log.error(msg)
        raise KeyError(msg)

    agg_dict: dict[str, Any] = {}
    for name, agg_spec in aggs.items():
        col = agg_spec.get("column")
        method = agg_spec.get("method")
        if col not in df.columns:
            log.warning("Column '%s' not found, skipping aggregation '%s'", col, name)
            continue
        if method == "size":
            agg_dict[name] = pd.NamedAgg(column=col, aggfunc="size")
        elif method == "nunique":
            agg_dict[name] = pd.NamedAgg(column=col, aggfunc="nunique")
        else:
            log.warning("Unknown aggregation method '%s' for '%s'", method, name)

    if not agg_dict:
        return df, {"applied": False, "warning": "No valid aggregations defined"}

    grouped = df.groupby(groupby_col, observed=True).agg(**agg_dict).reset_index()
    df = df.merge(grouped, on=groupby_col, how="left")

    report = {
        "applied": True,
        "aggregations": list(aggs.keys()),
        "original_rows": len(grouped),
        "columns_added": list(aggs.keys()),
    }
    log.info("[groupby_aggregations] added %s", report["columns_added"])
    return df, report


# ------------------------------------------------------------------
# 2) Temporal features – datetime extraction
# ------------------------------------------------------------------
def datetime_extraction(
        df: pd.DataFrame,
        tech_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extract features from a datetime column."""
    if not tech_cfg.get("enabled", True):
        return df, {"applied": False}
    params = tech_cfg.get("params", {})

    column = params.get("column")
    extracts = params.get("extract", [])
    drop_original = params.get("drop_original", False)

    if column not in df.columns:
        log.warning("Column '%s' not found for datetime extraction", column)
        return df, {"applied": False, "warning": f"Column '{column}' not found"}

    # Ensure datetime
    if df[column].dtype != "datetime64[ns]":
        try:
            df[column] = pd.to_datetime(df[column])
        except Exception as e:
            log.error("Failed to convert '%s' to datetime: %s", column, e)
            return df, {"applied": False, "error": str(e)}

    reports = {}
    for feature in extracts:
        if feature == "hour":
            new_col = f"{column}_{feature}"
            df[new_col] = df[column].dt.hour
            reports[feature] = {"column": new_col, "example": df[new_col].iloc[0]}
        else:
            log.warning("Unsupported extract '%s'", feature)

    if drop_original:
        df = df.drop(columns=[column])
        reports["dropped_original"] = True

    report = {
        "applied": True,
        "column": column,
        "extracted": extracts,
        "columns_added": [f"{column}_{f}" for f in extracts],
        "details": reports,
    }
    log.info("[datetime_extraction] extracted %s", report["columns_added"])
    return df, report


# ------------------------------------------------------------------
# 3) Numeric scaling – StandardScaler
# ------------------------------------------------------------------
def standard_scaling(
        df_train: pd.DataFrame,
        df_test: Optional[pd.DataFrame],
        tech_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[StandardScaler], dict[str, Any]]:
    """Fit StandardScaler on train, transform both."""
    if not tech_cfg.get("enabled", True):
        return df_train, df_test, None, {"applied": False}
    params = tech_cfg.get("params", {})

    log.debug("[standard_scaling] received params keys: %s", list(params.keys()))
    log.debug("[standard_scaling] columns value: %r", params.get("columns"))

    columns = params.get("columns", [])
    if not columns:
        log.warning("[standard_scaling] 'columns' list is empty – nothing to scale")
        return df_train, df_test, None, {"applied": False, "warning": "Empty columns list"}

    existing = [c for c in columns if c in df_train.columns]
    missing = [c for c in columns if c not in df_train.columns]
    if not existing:
        log.warning("[standard_scaling] None of the specified columns exist in df_train: %s", columns)
        return df_train, df_test, None, {"applied": False, "warning": f"No columns found in df_train: {columns}"}
    if missing:
        log.warning("[standard_scaling] Columns not found in train: %s", missing)

    scaler = StandardScaler()
    try:
        scaled_train = scaler.fit_transform(df_train[existing])
    except Exception as e:
        log.error("StandardScaler fit_transform failed: %s", e)
        return df_train, df_test, None, {"applied": False, "error": str(e)}

    df_train = df_train.copy()
    df_train[existing] = scaled_train

    if df_test is not None:
        try:
            scaled_test = scaler.transform(df_test[existing])
            df_test = df_test.copy()
            df_test[existing] = scaled_test
        except Exception as e:
            log.warning("Test transform failed: %s – skipping test scaling", e)

    mean = scaler.mean_.tolist()
    std = scaler.scale_.tolist()
    report = {
        "applied": True,
        "columns_scaled": existing,
        "means": dict(zip(existing, mean)),
        "stds": dict(zip(existing, std)),
        "n_samples_fit": len(df_train),
    }
    log.info("[standard_scaling] scaled %s", existing)
    return df_train, df_test, scaler, report


# ------------------------------------------------------------------
# 4) Frequency encoding
# ------------------------------------------------------------------
def frequency_encoding(
        df_train: pd.DataFrame,
        df_test: Optional[pd.DataFrame],
        tech_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], dict[str, Any], dict[str, Any]]:
    """Fit frequency encoding on train, transform both."""
    if not tech_cfg.get("enabled", True):
        return df_train, df_test, {}, {"applied": False}
    params = tech_cfg.get("params", {})

    log.debug("[frequency_encoding] received params keys: %s", list(params.keys()))
    log.debug("[frequency_encoding] columns value: %r", params.get("columns"))

    columns = params.get("columns", [])
    fillna_val = params.get("fillna_value", 0)

    if not columns:
        msg = "Empty columns list"
        log.warning("[frequency_encoding] %s", msg)
        return df_train, df_test, {}, {"applied": False, "warning": msg}

    existing_train = [c for c in columns if c in df_train.columns]
    missing_train = [c for c in columns if c not in df_train.columns]
    if not existing_train:
        log.warning("[frequency_encoding] None of specified columns exist: %s", columns)
        return df_train, df_test, {}, {"applied": False, "warning": "No columns found in train"}
    if missing_train:
        log.warning("[frequency_encoding] Columns missing in train: %s", missing_train)

    encoding_dict: dict[str, Any] = {}
    df_train = df_train.copy()
    if df_test is not None:
        df_test = df_test.copy()

    for col in existing_train:
        freq = df_train[col].value_counts(normalize=True)
        encoding_dict[col] = freq.to_dict()
        df_train[col] = df_train[col].map(freq).fillna(fillna_val)

        if df_test is not None:
            if col in df_test.columns:
                df_test[col] = df_test[col].map(freq).fillna(fillna_val)
            else:
                log.warning("[frequency_encoding] Column '%s' missing in test – skipped", col)

    report = {
        "applied": True,
        "columns_encoded": existing_train,
        "categories_per_column": {col: len(encoding_dict.get(col, {})) for col in existing_train},
        "fillna_value": fillna_val,
    }
    log.info("[frequency_encoding] encoded %s", existing_train)
    return df_train, df_test, encoding_dict, report


# ------------------------------------------------------------------
# 5) Ordinal encoding
# ------------------------------------------------------------------
def ordinal_encoding(
        df_train: pd.DataFrame,
        df_test: Optional[pd.DataFrame],
        tech_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], dict[str, Any], dict[str, Any]]:
    """Apply ordinal encoding using explicit category order."""
    if not tech_cfg.get("enabled", True):
        return df_train, df_test, {}, {"applied": False}
    params = tech_cfg.get("params", {})

    log.debug("[ordinal_encoding] received params keys: %s", list(params.keys()))
    log.debug("[ordinal_encoding] columns value: %r", params.get("columns"))
    log.debug("[ordinal_encoding] categories value: %r", params.get("categories"))

    columns = params.get("columns", [])
    categories = params.get("categories", {})

    if not columns:
        log.warning("[ordinal_encoding] 'columns' list is empty – no encoding applied")
        return df_train, df_test, {}, {"applied": False, "warning": "Empty columns list"}

    mappings: dict[str, dict[str, int]] = {}
    df_train = df_train.copy()
    if df_test is not None:
        df_test = df_test.copy()

    for col in columns:
        if col not in df_train.columns:
            log.warning("[ordinal_encoding] Column '%s' not in df_train – skipping", col)
            continue
        cat_order = categories.get(col, [])
        if not cat_order:
            log.warning("[ordinal_encoding] No category order for '%s'", col)
            continue

        mapping = {cat: idx for idx, cat in enumerate(cat_order)}
        mappings[col] = mapping

        df_train[col] = df_train[col].map(mapping).fillna(-1).astype(int)
        if df_test is not None:
            df_test[col] = df_test[col].map(mapping).fillna(-1).astype(int)

    report = {
        "applied": len(mappings) > 0,
        "columns_encoded": list(mappings.keys()),
        "mappings": mappings,
    }
    log.info("[ordinal_encoding] encoded %s", list(mappings.keys()))
    return df_train, df_test, mappings, report


# ------------------------------------------------------------------
# 6) Explicit column drop
# ------------------------------------------------------------------
def explicit_drop(
        df: pd.DataFrame,
        tech_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop specified columns."""
    if not tech_cfg.get("enabled", True):
        return df, {"applied": False}
    params = tech_cfg.get("params", {})

    columns = params.get("columns", [])
    reason = params.get("reason", "unspecified")

    to_drop = [c for c in columns if c in df.columns]
    not_found = [c for c in columns if c not in df.columns]

    if not to_drop:
        return df, {"applied": False, "warning": "No columns to drop were found"}

    df = df.drop(columns=to_drop)
    report = {
        "applied": True,
        "columns_dropped": to_drop,
        "columns_not_found": not_found,
        "reason": reason,
    }
    log.info("[explicit_drop] dropped %s", to_drop)
    return df, report


# ------------------------------------------------------------------
# 7) Missing flags passthrough
# ------------------------------------------------------------------
def passthrough_missing_flags(
        df: pd.DataFrame,
        tech_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep specified binary missing‑flag columns unchanged."""
    if not tech_cfg.get("enabled", True):
        return df, {"applied": False}
    params = tech_cfg.get("params", {})

    columns = params.get("columns", [])
    existing = [c for c in columns if c in df.columns]
    missing = [c for c in columns if c not in df.columns]

    if not existing:
        return df, {"applied": False, "warning": "No missing‑flag columns found"}

    stats = {}
    for col in existing:
        stats[col] = {
            "dtype": str(df[col].dtype),
            "n_unique": int(df[col].nunique()),
            "value_counts": df[col].value_counts(dropna=False).to_dict(),
            "na_count": int(df[col].isna().sum()),
        }

    report = {
        "applied": True,
        "columns_preserved": existing,
        "columns_not_found": missing,
        "stats": stats,
    }
    log.info("[passthrough_missing_flags] preserved %s", existing)
    return df, report