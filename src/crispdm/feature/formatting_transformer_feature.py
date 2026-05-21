# src/crispdm/feature/formatting_transformer_feature.py
"""
Transformation functions for Phase 3.5 — Data Formatting.

Includes no_split (shuffle), type_casting (schema alignment, dtype optimization,
category conversion), and array_conversion (numpy output for sklearn).
"""
from __future__ import annotations

from typing import Any, Optional
import numpy as np
import pandas as pd

from crispdm.common.logging_adapter_common import get_logger

log = get_logger(__name__)


# ------------------------------------------------------------------
# 1) No-split clustering (shuffle only)
# ------------------------------------------------------------------
def no_split_clustering(
    df_train: pd.DataFrame,
    df_test: Optional[pd.DataFrame],
    tech_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], dict[str, Any]]:
    """Shuffle train/test DataFrames (no actual split).

    Clustering does not need a supervised train/test split, but we shuffle
    to break any time-series ordering artifacts.
    """

    if not tech_cfg.get("enabled", True):
        return df_train, df_test, {"applied": False}
    params = tech_cfg.get("params", {})

    random_state = params.get("random_state", 42)
    shuffle_flag = params.get("shuffle", True)
    strategy = params.get("strategy", "random")

    before_train = len(df_train)
    before_test = len(df_test) if df_test is not None else 0

    if shuffle_flag:
        df_train = df_train.sample(frac=1, random_state=random_state).reset_index(drop=True)
        if df_test is not None:
            df_test = df_test.sample(frac=1, random_state=random_state + 1).reset_index(drop=True)
        log.info("[no_split] shuffled train=%d rows, test=%d rows",
                 before_train, before_test)

    report = {
        "applied": True,
        "strategy": strategy,
        "shuffle": shuffle_flag,
        "random_state": random_state,
        "train_rows_before": before_train,
        "train_rows_after": len(df_train),
        "test_rows_before": before_test,
        "test_rows_after": len(df_test) if df_test is not None else 0,
    }
    return df_train, df_test, report


# ------------------------------------------------------------------
# 2) Type casting — schema alignment & dtype optimization
# ------------------------------------------------------------------
def type_casting(
    df_train: pd.DataFrame,
    df_test: Optional[pd.DataFrame],
    tech_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], dict[str, Any]]:
    """Align schemas between train/test and optimize dtypes.

    Handles:
      - Dropping columns that exist only in one DataFrame (e.g. 'Usage')
      - Converting object columns to ``category``
      - Downcasting float64 → float32
      - Converting binary flags to bool
    """
    if not tech_cfg.get("enabled", True):
        return df_train, df_test, {"applied": False}
    params = tech_cfg.get("params", {})

    df_train = df_train.copy()
    df_test = df_test.copy() if df_test is not None else None

    changes: dict[str, Any] = {}

    # --- 1. Drop columns only in test (schema alignment) ---
    drop_cols = params.get("drop_columns", [])
    dropped_in_train = [c for c in drop_cols if c in df_train.columns]
    dropped_in_test = [c for c in drop_cols if c in (df_test.columns if df_test is not None else [])]

    if dropped_in_train:
        df_train.drop(columns=dropped_in_train, inplace=True)
        log.info("[type_casting] dropped from train: %s", dropped_in_train)
    if dropped_in_test and df_test is not None:
        df_test.drop(columns=dropped_in_test, inplace=True)
        log.info("[type_casting] dropped from test: %s", dropped_in_test)

    changes["columns_dropped"] = {"train": dropped_in_train, "test": dropped_in_test}

    # --- 2. Ensure test has same columns as train (add missing with NaN) ---
    if df_test is not None:
        missing_in_test = [c for c in df_train.columns if c not in df_test.columns]
        for col in missing_in_test:
            df_test[col] = np.nan
            log.warning("[type_casting] added missing column '%s' to test with NaN", col)
        changes["columns_added_to_test"] = missing_in_test

        # Also drop columns in test not in train
        extra_in_test = [c for c in df_test.columns if c not in df_train.columns]
        if extra_in_test:
            df_test.drop(columns=extra_in_test, inplace=True)
            log.warning("[type_casting] dropped extra columns from test: %s", extra_in_test)
        changes["columns_dropped_from_test"] = extra_in_test

    # --- 3. Drop BIGINTs sin señal (drop_bigint_no_signal) ---
    drop_bigint = params.get("drop_bigint_no_signal", [])
    dropped_bigint_train = [c for c in drop_bigint if c in df_train.columns]
    dropped_bigint_test = [c for c in drop_bigint if c in (df_test.columns if df_test is not None else [])]
    if dropped_bigint_train:
        df_train.drop(columns=dropped_bigint_train, inplace=True)
        log.info("[type_casting] dropped BIGINTs from train: %s", dropped_bigint_train)
    if dropped_bigint_test and df_test is not None:
        df_test.drop(columns=dropped_bigint_test, inplace=True)
        log.info("[type_casting] dropped BIGINTs from test: %s", dropped_bigint_test)
    changes["drop_bigint_no_signal"] = {"train": dropped_bigint_train, "test": dropped_bigint_test}

    # --- 4. Object columns → frequency encoding ---
    freq_cols = params.get("object_frequency_encode", [])
    fit_on = params.get("fit_on", "train")
    freq_encoded = []
    freq_mappings = {}
    for col in freq_cols:
        if col not in df_train.columns:
            continue
        # Compute frequency map on train only
        freq_map = df_train[col].value_counts(normalize=True).to_dict()
        freq_mappings[col] = freq_map
        df_train[col] = df_train[col].map(freq_map).fillna(0.0).astype("float32")
        if df_test is not None and col in df_test.columns:
            df_test[col] = df_test[col].map(freq_map).fillna(0.0).astype("float32")
        freq_encoded.append(col)
        log.debug("[type_casting] frequency encoded '%s' (train unique=%d)", col, len(freq_map))
    if freq_encoded:
        log.info("[type_casting] frequency encoded columns: %s", freq_encoded)
    changes["object_frequency_encode"] = freq_encoded
    changes["frequency_mappings"] = {col: len(m) for col, m in freq_mappings.items()}

    # --- 5. float64 → float32 ---
    f64_to_f32 = params.get("float64_to_float32", [])
    converted_f32 = []
    for col in f64_to_f32:
        if col in df_train.columns and df_train[col].dtype == "float64":
            df_train[col] = df_train[col].astype("float32")
            converted_f32.append(col)
        if df_test is not None and col in df_test.columns and df_test[col].dtype == "float64":
            df_test[col] = df_test[col].astype("float32")
    if converted_f32:
        log.info("[type_casting] float64→float32: %s", converted_f32)
    changes["float64_to_float32"] = converted_f32

    # --- 6. Flag → bool ---
    flag_to_bool = params.get("flag_to_bool", [])
    converted_bool = []
    for col in flag_to_bool:
        if col in df_train.columns:
            df_train[col] = df_train[col].astype(bool)
            converted_bool.append(col)
        if df_test is not None and col in df_test.columns:
            df_test[col] = df_test[col].astype(bool)
    if converted_bool:
        log.info("[type_casting] flag→bool: %s", converted_bool)
    changes["flag_to_bool"] = converted_bool

    report = {
        "applied": True,
        "train_dtypes": {col: str(dtype) for col, dtype in df_train.dtypes.items()},
        "test_dtypes": {col: str(dtype) for col, dtype in df_test.dtypes.items()} if df_test is not None else None,
        "changes": changes,
    }
    return df_train, df_test, report


# ------------------------------------------------------------------
# 3) Array conversion — numpy for sklearn
# ------------------------------------------------------------------
def array_conversion(
    df_train: pd.DataFrame,
    df_test: Optional[pd.DataFrame],
    tech_cfg: dict[str, Any],
) -> tuple[np.ndarray, Optional[np.ndarray], dict[str, Any]]:
    """Convert DataFrames to numpy arrays with unified dtype.

    Returns (X_train, X_test, report). X_test may be None.
    """
    if not tech_cfg.get("enabled", True):
        return (
            df_train.values if isinstance(df_train, pd.DataFrame) else df_train,
            df_test.values if isinstance(df_test, pd.DataFrame) else df_test,
            {"applied": False},
        )

    params = tech_cfg.get("params", {})
    to_numpy = params.get("to_numpy", False)
    unified_dtype = params.get("unified_dtype", "float32")

    # Ensure all columns are numeric (convert category/object if needed)
    df_train_num = df_train.select_dtypes(include=[np.number])
    if df_test is not None:
        df_test_num = df_test.select_dtypes(include=[np.number])
    else:
        df_test_num = None

    non_numeric_train = set(df_train.columns) - set(df_train_num.columns)
    if non_numeric_train:
        log.warning(
            "[array_conversion] dropped non‑numeric columns from train: %s",
            list(non_numeric_train),
        )
    if df_test is not None:
        non_numeric_test = set(df_test.columns) - set(df_test_num.columns)
        if non_numeric_test:
            log.warning(
                "[array_conversion] dropped non‑numeric columns from test: %s",
                list(non_numeric_test),
            )

    if not to_numpy:
        return df_train_num, df_test_num, {
            "applied": True,
            "to_numpy": False,
            "columns_used": list(df_train_num.columns),
        }

    # Fill any remaining NaN with 0.0 before conversion
    X_train = df_train_num.fillna(0.0).values.astype(unified_dtype)
    X_test = (
        df_test_num.fillna(0.0).values.astype(unified_dtype)
        if df_test_num is not None
        else None
    )

    report = {
        "applied": True,
        "to_numpy": True,
        "unified_dtype": unified_dtype,
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape) if X_test is not None else None,
        "columns_used": list(df_train_num.columns),
        "non_numeric_dropped_train": list(non_numeric_train),
        "non_numeric_dropped_test": list(non_numeric_test) if df_test is not None else [],
    }
    log.info(
        "[array_conversion] train shape=%s, test shape=%s, dtype=%s",
        X_train.shape,
        X_test.shape if X_test is not None else None,
        unified_dtype,
    )
    return X_train, X_test, report


# ------------------------------------------------------------------
# 4) Save auxiliary labels (e.g. IncidentGrade) before they are dropped
# ------------------------------------------------------------------
def save_auxiliary_labels(
    df_train: pd.DataFrame,
    df_test: Optional[pd.DataFrame],
    tech_cfg: dict[str, Any],
) -> tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    dict[str, Any],
]:
    """Extract and encode a label column, returning separate DataFrames.

    The original DataFrames are NOT modified. The returned DataFrames contain
    a single column ``label`` with the encoded values. The caller is responsible
    for persistence.
    """
    if not tech_cfg.get("enabled", True):
        return None, None, {"applied": False}

    params = tech_cfg.get("params", {})
    column = params.get("column", "IncidentGrade")
    encoding: dict = params.get("encoding", {})
    save_train = params.get("save_train", False)
    save_test = params.get("save_test", False)

    # Validate column exists
    if column not in df_train.columns:
        log.warning("[save_auxiliary_labels] column '%s' not in train – returning None", column)
        return None, None, {"applied": False, "column": column, "error": "column_missing"}

    # Encode train labels
    labels_train = df_train[[column]].copy()
    labels_train["label"] = labels_train[column].map(encoding)
    # Handle any unmapped values: assign -1 or drop? Use -1.
    if labels_train["label"].isna().any():
        n_unmapped = int(labels_train["label"].isna().sum())
        log.warning("[save_auxiliary_labels] %d train rows had unmapped '%s' values – set to -1",
                     n_unmapped, column)
        labels_train["label"] = labels_train["label"].fillna(-1).astype(int)
    else:
        labels_train["label"] = labels_train["label"].astype(int)
    #labels_train = labels_train[["label"]]
    # Al no filtrar, labels_train mantiene tanto [column] (IncidentGrade) como 'label'

    labels_test = None
    if df_test is not None and column in df_test.columns:
        labels_test = df_test[[column]].copy()
        labels_test["label"] = labels_test[column].map(encoding)
        if labels_test["label"].isna().any():
            n_unmapped = int(labels_test["label"].isna().sum())
            log.warning("[save_auxiliary_labels] %d test rows had unmapped '%s' values – set to -1",
                         n_unmapped, column)
            labels_test["label"] = labels_test["label"].fillna(-1).astype(int)
        else:
            labels_test["label"] = labels_test["label"].astype(int)
        # Mantiene ambas columnas para consistencia con el set de entrenamiento
        #labels_test = labels_test[["label"]]
    elif df_test is not None:
        log.warning("[save_auxiliary_labels] column '%s' not in test – returning None for test", column)

    report = {
        "applied": True,
        "column": column,
        "encoding": encoding,
        "train_shape": list(labels_train.shape) if labels_train is not None else None,
        "test_shape": list(labels_test.shape) if labels_test is not None else None,
        "save_train": save_train,
        "save_test": save_test,
    }
    log.info("[save_auxiliary_labels] extracted labels from '%s': train=%s",
             column, labels_train.shape if labels_train is not None else None)
    return labels_train, labels_test, report
