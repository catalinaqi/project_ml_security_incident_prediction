# src/crispdm/feature/cleaning_transformer_feature.py
from __future__ import annotations

import numpy as np
import pandas as pd

from crispdm.common.dict_facade_common import dget
from crispdm.common.logging_adapter_common import get_logger

log = get_logger(__name__)


def handle_missing_data(
    df_train: pd.DataFrame, df_test: pd.DataFrame | None, methods_cfg: dict
) -> dict:
    """Handle missing data: median imputation for numerics, mode imputation for categoricals.

    Expected config structure (the whole missing_data_handling block):
        {
            "enabled": bool,
            "techniques": {
                "median_imputation": {
                    "enabled": bool,
                    "params": {
                        "numeric_only": bool,
                        "exclude_columns": [col1, col2, ...]
                    }
                },
                "mode_imputation": {
                    "enabled": bool,
                    "columns": [col1, col2],
                    "add_missing_flags": [flag1, flag2, ...]
                }
            }
        }
    """
    log.info("[3.2] handle_missing_data called – train shape=%s, test shape=%s",
             df_train.shape, df_test.shape if df_test is not None else None)
    log.debug("[3.2] handle_missing_data config received: %s", methods_cfg)

    if not methods_cfg.get("enabled", True):
        log.info("[3.2] handle_missing_data disabled – skipping")
        return {"applied": False}

    techniques = methods_cfg.get("techniques", {})
    report: dict = {"techniques": {}, "imputation_values": {}}

    # ---- Median imputation ----
    median_cfg = techniques.get("median_imputation", {})
    if median_cfg.get("enabled", True):
        params = median_cfg.get("params", {})
        numeric_only = params.get("numeric_only", True)
        exclude_cols = params.get("exclude_columns", [])

        log.debug("[3.2] median_imputation: numeric_only=%s, exclude_columns=%s", numeric_only, exclude_cols)

        if numeric_only:
            candidate_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
        else:
            candidate_cols = df_train.columns.tolist()

        # Remove excluded columns
        candidate_cols = [c for c in candidate_cols if c not in exclude_cols]

        impute_values = {}
        for col in candidate_cols:
            if col not in df_train.columns:
                log.warning("[3.2] median_imputation: column '%s' not found in train – skipping", col)
                continue
            if df_train[col].isna().any():
                med = df_train[col].median()
                impute_values[col] = med
                df_train[col].fillna(med, inplace=True)
                log.debug("[3.2] median_imputation: imputed column '%s' with median=%.4f", col, med)
                if df_test is not None and col in df_test.columns:
                    df_test[col].fillna(med, inplace=True)

        report["techniques"]["median_imputation"] = {
            "columns_imputed": list(impute_values.keys()),
            "count": len(impute_values),
            "values": {k: float(v) for k, v in impute_values.items()},
        }
        report["imputation_values"]["median"] = {k: float(v) for k, v in impute_values.items()}

        if not impute_values:
            log.info("[3.2] median_imputation: no columns required imputation")
    else:
        log.debug("[3.2] median_imputation disabled")

    # ---- Mode imputation ----
    mode_cfg = techniques.get("mode_imputation", {})
    if mode_cfg.get("enabled", True):
        columns = mode_cfg.get("columns", [])
        add_missing_flags = mode_cfg.get("add_missing_flags", [])

        log.debug("[3.2] mode_imputation: columns=%s, add_missing_flags=%s", columns, add_missing_flags)

        impute_values = {}
        flags_created = []

        for col in columns:
            if col not in df_train.columns:
                log.warning("[3.2] mode_imputation: column '%s' not found in train – skipping", col)
                continue
            flag_col = f"{col}_is_missing"
            df_train[flag_col] = df_train[col].isna().astype(int)
            flags_created.append(flag_col)
            log.debug("[3.2] mode_imputation: created flag '%s'", flag_col)
            if df_test is not None and col in df_test.columns:
                df_test[flag_col] = df_test[col].isna().astype(int)

            mode_val = df_train[col].mode(dropna=True)
            if len(mode_val) > 0:
                mode_val = mode_val.iloc[0]
                impute_values[col] = mode_val
                df_train[col].fillna(mode_val, inplace=True)
                log.debug("[3.2] mode_imputation: imputed column '%s' with mode='%s'", col, mode_val)
                if df_test is not None and col in df_test.columns:
                    df_test[col].fillna(mode_val, inplace=True)

        # Handle SuspicionLevel flags (always added, regardless of column list)
        suspicion_cols = [c for c in df_train.columns if "SuspicionLevel" in c]
        for col in suspicion_cols:
            flag_col = f"{col}_is_missing"
            if flag_col not in df_train.columns:
                df_train[flag_col] = df_train[col].isna().astype(int)
                flags_created.append(flag_col)
                log.debug("[3.2] mode_imputation: created missing flag for '%s'", col)
                if df_test is not None and col in df_test.columns:
                    df_test[flag_col] = df_test[col].isna().astype(int)

        # Add any additional flags specified in config that weren't already created
        for flag in add_missing_flags:
            if flag not in flags_created and flag not in df_train.columns:
                log.warning("[3.2] mode_imputation: requested flag '%s' not created – column not imputed", flag)

        # Combine created flags with configured ones (avoid duplicates)
        all_flags = list(dict.fromkeys(add_missing_flags + flags_created))

        report["techniques"]["mode_imputation"] = {
            "columns_imputed": list(impute_values.keys()),
            "count": len(impute_values),
            "flags_created": all_flags,
            "values": {k: str(v) for k, v in impute_values.items()},
        }
        report["imputation_values"]["mode"] = {k: str(v) for k, v in impute_values.items()}

        if not impute_values:
            log.info("[3.2] mode_imputation: no columns required imputation (flags may have been created)")
    else:
        log.debug("[3.2] mode_imputation disabled")

    log.info("[3.2] handle_missing_data finished – median_imputed=%d, mode_imputed=%d",
             len(report["techniques"].get("median_imputation", {}).get("columns_imputed", [])),
             len(report["techniques"].get("mode_imputation", {}).get("columns_imputed", [])))
    return report


def handle_categorical_noise(
    df_train: pd.DataFrame, df_test: pd.DataFrame | None, techniques_cfg: dict
) -> dict:
    """Handle categorical noise: sentinel cleanup and rare grouping.

    Expected config structure (the whole categorical_noise block):
        {
            "enabled": bool,
            "techniques": {
                "sentinel_cleanup": {
                    "enabled": bool,
                    "params": {
                        "columns": {
                            "OSFamily": {"sentinel_value": 0, "replace_with": null}
                        }
                    }
                },
                "rare_grouping": {
                    "enabled": bool,
                    "params": {
                        "min_freq": float,
                        "unseen_category": str,
                        "exclude_columns": [col1, col2]
                    }
                }
            }
        }
    """
    log.info("[3.2] handle_categorical_noise called")
    log.debug("[3.2] handle_categorical_noise config received: %s", techniques_cfg)

    if not techniques_cfg.get("enabled", True):
        log.info("[3.2] handle_categorical_noise disabled – skipping")
        return {"applied": False}

    techniques = techniques_cfg.get("techniques", {})
    report: dict = {"techniques": {}}

    # ---- Sentinel cleanup ----
    sentinel_cfg = techniques.get("sentinel_cleanup", {})
    if sentinel_cfg.get("enabled", True):
        params = sentinel_cfg.get("params", {})
        columns_cfg = params.get("columns", {})

        log.debug("[3.2] sentinel_cleanup: columns_cfg=%s", columns_cfg)

        replacements = []
        for col, col_cfg in columns_cfg.items():
            if col not in df_train.columns:
                log.warning("[3.2] sentinel_cleanup: column '%s' not found in train – skipping", col)
                continue
            sentinel = col_cfg.get("sentinel_value")
            replace = col_cfg.get("replace_with")

            mask_train = df_train[col] == sentinel
            df_train.loc[mask_train, col] = replace

            entry = {
                "column": col,
                "sentinel_value": sentinel,
                "replaced_with": replace,
                "train_count": int(mask_train.sum()),
            }

            if df_test is not None and col in df_test.columns:
                mask_test = df_test[col] == sentinel
                df_test.loc[mask_test, col] = replace
                entry["test_count"] = int(mask_test.sum())

            replacements.append(entry)
            log.info("[3.2] sentinel_cleanup: column '%s' replaced sentinel %s with %s in %d train rows",
                      col, sentinel, replace, mask_train.sum())

        report["techniques"]["sentinel_cleanup"] = {"replacements": replacements, "count": len(replacements)}
    else:
        log.debug("[3.2] sentinel_cleanup disabled")

    # ---- Rare grouping ----
    rare_cfg = techniques.get("rare_grouping", {})
    if rare_cfg.get("enabled", True):
        params = rare_cfg.get("params", {})
        min_freq = params.get("min_freq", 0.001)
        unseen = params.get("unseen_category", "OTHER")
        exclude_cols = params.get("exclude_columns", [])

        log.debug("[3.2] rare_grouping: min_freq=%s, unseen=%s, exclude_columns=%s", min_freq, unseen, exclude_cols)

        object_cols = df_train.select_dtypes(include=["object", "category"]).columns.tolist()
        included = [c for c in object_cols if c not in exclude_cols]

        if not included:
            log.warning("[3.2] rare_grouping: no object columns to process after exclusions")
            report["techniques"]["rare_grouping"] = {
                "min_freq": min_freq,
                "unseen_category": unseen,
                "columns_processed": [],
                "count": 0,
                "mapping_summary": {},
            }
        else:
            mapping = {}
            for col in included:
                freq = df_train[col].value_counts(normalize=True, dropna=False)
                valid = freq[freq >= min_freq].index.tolist()
                mapping[col] = valid

                rare_before = df_train[col].isin(valid).sum()
                df_train[col] = df_train[col].apply(
                    lambda x: x if pd.isna(x) or x in valid else unseen
                )
                rare_after = df_train[col].isin(valid).sum()
                log.debug("[3.2] rare_grouping: column '%s' valid categories=%d, before rare=%d, after rare=%d",
                          col, len(valid), rare_before, rare_after)

                if df_test is not None and col in df_test.columns:
                    df_test[col] = df_test[col].apply(
                        lambda x: x if pd.isna(x) or x in valid else unseen
                    )

            report["techniques"]["rare_grouping"] = {
                "min_freq": min_freq,
                "unseen_category": unseen,
                "columns_processed": included,
                "count": len(included),
                "mapping_summary": {col: len(v) for col, v in mapping.items()},
            }
    else:
        log.debug("[3.2] rare_grouping disabled")

    log.info("[3.2] handle_categorical_noise finished – sentinel_cleaned=%d, rare_grouped=%d",
             len(report["techniques"].get("sentinel_cleanup", {}).get("replacements", [])),
             len(report["techniques"].get("rare_grouping", {}).get("columns_processed", [])))
    return report


def handle_duplicates(
    df_train: pd.DataFrame, df_test: pd.DataFrame | None, techniques_cfg: dict
) -> dict:
    """Remove exact duplicate rows.

    Expected config structure (the whole duplicate_handling block):
        {
            "enabled": bool,
            "techniques": {
                "exact_duplicates": {
                    "enabled": bool,
                    "params": {
                        "subset": null,
                        "keep": "first"
                    }
                }
            }
        }
    """
    log.info("[3.2] handle_duplicates called – train size=%d, test size=%s", len(df_train),
             len(df_test) if df_test is not None else "None")
    log.debug("[3.2] handle_duplicates config received: %s", techniques_cfg)

    if not techniques_cfg.get("enabled", True):
        log.info("[3.2] handle_duplicates disabled – skipping")
        return {"applied": False}

    techniques = techniques_cfg.get("techniques", {})
    report: dict = {"techniques": {}}

    dup_cfg = techniques.get("exact_duplicates", {})
    if dup_cfg.get("enabled", True):
        params = dup_cfg.get("params", {})
        subset = params.get("subset", None)
        keep = params.get("keep", "first")

        log.debug("[3.2] exact_duplicates: subset=%s, keep=%s", subset, keep)

        before_train = len(df_train)
        df_train.drop_duplicates(subset=subset, keep=keep, inplace=True)
        removed_train = before_train - len(df_train)

        removed_test = 0
        if df_test is not None:
            before_test = len(df_test)
            df_test.drop_duplicates(subset=subset, keep=keep, inplace=True)
            removed_test = before_test - len(df_test)

        log.info("[3.2] exact_duplicates removed %d train rows, %d test rows", removed_train, removed_test)

        report["techniques"]["exact_duplicates"] = {
            "subset": subset,
            "keep": keep,
            "train_before": before_train,
            "train_after": len(df_train),
            "train_removed": removed_train,
            "test_before": before_test if df_test is not None else None,
            "test_after": len(df_test) if df_test is not None else None,
            "test_removed": removed_test,
        }
    else:
        log.debug("[3.2] exact_duplicates disabled")
        report["techniques"]["exact_duplicates"] = {"applied": False}

    return report

