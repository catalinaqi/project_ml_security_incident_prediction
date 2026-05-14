# src/crispdm/data/profiling_utils_data.py
from __future__ import annotations

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# Stateless DataFrame analysis helpers for CRISP-DM Phase 2.
# Pure functions: no side effects, no I/O, no RunContext dependency.
# =============================================================================

"""Pure DataFrame profiling and drift-detection helpers for the CRISP-DM pipeline."""

from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency

from crispdm.configuration.enum_registry_config import ProblemType
from crispdm.common.logging_adapter_common import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level DEFAULT drift thresholds
# ---------------------------------------------------------------------------
PSI_WARN: float = 0.10
PSI_DRIFT: float = 0.20
KS_ALPHA: float = 0.05


# =============================================================================
# SECTION 1 — COLUMN SELECTION HELPERS
# =============================================================================


def numeric_cols(df: pd.DataFrame) -> list[str]:
    """Return column names with numeric dtype."""
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


# =============================================================================
# SECTION 2 — SCHEMA AND STATISTICS TABLES
# =============================================================================


def schema_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build per-column summary: dtype, null count, null %, cardinality."""
    return pd.DataFrame(
        {
            "column": list(df.columns),
            "dtype": [str(df[c].dtype) for c in df.columns],
            "n_null": [int(df[c].isna().sum()) for c in df.columns],
            "null_pct": [float(df[c].isna().mean() * 100.0) for c in df.columns],
            "n_unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
        }
    ).sort_values(["null_pct", "n_unique"], ascending=[False, False])


def describe_table(
        df: pd.DataFrame,
        include: Any = None,
        percentiles: Optional[list[float]] = None,
) -> pd.DataFrame:
    """Transpose df.describe() and promote index to column field."""
    kwargs: dict[str, Any] = {"include": include}
    if percentiles is not None:
        kwargs["percentiles"] = percentiles
    desc = df.describe(**kwargs).transpose()
    desc.insert(0, "column", desc.index.astype(str))
    return desc.reset_index(drop=True)


def min_max_mean_std(
        df: pd.DataFrame,
        *,
        numeric_only: bool = True,
        exclude_bigint_hashed: bool = False,
        metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Compute descriptive statistics for each column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    numeric_only : bool
        If True, only process numeric columns.
    exclude_bigint_hashed : bool
        If True, skip BIGINT columns with high cardinality (pseudo-categorical).
    metrics : list[str] | None
        If provided, only compute the specified metrics.
        Valid values: "count", "min", "max", "mean", "std".
        If None, computes all: min, max, mean, std.
    """
    cols = numeric_cols(df) if numeric_only else list(df.columns)

    # Filter out BIGINT pseudo-categorical columns if requested
    if exclude_bigint_hashed:
        filtered_cols = []
        for c in cols:
            n_unique = int(df[c].nunique(dropna=True))
            if n_unique > 1000:  # heuristic: high-cardinality integer → skip
                log.debug("[min_max_mean_std] excluding bigint-pseudo column: %s", c)
                continue
            filtered_cols.append(c)
        cols = filtered_cols

    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        has_data = bool(s.notna().any())
        row: dict[str, Any] = {"column": c}

        if metrics is not None:
            # Only compute requested metrics
            if "count" in metrics:
                row["count"] = int(s.notna().sum()) if has_data else 0
            if "min" in metrics:
                row["min"] = float(s.min()) if has_data else None
            if "max" in metrics:
                row["max"] = float(s.max()) if has_data else None
            if "mean" in metrics:
                row["mean"] = float(s.mean()) if has_data else None
            if "std" in metrics:
                row["std"] = float(s.std()) if has_data else None
        else:
            # Default: all classic metrics
            row["min"] = float(s.min()) if has_data else None
            row["max"] = float(s.max()) if has_data else None
            row["mean"] = float(s.mean()) if has_data else None
            row["std"] = float(s.std()) if has_data else None

        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# SECTION 3 — DUPLICATE DETECTION
# =============================================================================


def duplicates_summary(
        df: pd.DataFrame, *, subset: Optional[list[str]] = None, keep: str = "first"
) -> pd.DataFrame:
    """Produce one-row DataFrame with duplicate statistics."""
    dup_mask = df.duplicated(subset=subset, keep=keep)  # type: ignore[arg-type]
    n = len(df)
    return pd.DataFrame(
        [
            {
                "rows": int(n),
                "duplicates": int(dup_mask.sum()),
                "dup_pct": float(dup_mask.mean() * 100.0) if n else 0.0,
                "subset": str(subset),
                "keep": str(keep),
            }
        ]
    )


# =============================================================================
# SECTION 4 — STATISTICAL DRIFT DETECTION (PSI + KS)
# =============================================================================


def compute_psi(expected: pd.Series, actual: pd.Series, *, n_bins: int = 10) -> float:
    """Compute Population Stability Index between two numeric series."""
    # Step 1: Drop NaN and guard empty
    exp = expected.dropna()
    act = actual.dropna()
    if len(exp) == 0 or len(act) == 0:
        return 0.0

    # Step 2: Build shared bins
    combined_min = float(min(exp.min(), act.min()))
    combined_max = float(max(exp.max(), act.max()))
    if combined_min == combined_max:
        return 0.0
    bins = np.linspace(combined_min, combined_max, n_bins + 1)

    # Step 3: Normalised frequencies
    exp_counts, _ = np.histogram(exp, bins=bins)
    act_counts, _ = np.histogram(act, bins=bins)
    exp_pct = np.clip(exp_counts / len(exp), 1e-6, None)
    act_pct = np.clip(act_counts / len(act), 1e-6, None)

    # Step 4: PSI formula
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def compute_ks(expected: pd.Series, actual: pd.Series) -> tuple[float, float]:
    """Compute KS statistic and p-value."""
    exp = expected.dropna()
    act = actual.dropna()
    if len(exp) == 0 or len(act) == 0:
        return 0.0, 1.0
    result = stats.ks_2samp(exp.to_numpy(), act.to_numpy())
    return float(result.statistic), float(result.pvalue)


def build_drift_report(
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        numeric_feature_cols: list[str],
        task: ProblemType,
        *,
        target_col: Optional[str] = None,
        psi_drift: float = PSI_DRIFT,
        ks_alpha: float = KS_ALPHA,
        n_bins: int,
) -> pd.DataFrame:
    """Build per-column drift summary — task-aware."""
    PT = ProblemType
    rows: list[dict] = []

    # Step 1: PSI + KS on features
    for col in numeric_feature_cols:
        psi = compute_psi(df_train[col], df_test[col], n_bins=n_bins)
        ks_stat, ks_pval = compute_ks(df_train[col], df_test[col])
        rows.append(
            {
                "column": col,
                "check_type": "feature_psi_ks",
                "psi": round(psi, 4),
                "ks_stat": round(ks_stat, 4),
                "ks_pvalue": round(ks_pval, 4),
                "chi2_stat": None,
                "chi2_pvalue": None,
                "drift_flag": psi >= psi_drift or ks_pval < ks_alpha,
            }
        )

    # Step 2: Classification target Chi-square
    if task == PT.CLASSIFICATION and target_col and target_col in df_train.columns:
        train_counts = df_train[target_col].value_counts()
        test_counts = df_test[target_col].value_counts()
        all_cats = sorted(set(train_counts.index) | set(test_counts.index))
        contingency = pd.DataFrame(
            {
                "train": [train_counts.get(c, 0) for c in all_cats],
                "test": [test_counts.get(c, 0) for c in all_cats],
            }
        )
        chi2, p_chi2, _, _ = chi2_contingency(contingency.to_numpy())
        rows.append(
            {
                "column": target_col,
                "check_type": "target_chi2",
                "psi": None,
                "ks_stat": None,
                "ks_pvalue": None,
                "chi2_stat": round(float(chi2), 4),
                "chi2_pvalue": round(float(p_chi2), 4),
                "drift_flag": float(p_chi2) < ks_alpha,
            }
        )

    # Step 3: Regression target PSI + KS
    elif task == PT.REGRESSION and target_col and target_col in df_train.columns:
        psi = compute_psi(df_train[target_col], df_test[target_col], n_bins=n_bins)
        ks_stat, ks_pval = compute_ks(df_train[target_col], df_test[target_col])
        rows.append(
            {
                "column": target_col,
                "check_type": "target_psi_ks",
                "psi": round(psi, 4),
                "ks_stat": round(ks_stat, 4),
                "ks_pvalue": round(ks_pval, 4),
                "chi2_stat": None,
                "chi2_pvalue": None,
                "drift_flag": psi >= psi_drift or ks_pval < ks_alpha,
            }
        )

    # Step 4: Timeseries window drift
    elif task == PT.TIMESERIES:
        mid = len(df_train) // 2
        df_early = df_train.iloc[:mid]
        df_late = df_train.iloc[mid:]
        for col in numeric_feature_cols:
            psi = compute_psi(df_early[col], df_late[col], n_bins=n_bins)
            ks_stat, ks_pval = compute_ks(df_early[col], df_late[col])
            rows.append(
                {
                    "column": col,
                    "check_type": "timeseries_window_psi_ks",
                    "psi": round(psi, 4),
                    "ks_stat": round(ks_stat, 4),
                    "ks_pvalue": round(ks_pval, 4),
                    "chi2_stat": None,
                    "chi2_pvalue": None,
                    "drift_flag": psi >= psi_drift or ks_pval < ks_alpha,
                }
            )

    df_result = pd.DataFrame(rows)
    if df_result.empty:
        return df_result
    return df_result.sort_values("psi", ascending=False, na_position="last").reset_index(drop=True)


# =============================================================================
# SECTION 5 — NEW PHASE 2 TECHNIQUES (2.1, 2.2, 2.3, 2.4)
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# 2.1 — Data Acquisition
# ─────────────────────────────────────────────────────────────────────────────


def hierarchy_profiling_report(
        df: pd.DataFrame,
        *,
        hierarchy_levels: list[str],
        compute_ratios: bool = True,
        expected_ratios: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    """Analyze Evidence→Alert→Incident hierarchy."""
    # Step 1: Count unique per level
    counts = {lvl: int(df[lvl].nunique(dropna=True)) for lvl in hierarchy_levels if lvl in df.columns}

    # Step 2: Compute ratios
    ratios: dict[str, float] = {}
    if compute_ratios and len(hierarchy_levels) >= 2:
        for i in range(len(hierarchy_levels) - 1):
            parent = hierarchy_levels[i + 1]
            child = hierarchy_levels[i]
            if parent in counts and child in counts and counts[parent] > 0:
                ratios[f"{child}_per_{parent}"] = round(counts[child] / counts[parent], 2)

    # Step 3: Compare to expected
    deviations = {}
    if expected_ratios:
        for key, expected in expected_ratios.items():
            if key in ratios:
                deviations[f"{key}_deviation"] = round(ratios[key] - expected, 2)

    log.debug("[hierarchy_profiling_report] counts=%s ratios=%s", counts, ratios)
    return {"counts": counts, "ratios": ratios, "deviations": deviations}


# ─────────────────────────────────────────────────────────────────────────────
# 2.2 — Data Description
# ─────────────────────────────────────────────────────────────────────────────


def column_metadata_report(
        df: pd.DataFrame,
        *,
        include_cardinality: bool = True,
        include_dtypes: bool = True,
        cardinality_threshold: int = 1000,
        detect_bigint_pseudo_categorical: bool = True,
) -> pd.DataFrame:
    """Generate column metadata with cardinality classification.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    include_cardinality : bool
        Whether to classify cardinality as low/medium/high.
    include_dtypes : bool
        Whether to include detailed dtype string in the output.
    cardinality_threshold : int
        Threshold distinguishing medium from high cardinality.
    detect_bigint_pseudo_categorical : bool
        Whether to flag BIGINT columns that may be pseudo-categorical.
    """
    rows = []
    for col in df.columns:
        n_unique = int(df[col].nunique(dropna=True))
        dtype_str = str(df[col].dtype)

        # Step 1: Classify cardinality
        if include_cardinality:
            if n_unique <= 20:
                cardinality_class = "low"
            elif n_unique <= cardinality_threshold:
                cardinality_class = "medium"
            else:
                cardinality_class = "high"
        else:
            cardinality_class = None

        # Step 2: Detect BIGINT pseudo-categorical
        is_bigint_pseudo = False
        if detect_bigint_pseudo_categorical and dtype_str.startswith("int") and n_unique > cardinality_threshold:
            is_bigint_pseudo = True

        row = {
            "column": col,
            "n_unique": n_unique,
            "cardinality_class": cardinality_class,
            "is_bigint_pseudo_categorical": is_bigint_pseudo,
        }

        # Step 3: Include dtype info only if requested
        if include_dtypes:
            row["dtype"] = dtype_str

        rows.append(row)

    return pd.DataFrame(rows)


def schema_comparison_report(
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        *,
        check_column_names: bool = True,
        check_dtypes: bool = True,
        check_column_order: bool = False,
        strict_mode: bool = False,
        report_missing_in_train: bool = True,
        report_missing_in_test: bool = True,
) -> dict[str, Any]:
    """Compare schemas between train and test.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training DataFrame.
    df_test : pd.DataFrame
        Test/validation DataFrame.
    check_column_names : bool
        Whether to compare column name sets.
    check_dtypes : bool
        Whether to compare dtypes for common columns.
    check_column_order : bool
        Whether to verify exact column order match.
    strict_mode : bool
        If True, treat missing_in_train as incompatibility.
    report_missing_in_train : bool
        Whether to include missing_in_train list in output.
    report_missing_in_test : bool
        Whether to include missing_in_test list in output.
    """
    # Step 1: Column name comparison
    train_cols = set(df_train.columns)
    test_cols = set(df_test.columns)
    missing_in_test = sorted(train_cols - test_cols) if check_column_names else []
    missing_in_train = sorted(test_cols - train_cols) if check_column_names else []

    # Step 2: Dtype comparison
    dtype_mismatches = []
    if check_dtypes:
        common_cols = train_cols & test_cols
        for col in common_cols:
            if str(df_train[col].dtype) != str(df_test[col].dtype):
                dtype_mismatches.append(
                    {
                        "column": col,
                        "train_dtype": str(df_train[col].dtype),
                        "test_dtype": str(df_test[col].dtype),
                    }
                )

    # Step 3: Check column order
    column_order_match = True
    if check_column_order:
        column_order_match = list(df_train.columns) == list(df_test.columns)

    # Step 4: Build report
    is_compatible = len(missing_in_test) == 0 and len(dtype_mismatches) == 0 and column_order_match
    if strict_mode:
        is_compatible = is_compatible and len(missing_in_train) == 0

    report: dict[str, Any] = {
        "is_compatible": is_compatible,
        "dtype_mismatches": dtype_mismatches,
        "column_order_match": column_order_match if check_column_order else None,
    }

    if report_missing_in_test:
        report["missing_in_test"] = missing_in_test
    if report_missing_in_train:
        report["missing_in_train"] = missing_in_train

    log.debug("[schema_comparison] compatible=%s missing_in_test=%d", is_compatible, len(missing_in_test))
    return report


def multi_value_parser(
        df: pd.DataFrame,
        *,
        columns: list[str],
        delimiter: str = ",",
        max_values_per_row: int = 10,
        min_frequency: float = 0.001,
        report_top_n: int = 20,
) -> dict[str, Any]:
    """Parse comma-separated values in columns like MitreTechniques, Roles."""
    results = {}

    for col in columns:
        if col not in df.columns:
            continue

        # Step 1: Split and flatten
        all_values: list[str] = []
        for cell in df[col].dropna():
            parts = str(cell).split(delimiter)[:max_values_per_row]
            all_values.extend([p.strip() for p in parts if p.strip()])

        # Step 2: Frequency analysis
        value_counts = pd.Series(all_values).value_counts()
        total = len(all_values)
        filtered = value_counts[value_counts / total >= min_frequency]

        results[col] = {
            "total_values": total,
            "unique_values": len(value_counts),
            "top_values": filtered.head(report_top_n).to_dict(),
        }
        log.debug("[multi_value_parser] col=%s total=%d unique=%d", col, total, len(value_counts))

    return results


def cardinality_profiler(
        df: pd.DataFrame,
        *,
        target_columns: dict[str, list[str]],
        report_top_n: int = 20,
        flag_if_cardinality_exceeds: int = 10000,
) -> dict[str, Any]:
    """Classify columns by cardinality and flag high-cardinality columns."""
    results = {}

    for category, cols in target_columns.items():
        category_results = []
        for col in cols:
            if col not in df.columns:
                continue

            n_unique = int(df[col].nunique(dropna=True))
            value_counts = df[col].value_counts().head(report_top_n)

            category_results.append(
                {
                    "column": col,
                    "n_unique": n_unique,
                    "exceeds_threshold": n_unique > flag_if_cardinality_exceeds,
                    "top_values": value_counts.to_dict(),
                }
            )

        results[category] = category_results
        log.debug("[cardinality_profiler] category=%s cols=%d", category, len(category_results))

    return results


def target_distribution_report(
        df: pd.DataFrame,
        *,
        target_column: str,
        detect_none_values: bool = True,
        compute_imbalance_ratio: bool = True,
        report_value_counts: bool = True,
) -> dict[str, Any]:
    """Analyze target distribution with 'None' detection.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target_column : str
        Name of the target column.
    detect_none_values : bool
        Whether to detect 'None' string values.
    compute_imbalance_ratio : bool
        Whether to compute the imbalance ratio.
    report_value_counts : bool
        Whether to include value counts in the output dictionary.
    """
    if target_column not in df.columns:
        return {}

    # Step 1: Value counts
    value_counts = df[target_column].value_counts(dropna=False)

    # Step 2: Detect 'None' string values
    has_none_string = False
    none_count = 0
    if detect_none_values:
        none_mask = df[target_column].astype(str).str.lower() == "none"
        none_count = int(none_mask.sum())
        has_none_string = none_count > 0

    # Step 3: Imbalance ratio
    imbalance_ratio = None
    if compute_imbalance_ratio and len(value_counts) > 1:
        imbalance_ratio = round(float(value_counts.max() / value_counts.min()), 2)

    log.debug(
        "[target_distribution_report] target=%s classes=%d none_count=%d",
        target_column,
        len(value_counts),
        none_count,
    )

    report: dict[str, Any] = {
        "target_column": target_column,
        "has_none_string": has_none_string,
        "none_count": none_count,
        "imbalance_ratio": imbalance_ratio,
    }

    if report_value_counts:
        report["value_counts"] = value_counts.to_dict()

    return report


def detect_id_columns(
        df: pd.DataFrame, *, id_patterns: list[str], uniqueness_min: float = 0.95
) -> list[str]:
    """Detect identifier columns with uniqueness > threshold."""
    id_cols = []

    for col in df.columns:
        # Step 1: Check pattern match
        matches_pattern = any(
            col.endswith(pattern.replace("*", "")) or col == pattern for pattern in id_patterns
        )

        # Step 2: Check uniqueness
        uniqueness = df[col].nunique(dropna=True) / len(df) if len(df) > 0 else 0.0

        if matches_pattern and uniqueness >= uniqueness_min:
            id_cols.append(col)

    log.debug("[detect_id_columns] found=%d cols=%s", len(id_cols), id_cols)
    return id_cols


def entity_conditional_sparsity(
        df: pd.DataFrame,
        *,
        entity_column: str,
        conditional_columns: dict[str, list[str]],
        report_null_percentages: bool = True,
) -> dict[str, Any]:
    """Detect EntityType-dependent NULL patterns."""
    if entity_column not in df.columns:
        return {}

    results = {}

    for entity_type, cols in conditional_columns.items():
        entity_mask = df[entity_column] == entity_type
        n_rows = int(entity_mask.sum())

        if n_rows == 0:
            continue

        col_nulls = {}
        if report_null_percentages:
            for col in cols:
                if col in df.columns:
                    null_pct = round(float(df.loc[entity_mask, col].isna().mean() * 100), 2)
                    col_nulls[col] = null_pct

        results[entity_type] = {"n_rows": n_rows, "null_percentages": col_nulls}
        log.debug("[entity_conditional_sparsity] entity=%s rows=%d", entity_type, n_rows)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2.3 — Data Quality Verification
# ─────────────────────────────────────────────────────────────────────────────


def completeness_report(
        df: pd.DataFrame,
        *,
        include_patterns: bool = True,
        show_top_columns: int = 50,
        distinguish_null_by_design: bool = True,
) -> dict[str, Any]:
    """Generate completeness report with null patterns."""
    # Step 1: Null counts per column
    null_df = pd.DataFrame(
        {
            "column": list(df.columns),
            "n_null": [int(df[c].isna().sum()) for c in df.columns],
            "null_pct": [round(float(df[c].isna().mean() * 100), 2) for c in df.columns],
        }
    ).sort_values("null_pct", ascending=False)

    # Step 2: Overall stats
    total_nulls = int(null_df["n_null"].sum())
    total_cells = len(df) * len(df.columns)
    overall_null_pct = round(100 * total_nulls / total_cells, 2) if total_cells > 0 else 0.0

    log.debug("[completeness_report] total_nulls=%d overall_pct=%.2f", total_nulls, overall_null_pct)

    return {
        "total_nulls": total_nulls,
        "overall_null_pct": overall_null_pct,
        "top_columns": null_df.head(show_top_columns).to_dict(orient="records"),
    }


def detect_sentinel_values(
        df: pd.DataFrame, *, sentinel_values: list[int | float], check_columns: list[str]
) -> dict[str, Any]:
    """Detect sentinel values like -1, 999, 9999."""
    results = {}

    for col in check_columns:
        if col not in df.columns:
            continue

        col_results = {}
        for sentinel in sentinel_values:
            count = int((df[col] == sentinel).sum())
            if count > 0:
                col_results[str(sentinel)] = count

        if col_results:
            results[col] = col_results

    log.debug("[detect_sentinel_values] cols_with_sentinels=%d", len(results))
    return results


def crosstab_leakage_analysis(
        df: pd.DataFrame,
        *,
        target_column: str,
        suspect_columns: list[str],
        normalize: str = "index",
        leakage_threshold: float = 0.95,
) -> dict[str, Any]:
    """Detect leakage via crosstab correlation > threshold."""
    if target_column not in df.columns:
        return {}

    leakage_suspects = []

    for col in suspect_columns:
        if col not in df.columns or col == target_column:
            continue

        # Step 1: Build crosstab
        ct = pd.crosstab(df[col], df[target_column], normalize=normalize)

        # Step 2: Check if any column has correlation > threshold
        max_corr = float(ct.max().max()) if not ct.empty else 0.0

        if max_corr >= leakage_threshold:
            leakage_suspects.append({"column": col, "max_correlation": round(max_corr, 4)})

    log.debug("[crosstab_leakage_analysis] suspects=%d", len(leakage_suspects))
    return {"leakage_suspects": leakage_suspects}


def post_triage_detector(
        df: pd.DataFrame, *, high_missingness_threshold: float = 0.9, suspect_columns: list[str]
) -> list[str]:
    """Detect features likely generated after incident triage."""
    post_triage_cols = []

    for col in suspect_columns:
        if col not in df.columns:
            continue

        null_rate = float(df[col].isna().mean())
        if null_rate >= high_missingness_threshold:
            post_triage_cols.append(col)

    log.debug("[post_triage_detector] found=%d", len(post_triage_cols))
    return post_triage_cols


def timestamp_range_validator(
        df: pd.DataFrame,
        *,
        timestamp_column: str,
        expected_min_days: int,
        detect_timezone_issues: bool = True,
) -> dict[str, Any]:
    """Verify observation window duration."""
    if timestamp_column not in df.columns:
        return {}

    # Step 1: Convert to datetime
    ts = pd.to_datetime(df[timestamp_column], errors="coerce")
    ts_clean = ts.dropna()

    if len(ts_clean) == 0:
        return {"valid": False, "reason": "no_valid_timestamps"}

    # Step 2: Compute range
    min_ts = ts_clean.min()
    max_ts = ts_clean.max()
    actual_days = (max_ts - min_ts).days

    # Step 3: Validation
    is_valid = actual_days >= expected_min_days

    log.debug(
        "[timestamp_range_validator] actual_days=%d expected=%d valid=%s",
        actual_days,
        expected_min_days,
        is_valid,
    )

    return {
        "valid": is_valid,
        "actual_days": actual_days,
        "expected_min_days": expected_min_days,
        "min_timestamp": str(min_ts),
        "max_timestamp": str(max_ts),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2.4 — Exploratory Analysis
# ─────────────────────────────────────────────────────────────────────────────


def column_catalog_by_roles(
        df: pd.DataFrame,
        *,
        roles: dict[str, list[str]],
        include_created_in_phase_2: bool = False,
        categorize_by_role: bool = True,
) -> dict[str, Any]:
    """Categorize columns by their analytical role.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    roles : dict[str, list[str]]
        Dictionary mapping role names to lists of column names.
    include_created_in_phase_2 : bool
        Whether to include columns created in Phase 2 (not yet available).
    categorize_by_role : bool
        If True, includes an 'uncategorized' list for columns not matched
        to any defined role.
    """
    catalog = {}

    for role_name, col_list in roles.items():
        present_cols = [c for c in col_list if c in df.columns]
        catalog[role_name] = present_cols

    # Report uncategorized columns if requested
    if categorize_by_role:
        all_categorized: set[str] = set()
        for col_list in roles.values():
            all_categorized.update(col_list)
        uncategorized = [c for c in df.columns if c not in all_categorized]
        catalog["uncategorized"] = uncategorized
        log.debug("[column_catalog_by_roles] uncategorized=%d", len(uncategorized))

    log.debug("[column_catalog_by_roles] roles=%d", len(catalog))
    return catalog