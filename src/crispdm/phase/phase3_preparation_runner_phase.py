# src/crispdm/phase/phase3_preparation_runner_phase.py
from __future__ import annotations

from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from crispdm.common.context_facade_common import RunContext
from crispdm.common.dict_facade_common import dget, enabled
from crispdm.common.logging_adapter_common import get_logger
from crispdm.configuration.enum_registry_config import StepsPhase
from crispdm.data.persist_persister_data import save_json
from crispdm.feature.selection_selector_feature import (
    bigint_cleanup,
    manual_include_exclude,
    drop_technical_columns,
    remove_constant_features,
    remove_duplicate_features,
)
from crispdm.feature.cleaning_transformer_feature import (
    handle_missing_data,
    handle_categorical_noise,
    handle_duplicates,
)
from crispdm.reporting.artifact_persister_reporting import save_figure
from crispdm.registry.generator_registry_registry import write_output_artifacts

# Add these imports at the top (existing imports remain)
from crispdm.feature.engineering_transformer_feature import (
    groupby_aggregations,
    datetime_extraction,
    standard_scaling,
    frequency_encoding,
    ordinal_encoding,
    explicit_drop,
    passthrough_missing_flags,
)
from crispdm.feature.formatting_transformer_feature import (
    no_split_clustering,
    type_casting,
    array_conversion,
)

log = get_logger(__name__)


# =============================================================================
# Public entry points — delegate to services
# =============================================================================


def run_step_3_1(ctx: RunContext) -> RunContext:
    """Execute CRISP-DM Phase 3.1 — Data Selection.

    Delegates to :class:`SelectionService` for all transformations.
    Persists intermediate JSON reports and an overview plot.
    """
    step_key = StepsPhase.STEP_3_1.value
    step_cfg: dict[str, Any] = dget(
        ctx.config.phases.phase3_data_preparation.steps, step_key, {}
    )
    if not enabled(step_cfg, default=True):
        log.info("[3.1] step disabled – skipping")
        return ctx

    if ctx.df_train is None:
        raise RuntimeError("[3.1] df_train is None — run Phase 2 first")

    log.info("[3.1] start train shape=%s", ctx.df_train.shape)
    methods: dict[str, Any] = step_cfg.get("methods", {})
    # 1. Sentinel removal
    sentinel_cfg = methods.get("sentinel_removal", {}).get("techniques", {}).get("bigint_cleanup", {})
    df, report_1 = bigint_cleanup(ctx.df_train, sentinel_cfg)
    ctx.df_train = df
    if output_path := sentinel_cfg.get("output"):
        save_json(report_1, ctx.phase3_dir / output_path)

    # 2. Dataset definition
    dataset_cfg = methods.get("dataset_definition", {}).get("techniques", {})
    man_cfg = dataset_cfg.get("manual_include_exclude", {})
    if enabled(man_cfg):
        df, report_2a = manual_include_exclude(ctx.df_train, man_cfg)
        ctx.df_train = df
        if output_path := man_cfg.get("output"):
            save_json(report_2a, ctx.phase3_dir / output_path)
    tech_cfg = dataset_cfg.get("drop_technical_columns", {})
    if enabled(tech_cfg):
        df, report_2b = drop_technical_columns(ctx.df_train, tech_cfg)
        ctx.df_train = df
        if output_path := tech_cfg.get("output"):
            save_json(report_2b, ctx.phase3_dir / output_path)

    # 3. Feature selection
    feat_cfg = methods.get("feature_selection", {}).get("techniques", {})
    const_cfg = feat_cfg.get("remove_constant_features", {})
    if enabled(const_cfg):
        df, report_3a = remove_constant_features(ctx.df_train, const_cfg)
        ctx.df_train = df
        if output_path := const_cfg.get("output"):
            save_json(report_3a, ctx.phase3_dir / output_path)
    dup_cfg = feat_cfg.get("remove_duplicate_features", {})
    if enabled(dup_cfg):
        df, report_3b = remove_duplicate_features(ctx.df_train, dup_cfg)
        ctx.df_train = df
        if output_path := dup_cfg.get("output"):
            save_json(report_3b, ctx.phase3_dir / output_path)

    # 4. Save final feature list
    output_artifacts = step_cfg.get("output_artifacts", {})
    if path := output_artifacts.get("final_features"):
        final_features = sorted(ctx.df_train.columns.tolist())
        save_json(
            {"final_features": final_features, "count": len(final_features)},
            ctx.phase3_dir / path,
            )

    # 5. Generate overview plot
    if path := output_artifacts.get("overview_plot"):
        n_initial = ctx.df_train.shape[1]
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.barh(["Features"], [n_initial], color="steelblue", label="Initial")
        ax.barh(["Features"], [n_initial], color="orange", alpha=0.6, label="After Selection")
        ax.set_xlabel("Number of Features")
        ax.set_title("Phase 3.1 — Feature Selection Overview")
        ax.legend()
        plt.tight_layout()
        save_figure(fig, out_path=ctx.phase3_dir / path, dpi=150)

    log.info("[3.1] done train shape=%s", ctx.df_train.shape)
    write_output_artifacts(ctx, step_key=step_key, step_cfg=step_cfg,
                           df_train=ctx.df_train, df_test=ctx.df_test)
    return ctx


def run_step_3_2(ctx: RunContext) -> RunContext:
    """Execute CRISP-DM Phase 3.2 — Data Cleaning.

    Delegates to feature functions for all transformations.
    Persists intermediate JSON reports (from each technique's 'output' field)
    and a consolidated cleaning summary.
    """
    step_key = StepsPhase.STEP_3_2.value
    step_cfg: dict[str, Any] = dget(
        ctx.config.phases.phase3_data_preparation.steps, step_key, {}
    )
    log.debug("[3.2] step_cfg from config: %s", step_cfg)

    if not enabled(step_cfg, default=True):
        log.info("[3.2] step disabled – skipping")
        return ctx

    if ctx.df_train is None:
        raise RuntimeError("[3.2] df_train is None — run Phase 3.1 first")

    log.info("[3.2] start train shape=%s, test shape=%s",
             ctx.df_train.shape,
             ctx.df_test.shape if ctx.df_test is not None else None)

    methods: dict[str, Any] = step_cfg.get("methods", {})

    # ------------------------------------------------------------------
    # 1. Missing data handling
    # ------------------------------------------------------------------
    missing_cfg = methods.get("missing_data_handling", {})
    if enabled(missing_cfg):
        log.debug("[3.2] calling handle_missing_data with config keys: %s", list(missing_cfg.keys()))
        missing_report = handle_missing_data(ctx.df_train, ctx.df_test, missing_cfg)
    else:
        log.info("[3.2] missing_data_handling disabled – skipping")
        missing_report = {"applied": False}

    # Persist intermediate reports from techniques inside missing_data_handling
    missing_techniques = missing_cfg.get("techniques", {})
    for tech_name, tech_cfg in missing_techniques.items():
        if output_path := tech_cfg.get("output"):
            # Extract the relevant sub-report
            tech_report = missing_report.get("techniques", {}).get(tech_name, {})
            save_json(tech_report, ctx.phase3_dir / output_path)
            log.debug("[3.2] saved %s report to %s", tech_name, output_path)

    # ------------------------------------------------------------------
    # 2. Categorical noise
    # ------------------------------------------------------------------
    cat_cfg = methods.get("categorical_noise", {})
    if enabled(cat_cfg):
        log.debug("[3.2] calling handle_categorical_noise with config keys: %s", list(cat_cfg.keys()))
        cat_report = handle_categorical_noise(ctx.df_train, ctx.df_test, cat_cfg)
    else:
        log.info("[3.2] categorical_noise disabled – skipping")
        cat_report = {"applied": False}

    # Persist intermediate reports from techniques inside categorical_noise
    cat_techniques = cat_cfg.get("techniques", {})
    for tech_name, tech_cfg in cat_techniques.items():
        if output_path := tech_cfg.get("output"):
            tech_report = cat_report.get("techniques", {}).get(tech_name, {})
            if tech_report:
                save_json(tech_report, ctx.phase3_dir / output_path)
                log.debug("[3.2] saved %s report to %s", tech_name, output_path)

    # ------------------------------------------------------------------
    # 3. Duplicate handling
    # ------------------------------------------------------------------
    dup_cfg = methods.get("duplicate_handling", {})
    if enabled(dup_cfg):
        log.debug("[3.2] calling handle_duplicates with config keys: %s", list(dup_cfg.keys()))
        dup_report = handle_duplicates(ctx.df_train, ctx.df_test, dup_cfg)
    else:
        log.info("[3.2] duplicate_handling disabled – skipping")
        dup_report = {"applied": False}

    # Persist intermediate reports from techniques inside duplicate_handling
    dup_techniques = dup_cfg.get("techniques", {})
    for tech_name, tech_cfg in dup_techniques.items():
        if output_path := tech_cfg.get("output"):
            tech_report = dup_report.get("techniques", {}).get(tech_name, {})
            if tech_report:
                save_json(tech_report, ctx.phase3_dir / output_path)
                log.debug("[3.2] saved %s report to %s", tech_name, output_path)

    # ------------------------------------------------------------------
    # 4. Build and persist consolidated cleaning summary
    # ------------------------------------------------------------------
    output_artifacts = step_cfg.get("output_artifacts", {})
    cleaning_summary = {
        "step": "3.2",
        "missing_data": missing_report,
        "categorical_noise": cat_report,
        "duplicate_handling": dup_report,
        "train_shape_after": list(ctx.df_train.shape),
        "test_shape_after": list(ctx.df_test.shape) if ctx.df_test is not None else None,
    }
    if path := output_artifacts.get("cleaning_summary"):
        save_json(cleaning_summary, ctx.phase3_dir / path)
        log.info("[3.2] saved consolidated cleaning summary to %s", path)

    log.info("[3.2] done train shape=%s", ctx.df_train.shape)
    write_output_artifacts(ctx, step_key=step_key, step_cfg=step_cfg,
                           df_train=ctx.df_train, df_test=ctx.df_test)
    return ctx




# =============================================================================
# (existing code before)

# =============================================================================
# STEP 3.3 — DATA TRANSFORMATION
# =============================================================================
def run_step_3_3(ctx: RunContext) -> RunContext:
    """Execute CRISP-DM Phase 3.3 — Data Transformation."""
    step_key = StepsPhase.STEP_3_3.value
    step_cfg: dict[str, Any] = dget(
        ctx.config.phases.phase3_data_preparation.steps, step_key, {}
    )
    if not enabled(step_cfg, default=True):
        log.info("[3.3] step disabled – skipping")
        return ctx

    if ctx.df_train is None:
        raise RuntimeError("[3.3] df_train is None — run Phase 3.2 first")

    log.info("[3.3] start train shape=%s", ctx.df_train.shape)
    methods: dict[str, Any] = step_cfg.get("methods", {})

    # -----------------------------------------------------------------
    # 1. Feature engineering
    # -----------------------------------------------------------------
    fe_tech = methods.get("feature_engineering", {}).get("techniques", {}).get("groupby_aggregations", {})
    if fe_tech.get("enabled", True):
        ctx.df_train, report_fe = groupby_aggregations(ctx.df_train, fe_tech)
        if output_path := fe_tech.get("output"):
            save_json(report_fe, ctx.phase3_dir / output_path)
    else:
        log.info("[3.3] groupby_aggregations disabled")

    # -----------------------------------------------------------------
    # 2. Temporal features
    # -----------------------------------------------------------------
    temp_tech = methods.get("temporal_features", {}).get("techniques", {}).get("datetime_extraction", {})
    if temp_tech.get("enabled", True):
        ctx.df_train, report_temp = datetime_extraction(ctx.df_train, temp_tech)
        # Save PNG in the technique's output path
        img_path = temp_tech.get("output")  # "3.3.temporal_features.datetime_extraction.distribution.png"
        output_artifacts = step_cfg.get("output_artifacts", {})

        if "Timestamp_hour" in ctx.df_train.columns:
            # --- 1. Generate and save temporal distribution plot ---
            if img_path:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(ctx.df_train["Timestamp_hour"], bins=24, color="steelblue", edgecolor="black")
                ax.set_xlabel("Hour of Day")
                ax.set_ylabel("Frequency")
                ax.set_title("Phase 3.3 — Temporal Distribution of Alerts")
                ax.set_xticks(range(0, 24, 2))
                plt.tight_layout()
                save_figure(fig, out_path=ctx.phase3_dir / img_path, dpi=150)
                log.info("[3.3] saved temporal distribution plot to %s", img_path)

            # --- 2. Build consolidated temporal report (extraction + stats) ---
            temporal_report = {
                "extraction": report_temp,  # contains applied, columns_added, etc.
                "hour_statistics": {
                    "mean": float(ctx.df_train["Timestamp_hour"].mean()),
                    "std": float(ctx.df_train["Timestamp_hour"].std()),
                    "min": int(ctx.df_train["Timestamp_hour"].min()),
                    "max": int(ctx.df_train["Timestamp_hour"].max()),
                    "count": len(ctx.df_train),
                },
            }

            # --- 3. Save consolidated JSON in output_artifacts ---
            if path_stats := output_artifacts.get("temporal_stats"):
                save_json(temporal_report, ctx.phase3_dir / path_stats)
                log.info("[3.3] saved temporal stats to %s", path_stats)
        else:
            log.warning("[3.3] Timestamp_hour not found after extraction – skipping plot and stats")
            # Still save the extraction report even without the hour column
            if path_stats := output_artifacts.get("temporal_stats"):
                save_json({"extraction": report_temp, "hour_statistics": None},
                          ctx.phase3_dir / path_stats)
    else:
        log.info("[3.3] datetime_extraction disabled")

    # -----------------------------------------------------------------
    # 3. Numeric scaling
    # -----------------------------------------------------------------
    ns_tech = methods.get("numeric_scaling", {}).get("techniques", {}).get("standard_scaling", {})
    scaler = None
    if ns_tech.get("enabled", True):
        ctx.df_train, ctx.df_test, scaler, report_ns = standard_scaling(ctx.df_train, ctx.df_test, ns_tech)
        if output_path := ns_tech.get("output"):
            save_json(report_ns, ctx.phase3_dir / output_path)
    else:
        log.info("[3.3] standard_scaling disabled")

    # -----------------------------------------------------------------
    # 4. Frequency encoding
    # -----------------------------------------------------------------
    freq_tech = methods.get("feature_scaling", {}).get("techniques", {}).get("frequency_encoding", {})
    if freq_tech.get("enabled", True):
        ctx.df_train, ctx.df_test, encoding_dict, report_freq = frequency_encoding(ctx.df_train, ctx.df_test, freq_tech)
        if output_path := freq_tech.get("output"):
            save_json(report_freq, ctx.phase3_dir / output_path)
    else:
        log.info("[3.3] frequency_encoding disabled")

    # -----------------------------------------------------------------
    # 5. Ordinal encoding
    # -----------------------------------------------------------------
    ord_tech = methods.get("encoding", {}).get("techniques", {}).get("ordinal_encoding", {})
    if ord_tech.get("enabled", True):
        ctx.df_train, ctx.df_test, mappings, report_ord = ordinal_encoding(ctx.df_train, ctx.df_test, ord_tech)
        if output_path := ord_tech.get("output"):
            save_json(report_ord, ctx.phase3_dir / output_path)
    else:
        log.info("[3.3] ordinal_encoding disabled")

    # -----------------------------------------------------------------
    # 6. Column dropping
    # -----------------------------------------------------------------
    drop_tech = methods.get("column_dropping", {}).get("techniques", {}).get("explicit_drop", {})
    if drop_tech.get("enabled", True):
        ctx.df_train, report_drop = explicit_drop(ctx.df_train, drop_tech)
        if output_path := drop_tech.get("output"):
            save_json(report_drop, ctx.phase3_dir / output_path)
        if ctx.df_test is not None:
            ctx.df_test, _ = explicit_drop(ctx.df_test, drop_tech)
    else:
        log.info("[3.3] explicit_drop disabled")

    # -----------------------------------------------------------------
    # 7. Missing flags passthrough
    # -----------------------------------------------------------------
    miss_tech = methods.get("missing_flags", {}).get("techniques", {}).get("passthrough", {})
    if miss_tech.get("enabled", True):
        ctx.df_train, report_miss = passthrough_missing_flags(ctx.df_train, miss_tech)
        if output_path := miss_tech.get("output"):
            save_json(report_miss, ctx.phase3_dir / output_path)
        if ctx.df_test is not None:
            ctx.df_test, _ = passthrough_missing_flags(ctx.df_test, miss_tech)
    else:
        log.info("[3.3] missing_flags passthrough disabled")

    # -----------------------------------------------------------------
    # 8. Consolidated reports
    # -----------------------------------------------------------------
    output_artifacts = step_cfg.get("output_artifacts", {})
    if path := output_artifacts.get("final_feature_schema"):
        schema = {
            "columns": sorted(ctx.df_train.columns.tolist()),
            "dtypes": {col: str(dtype) for col, dtype in ctx.df_train.dtypes.items()},
            "n_columns": len(ctx.df_train.columns),
            "n_rows": len(ctx.df_train),
        }
        save_json(schema, ctx.phase3_dir / path)

    if path := output_artifacts.get("transformation_summary"):
        summary = {
            "step": "3.3",
            "train_shape_after": list(ctx.df_train.shape),
            "test_shape_after": list(ctx.df_test.shape) if ctx.df_test is not None else None,
        }
        save_json(summary, ctx.phase3_dir / path)

    write_output_artifacts(ctx, step_key=step_key, step_cfg=step_cfg,
                           df_train=ctx.df_train, df_test=ctx.df_test)
    log.info("[3.3] done train shape=%s", ctx.df_train.shape)
    return ctx

# =============================================================================
# STEP 3.5 — DATA FORMATTING
# =============================================================================
def run_step_3_5(ctx: RunContext) -> RunContext:
    """Execute CRISP-DM Phase 3.5 — Data Formatting."""
    step_key = StepsPhase.STEP_3_5.value
    step_cfg: dict[str, Any] = dget(
        ctx.config.phases.phase3_data_preparation.steps, step_key, {}
    )
    if not enabled(step_cfg, default=True):
        log.info("[3.5] step disabled – skipping")
        return ctx

    if ctx.df_train is None:
        raise RuntimeError("[3.5] df_train is None — run Phase 3.3 first")

    log.info("[3.5] start train shape=%s", ctx.df_train.shape)
    methods: dict[str, Any] = step_cfg.get("methods", {})
    output_artifacts = step_cfg.get("output_artifacts", {})

    # -----------------------------------------------------------------
    # 1. Data split — no_split_clustering (shuffle)
    # -----------------------------------------------------------------
    split_tech = (
        methods.get("data_split", {})
        .get("techniques", {})
        .get("no_split_clustering", {})
    )
    if split_tech.get("enabled", True):
        ctx.df_train, ctx.df_test, report_split = no_split_clustering(
            ctx.df_train, ctx.df_test, split_tech
        )
        if output_path := split_tech.get("output"):
            save_json(report_split, ctx.phase3_dir / output_path)
    else:
        log.info("[3.5] no_split_clustering disabled")

    # -----------------------------------------------------------------
    # 2. Dataset formatting — type_casting
    # -----------------------------------------------------------------
    cast_tech = (
        methods.get("dataset_formatting", {})
        .get("techniques", {})
        .get("type_casting", {})
    )
    if cast_tech.get("enabled", True):
        ctx.df_train, ctx.df_test, report_cast = type_casting(
            ctx.df_train, ctx.df_test, cast_tech
        )
        if output_path := cast_tech.get("output"):
            save_json(report_cast, ctx.phase3_dir / output_path)
    else:
        log.info("[3.5] type_casting disabled")

    # -----------------------------------------------------------------
    # 3. Dataset formatting — array_conversion (numpy)
    # -----------------------------------------------------------------
    array_tech = (
        methods.get("dataset_formatting", {})
        .get("techniques", {})
        .get("array_conversion", {})
    )
    if array_tech.get("enabled", True):
        X_train, X_test, report_array = array_conversion(
            ctx.df_train, ctx.df_test, array_tech
        )
        # Save arrays in artifacts for Phase 4 consumption
        ctx.artifacts["X_train"] = X_train
        ctx.artifacts["X_test"] = X_test
        ctx.artifacts["array_conversion_report"] = report_array
        if output_path := array_tech.get("output"):
            save_json(report_array, ctx.phase3_dir / output_path)
        log.info("[3.5] array conversion done: train=%s, test=%s",
                 X_train.shape, X_test.shape if X_test is not None else None)
    else:
        log.info("[3.5] array_conversion disabled")

    # -----------------------------------------------------------------
    # 4. Save final schema report (output_artifacts)
    # -----------------------------------------------------------------
    if path := output_artifacts.get("data_schema_final"):
        schema = {
            "columns": sorted(ctx.df_train.columns.tolist()),
            "dtypes": {col: str(dtype) for col, dtype in ctx.df_train.dtypes.items()},
            "n_columns": len(ctx.df_train.columns),
            "n_rows": len(ctx.df_train),
        }
        save_json(schema, ctx.phase3_dir / path)

    if path := output_artifacts.get("pipeline_integrity_check"):
        integrity = {
            "step": "3.5",
            "train_shape_after": list(ctx.df_train.shape),
            "test_shape_after": list(ctx.df_test.shape) if ctx.df_test is not None else None,
            "numpy_converted": "X_train" in ctx.artifacts,
            "techniques_applied": [
                k for k in ["no_split_clustering", "type_casting", "array_conversion"]
                if enabled(
                    methods.get("data_split" if k == "no_split_clustering" else "dataset_formatting", {})
                    .get("techniques", {})
                    .get(k, {})
                )
            ],
        }
        save_json(integrity, ctx.phase3_dir / path)

    write_output_artifacts(
        ctx,
        step_key=step_key,
        step_cfg=step_cfg,
        df_train=ctx.df_train,
        df_test=ctx.df_test,
    )
    log.info("[3.5] done train shape=%s", ctx.df_train.shape)
    return ctx
