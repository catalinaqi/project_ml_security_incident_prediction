# src/crispdm/phase/phase2_understanding_runner_stage.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from crispdm.configuration.enum_registry_config import PhaseDir, StepsPhase
from crispdm.common.context_facade_common import RunContext
from crispdm.common.dict_facade_common import dget, enabled
from crispdm.common.logging_adapter_common import get_logger
from crispdm.data.load_utils_data import ReadStrategyContract
from crispdm.data.persist_utils_data import save_parquet
from crispdm.data.profiling_utils_data import (
    cardinality_profiler,
    column_catalog_by_roles,
    column_metadata_report,
    completeness_report,
    crosstab_leakage_analysis,
    detect_id_columns,
    detect_sentinel_values,
    duplicates_summary,
    entity_conditional_sparsity,
    hierarchy_profiling_report,
    multi_value_parser,
    numeric_cols,
    post_triage_detector,
    schema_comparison_report,
    target_distribution_report,
    timestamp_range_validator,
)
from crispdm.reporting.artifacts_service_reporting import save_figure, save_table_png
from crispdm.reporting.plots_utils_reporting import (
    plot_categorical_distribution,
    plot_missingness_top,
    plot_target_by_category,
    plot_temporal_overview,
)

log = get_logger(__name__)


def _load_single_csv(
        path: str, csv_params: dict[str, Any], strategy: ReadStrategyContract
) -> pd.DataFrame:
    """Load and sample single CSV file."""
    # Step 1: Read CSV with params
    df = pd.read_csv(path, **csv_params)
    log.info("[_load_single_csv] loaded path=%s shape=%s", path, df.shape)

    # Step 2: Apply sampling strategy
    if strategy.sample_rows and len(df) > strategy.sample_rows:
        if strategy.sample_method == "stratified" and strategy.stratify_column in df.columns:
            df = df.groupby(strategy.stratify_column, group_keys=False).apply(
                lambda x: x.sample(
                    n=min(len(x), strategy.sample_rows // df[strategy.stratify_column].nunique()),
                    random_state=strategy.random_state,
                )
            )
        else:
            df = df.sample(n=strategy.sample_rows, random_state=strategy.random_state)
        log.info("[_load_single_csv] sampled to shape=%s", df.shape)

    return df


def run_initial_data_collection(ctx: RunContext) -> RunContext:
    """Phase 2.1 - Load train/test CSVs separately and run hierarchy profiling."""
    # Step 1: Check enabled
    s2_cfg = ctx.config.phases.phase2_data_understanding
    if not s2_cfg.enabled:
        log.info("[2.1] Phase 2 disabled")
        return ctx

    steps_cfg: dict[str, Any] = s2_cfg.steps or {}
    step21: dict[str, Any] = steps_cfg.get(StepsPhase.STEP_2_1.value) or {}
    if not enabled(step21, default=True):
        raise ValueError("[2.1] step_2_1_data_acquisition disabled")

    # Step 2: Get paths and strategy
    dataset_input: dict[str, Any] = s2_cfg.dataset_input or {}
    train_path: str | None = ctx.config.pipeline.variables.get("train_path")
    test_path: str | None = ctx.config.pipeline.variables.get("test_path")
    if not train_path:
        raise ValueError("[2.1] train_path missing")

    csv_params: dict[str, Any] = dataset_input.get("csv_params") or {}
    read_strategy_dict: dict[str, Any] = s2_cfg.read_strategy or {}
    strategy = ReadStrategyContract.from_dict(read_strategy_dict)

    # Step 3: Navigate to techniques
    methods_cfg: dict[str, Any] = step21.get("methods") or {}
    data_acq: dict[str, Any] = methods_cfg.get("data_acquisition") or {}
    techniques: dict[str, Any] = data_acq.get("techniques") or {}

    # Step 4: Load train CSV
    load_train_cfg = techniques.get("load_csv_train") or {}
    if enabled(load_train_cfg, default=True):
        params = load_train_cfg.get("params") or {}
        csv_params.update(params)
        df_train = _load_single_csv(train_path, csv_params, strategy)
        output_file = load_train_cfg.get("output", "2.1.data_acquisition.load_csv_train.json")
        (ctx.phase2_dir / output_file).write_text(
            json.dumps({"shape": list(df_train.shape), "columns": list(df_train.columns)}, indent=2),
            encoding="utf-8",
        )
        log.info("[2.1] loaded train shape=%s", df_train.shape)
    else:
        raise ValueError("[2.1] load_csv_train disabled")

    # Step 5: Load test CSV
    load_test_cfg = techniques.get("load_csv_test") or {}
    if enabled(load_test_cfg, default=True) and test_path:
        params = load_test_cfg.get("params") or {}
        csv_params.update(params)
        df_test = _load_single_csv(test_path, csv_params, strategy)
        output_file = load_test_cfg.get("output", "2.1.data_acquisition.load_csv_test.json")
        (ctx.phase2_dir / output_file).write_text(
            json.dumps({"shape": list(df_test.shape), "columns": list(df_test.columns)}, indent=2),
            encoding="utf-8",
        )
        log.info("[2.1] loaded test shape=%s", df_test.shape)
    else:
        df_test = pd.DataFrame()

    # Step 6: Hierarchy profiling
    hierarchy_cfg = techniques.get("hierarchy_profiling") or {}
    if enabled(hierarchy_cfg, default=True):
        params = hierarchy_cfg.get("params") or {}
        hierarchy_report = hierarchy_profiling_report(df_train, **params)
        output_file = hierarchy_cfg.get("output", "2.1.data_acquisition.hierarchy_profiling.json")
        (ctx.phase2_dir / output_file).write_text(json.dumps(hierarchy_report, indent=2), encoding="utf-8")
        log.info("[2.1] hierarchy profiling done")

    # Step 7: Save parquets
    output_artifacts: dict[str, Any] = step21.get("output_artifacts") or {}
    train_parquet = output_artifacts.get("sample_train_parquet", "2.1.data_acquisition.sample_train.parquet")
    test_parquet = output_artifacts.get("sample_test_parquet", "2.1.data_acquisition.sample_test.parquet")

    save_parquet(df_train, ctx.phase2_dir / train_parquet, compression="snappy")
    if not df_test.empty:
        save_parquet(df_test, ctx.phase2_dir / test_parquet, compression="snappy")
    log.info("[2.1] saved parquets")

    # Step 8: Store in context
    ctx.df_train = df_train
    ctx.df_test = df_test
    ctx.df = pd.concat([df_train, df_test], ignore_index=True) if not df_test.empty else df_train
    ctx.source_col = None

    log.info("[2.1] done train=%s test=%s", df_train.shape, df_test.shape)
    return ctx


def run_data_description(ctx: RunContext) -> RunContext:
    """Phase 2.2 - Schema analysis, categorical profiling, special features."""
    # Step 1: Guard
    if ctx.df is None:
        raise RuntimeError("[2.2] ctx.df is None")

    df = ctx.df
    s2_cfg = ctx.config.phases.phase2_data_understanding
    steps_cfg: dict[str, Any] = s2_cfg.steps or {}
    step22: dict[str, Any] = steps_cfg.get(StepsPhase.STEP_2_2.value) or {}

    if not enabled(step22, default=True):
        log.info("[2.2] step disabled")
        return ctx

    log.info("[2.2] start shape=%s", df.shape)
    methods_cfg: dict[str, Any] = step22.get("methods") or {}

    # Step 2: Schema Analysis
    schema_analysis = methods_cfg.get("schema_analysis") or {}
    if enabled(schema_analysis, default=True):
        techniques = schema_analysis.get("techniques") or {}

        # column_metadata
        tech_cfg = techniques.get("column_metadata") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}
            metadata = column_metadata_report(df, **params)
            output_file = tech_cfg.get("output", "2.2.schema_analysis.column_metadata.json")
            (ctx.phase2_dir / output_file).write_text(
                json.dumps(metadata.to_dict(orient="records"), indent=2), encoding="utf-8"
            )
            log.info("[2.2] column_metadata done")

        # schema_comparison
        tech_cfg = techniques.get("schema_comparison") or {}
        if enabled(tech_cfg, default=True) and ctx.df_train is not None and ctx.df_test is not None:
            params = tech_cfg.get("params") or {}
            comparison = schema_comparison_report(ctx.df_train, ctx.df_test, **params)
            output_file = tech_cfg.get("output", "2.2.schema_analysis.schema_comparison.json")
            (ctx.phase2_dir / output_file).write_text(json.dumps(comparison, indent=2), encoding="utf-8")
            log.info("[2.2] schema_comparison done")

    # Step 3: Descriptive Statistics
    desc_stats = methods_cfg.get("descriptive_statistics") or {}
    if enabled(desc_stats, default=True):
        techniques = desc_stats.get("techniques") or {}

        # basic_stats
        tech_cfg = techniques.get("basic_stats") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}
            from crispdm.data.profiling_utils_data import min_max_mean_std
            stats = min_max_mean_std(df, numeric_only=params.get("numeric_only", True))
            output_file = tech_cfg.get("output", "2.2.descriptive_statistics.basic_stats.json")
            (ctx.phase2_dir / output_file).write_text(
                json.dumps(stats.to_dict(orient="records"), indent=2), encoding="utf-8"
            )
            log.info("[2.2] basic_stats done")

    # Step 4: Categorical Analysis
    cat_analysis = methods_cfg.get("categorical_analysis") or {}
    if enabled(cat_analysis, default=True):
        techniques = cat_analysis.get("techniques") or {}

        # multi_value_parsing
        tech_cfg = techniques.get("multi_value_parsing") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}
            parsed = multi_value_parser(df, **params)
            output_file = tech_cfg.get("output", "2.2.categorical_analysis.multi_value_parsing.json")
            (ctx.phase2_dir / output_file).write_text(json.dumps(parsed, indent=2), encoding="utf-8")
            log.info("[2.2] multi_value_parsing done")

        # cardinality_profiling
        tech_cfg = techniques.get("cardinality_profiling") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}
            cardinality = cardinality_profiler(df, **params)
            output_file = tech_cfg.get("output", "2.2.categorical_analysis.cardinality_profiling.json")
            (ctx.phase2_dir / output_file).write_text(json.dumps(cardinality, indent=2), encoding="utf-8")
            log.info("[2.2] cardinality_profiling done")

        # target_distribution
        tech_cfg = techniques.get("target_distribution") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}
            target_dist = target_distribution_report(df, **params)
            output_file = tech_cfg.get("output", "2.2.categorical_analysis.target_distribution.json")
            (ctx.phase2_dir / output_file).write_text(json.dumps(target_dist, indent=2, default=str), encoding="utf-8")
            log.info("[2.2] target_distribution done")

    # Step 5: Special Feature Detection
    special_feat = methods_cfg.get("special_feature_detection") or {}
    if enabled(special_feat, default=True):
        techniques = special_feat.get("techniques") or {}

        # id_column_detection
        tech_cfg = techniques.get("id_column_detection") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}
            id_cols = detect_id_columns(df, **params)
            output_file = tech_cfg.get("output", "2.2.special_feature_detection.id_columns.json")
            (ctx.phase2_dir / output_file).write_text(json.dumps({"id_columns": id_cols}, indent=2), encoding="utf-8")
            log.info("[2.2] id_column_detection done")

        # entity_conditional_sparsity
        tech_cfg = techniques.get("entity_conditional_sparsity") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}
            sparsity = entity_conditional_sparsity(df, **params)
            output_file = tech_cfg.get("output", "2.2.special_feature_detection.entity_conditional_sparsity.json")
            (ctx.phase2_dir / output_file).write_text(json.dumps(sparsity, indent=2), encoding="utf-8")
            log.info("[2.2] entity_conditional_sparsity done")

    log.info("[2.2] done")
    return ctx


def run_data_quality_verification(ctx: RunContext) -> RunContext:
    """Phase 2.3 - Completeness, leakage detection, drift analysis."""
    # Step 1: Guard
    if ctx.df is None:
        raise RuntimeError("[2.3] ctx.df is None")

    df = ctx.df
    s2_cfg = ctx.config.phases.phase2_data_understanding
    steps_cfg: dict[str, Any] = s2_cfg.steps or {}
    step23: dict[str, Any] = steps_cfg.get(StepsPhase.STEP_2_3.value) or {}

    if not enabled(step23, default=True):
        log.info("[2.3] step disabled")
        return ctx

    log.info("[2.3] start shape=%s", df.shape)
    methods_cfg: dict[str, Any] = step23.get("methods") or {}
    dpi: int = int(dget(s2_cfg.output_policy or {}, "dpi", 150))

    # Step 2: Missing Data Profiling
    missing_prof = methods_cfg.get("missing_data_profiling") or {}
    if enabled(missing_prof, default=True):
        techniques = missing_prof.get("techniques") or {}

        # completeness_report
        tech_cfg = techniques.get("completeness_report") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}
            completeness = completeness_report(df, **params)
            output_file = tech_cfg.get("output", "2.3.missing_data_profiling.completeness_report.json")
            (ctx.phase2_dir / output_file).write_text(json.dumps(completeness, indent=2), encoding="utf-8")

            # Plot missingness
            fig = plot_missingness_top(df, top_n=30, title="Phase 2.3 - Top Missing Columns")
            save_figure(fig, out_path=ctx.phase2_dir / "2.3.missing_data_profiling.missingness_top.png", dpi=dpi)
            log.info("[2.3] completeness_report done")

    # Step 3: Structural Integrity
    struct_int = methods_cfg.get("structural_integrity") or {}
    if enabled(struct_int, default=True):
        techniques = struct_int.get("techniques") or {}

        # duplicate_detection
        tech_cfg = techniques.get("duplicate_detection") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}
            dup_summary = duplicates_summary(df, subset=params.get("subset"), keep=params.get("keep", "first"))
            output_file = tech_cfg.get("output", "2.3.structural_integrity.duplicate_detection.json")
            (ctx.phase2_dir / output_file).write_text(
                json.dumps(dup_summary.to_dict(orient="records"), indent=2), encoding="utf-8"
            )
            save_table_png(dup_summary, out_path=ctx.phase2_dir / "2.3.structural_integrity.duplicates.png", title="Phase 2.3 - Duplicates", dpi=dpi)
            log.info("[2.3] duplicate_detection done")

        # sentinel_detection
        tech_cfg = techniques.get("sentinel_detection") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}
            sentinels = detect_sentinel_values(df, **params)
            output_file = tech_cfg.get("output", "2.3.structural_integrity.sentinel_detection.json")
            (ctx.phase2_dir / output_file).write_text(json.dumps(sentinels, indent=2), encoding="utf-8")
            log.info("[2.3] sentinel_detection done")

    # Step 4: Potential Leakage Detection
    leakage_det = methods_cfg.get("potential_leakage_detection") or {}
    if enabled(leakage_det, default=True):
        techniques = leakage_det.get("techniques") or {}

        # crosstab_analysis
        tech_cfg = techniques.get("crosstab_analysis") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}
            crosstab_result = crosstab_leakage_analysis(df, **params)
            output_file = tech_cfg.get("output", "2.3.potential_leakage_detection.crosstab_analysis.json")
            (ctx.phase2_dir / output_file).write_text(json.dumps(crosstab_result, indent=2), encoding="utf-8")
            log.info("[2.3] crosstab_analysis done")

        # post_triage_detection
        tech_cfg = techniques.get("post_triage_detection") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}
            post_triage = post_triage_detector(df, **params)
            output_file = tech_cfg.get("output", "2.3.potential_leakage_detection.post_triage_detection.json")
            (ctx.phase2_dir / output_file).write_text(json.dumps({"post_triage_columns": post_triage}, indent=2), encoding="utf-8")
            log.info("[2.3] post_triage_detection done")

    # Step 5: Temporal Integrity
    temporal_int = methods_cfg.get("temporal_integrity") or {}
    if enabled(temporal_int, default=True):
        techniques = temporal_int.get("techniques") or {}

        # timestamp_range
        tech_cfg = techniques.get("timestamp_range") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}
            ts_validation = timestamp_range_validator(df, **params)
            output_file = tech_cfg.get("output", "2.3.temporal_integrity.timestamp_range.json")
            (ctx.phase2_dir / output_file).write_text(json.dumps(ts_validation, indent=2), encoding="utf-8")
            log.info("[2.3] timestamp_range done")

    # Step 6: Statistical Health (drift)
    stat_health = methods_cfg.get("statistical_health") or {}
    if enabled(stat_health, default=True):
        techniques = stat_health.get("techniques") or {}

        # data_drift
        tech_cfg = techniques.get("data_drift") or {}
        if enabled(tech_cfg, default=True) and ctx.df_train is not None and ctx.df_test is not None:
            from crispdm.data.profiling_utils_data import build_drift_report
            params = tech_cfg.get("params") or {}
            drift_cols = numeric_cols(df)
            drift_df = build_drift_report(
                ctx.df_train, ctx.df_test, drift_cols, task=ctx.task,
                target_col=ctx.config.pipeline.variables.get("target_col"),
                psi_drift=0.20, ks_alpha=0.05, n_bins=10
            )
            output_file = tech_cfg.get("output", "2.3.statistical_health.data_drift.json")
            (ctx.phase2_dir / output_file).write_text(
                json.dumps(drift_df.to_dict(orient="records"), indent=2), encoding="utf-8"
            )
            save_table_png(drift_df, out_path=ctx.phase2_dir / "2.3.statistical_health.drift_summary.png", title="Phase 2.3 - Drift Analysis", dpi=dpi)
            log.info("[2.3] data_drift done")

    log.info("[2.3] done")
    return ctx


def run_exploratory_analysis(ctx: RunContext) -> RunContext:
    """Phase 2.4 - Feature catalog and visual EDA."""
    # Step 1: Guard
    if ctx.df is None:
        raise RuntimeError("[2.4] ctx.df is None")

    df = ctx.df
    s2_cfg = ctx.config.phases.phase2_data_understanding
    steps_cfg: dict[str, Any] = s2_cfg.steps or {}
    step24: dict[str, Any] = steps_cfg.get(StepsPhase.STEP_2_4.value) or {}

    if not enabled(step24, default=True):
        log.info("[2.4] step disabled")
        return ctx

    log.info("[2.4] start shape=%s", df.shape)
    methods_cfg: dict[str, Any] = step24.get("methods") or {}
    dpi: int = int(dget(s2_cfg.output_policy or {}, "dpi", 150))

    # Step 2: Feature Inventory
    feat_inv = methods_cfg.get("feature_inventory") or {}
    if enabled(feat_inv, default=True):
        techniques = feat_inv.get("techniques") or {}

        # column_catalog
        tech_cfg = techniques.get("column_catalog") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}
            catalog = column_catalog_by_roles(df, **params)
            output_file = tech_cfg.get("output", "2.4.feature_inventory.column_catalog.json")
            (ctx.phase2_dir / output_file).write_text(json.dumps(catalog, indent=2), encoding="utf-8")
            log.info("[2.4] column_catalog done")

    # Step 3: Visual EDA
    visual_eda = methods_cfg.get("visual_eda") or {}
    if enabled(visual_eda, default=True):
        techniques = visual_eda.get("techniques") or {}

        # categorical_distributions
        tech_cfg = techniques.get("categorical_distributions") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}
            fig = plot_categorical_distribution(
                df, columns=params.get("columns", []), title="Phase 2.4 - Categorical Distributions",
                sample_rows=params.get("sample_rows")
            )
            output_file = tech_cfg.get("output", "2.4.visual_eda.categorical_distributions.png")
            save_figure(fig, out_path=ctx.phase2_dir / output_file, dpi=dpi)
            log.info("[2.4] categorical_distributions done")

        # temporal_overview
        tech_cfg = techniques.get("temporal_overview") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}
            fig = plot_temporal_overview(
                df, time_column=params.get("time_column", "Timestamp"),
                title="Phase 2.4 - Temporal Overview",
                resample_rule=params.get("resample_rule", "D"),
                sample_rows=params.get("sample_rows")
            )
            output_file = tech_cfg.get("output", "2.4.visual_eda.temporal_overview.png")
            save_figure(fig, out_path=ctx.phase2_dir / output_file, dpi=dpi)
            log.info("[2.4] temporal_overview done")

        # target_by_category
        tech_cfg = techniques.get("target_by_category") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}
            fig = plot_target_by_category(
                df, target_column=params.get("target_column", "IncidentGrade"),
                group_by=params.get("group_by", "Category"),
                title="Phase 2.4 - Target by Category",
                sample_rows=params.get("sample_rows")
            )
            output_file = tech_cfg.get("output", "2.4.visual_eda.target_by_category.png")
            save_figure(fig, out_path=ctx.phase2_dir / output_file, dpi=dpi)
            log.info("[2.4] target_by_category done")

    log.info("[2.4] done")
    return ctx