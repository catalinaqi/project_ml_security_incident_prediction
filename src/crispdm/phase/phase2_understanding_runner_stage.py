# src/crispdm/phase/phase2_understanding_runner_stage.py
from __future__ import annotations

import json
from typing import Any

import pandas as pd

from crispdm.common.dict_facade_common import ensure_native_types

from crispdm.common.context_facade_common import RunContext
from crispdm.common.dict_facade_common import dget, enabled
from crispdm.common.logging_adapter_common import get_logger
from crispdm.common.path_service_common import resolve_path
from crispdm.configuration.enum_registry_config import StepsPhase
from crispdm.data.load_utils_data import ReadStrategyContract, load_with_origin
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
    min_max_mean_std,
)
from crispdm.reporting.artifacts_service_reporting import save_figure, save_table_png
from crispdm.reporting.plots_utils_reporting import (
    plot_categorical_distribution,
    plot_missingness_top,
    plot_target_by_category,
    plot_temporal_overview,
)
# Import write_output_artifacts from the registry module in /registry/
from crispdm.registry.phase2_artifact_generator_registry import write_output_artifacts

log = get_logger(__name__)

#
# def _get_combined_df(ctx: RunContext) -> pd.DataFrame | None:
#     """
#     Return concatenated train + test DataFrame, or None if both are empty.
#
#     Used by Phase 2 runners (2.2, 2.3, 2.4) that need a single DataFrame
#     for profiling and analysis. The individual DataFrames remain accessible
#     via ``ctx.df_train`` and ``ctx.df_test`` for split-aware operations
#     (e.g. schema comparison, drift detection).
#     """
#     # Step 1: Guard — both None → return None
#     if ctx.df_train is None and ctx.df_test is None:
#         log.warning("[_get_combined_df] both df_train and df_test are None")
#         return None
#
#     # Step 2: Only one DataFrame exists → return it directly
#     if ctx.df_train is None:
#         return ctx.df_test
#     if ctx.df_test is None:
#         return ctx.df_train
#
#     # Step 3: Both exist → concatenate and log result
#     df_combined: pd.DataFrame = pd.concat(
#         [ctx.df_train, ctx.df_test], ignore_index=True
#     )
#     log.debug(
#         "[_get_combined_df] combined train=%s + test=%s → shape=%s",
#         ctx.df_train.shape,
#         ctx.df_test.shape,
#         df_combined.shape,
#     )
#     return df_combined


def _load_single_csv(
        path: str, csv_params: dict[str, Any], strategy: ReadStrategyContract
) -> pd.DataFrame:
    """Load and sample single CSV file."""
    # Step 1: Convert OmegaConf containers to native Python types.
    # OmegaConf.merge() preserves ListConfig/DictConfig which pandas does not accept.
    params_native: dict[str, Any] = ensure_native_types(csv_params)

    # Step 2: Filter out non-pandas kwargs from csv_params
    # infer_datetime is a config-level hint, not a valid pd.read_csv parameter
    VALID_READ_CSV_KWARGS = {
        "sep", "delimiter", "header", "names", "index_col",
        "usecols", "dtype", "engine", "converters", "true_values",
        "false_values", "skipinitialspace", "skiprows", "nrows",
        "na_values", "keep_default_na", "na_filter", "skip_blank_lines",
        "parse_dates", "date_parser", "date_format", "dayfirst",
        "cache_dates", "iterator", "chunksize", "compression",
        "thousands", "decimal", "lineterminator", "quotechar",
        "quoting", "doublequote", "escapechar", "comment",
        "encoding", "encoding_errors", "dialect", "on_bad_lines",
        "low_memory", "memory_map", "float_precision",
    }
    filtered_params = {k: v for k, v in params_native.items() if k in VALID_READ_CSV_KWARGS}
    skipped = set(params_native.keys()) - set(filtered_params.keys())
    if skipped:
        log.debug("[_load_single_csv] skipping non-pandas csv_params keys: %s", skipped)

    # Step 3: Read CSV via load_with_origin which dispatches to the correct
    # sampling primitive (stratified, random, or head) based on strategy.
    # The sampling is done during chunked reading — no post-hoc down-sampling needed.
    df, _ = load_with_origin(path, csv_params=filtered_params, strategy=strategy)
    log.info("[_load_single_csv] loaded path=%s shape=%s", path, df.shape)

    return df

def _run_technique_on_splits(
        ctx: RunContext,
        technique_cfg: dict[str, Any],
        technique_func: callable,
        **extra_params: Any,
) -> None:
    """
    Execute a technique on train and/or test DataFrames according to
    the 'split' configuration. Writes one or two output files with
    suffixes `.train.json` and/or `.test.json` for `both`; no suffix
    for `compare`; only `.train.json` for `train`; only `.test.json` for `test`.

    Parameters
    ----------
    ctx : RunContext
        Run context with df_train and df_test.
    technique_cfg : dict
        Technique configuration from YAML (must have "split" and "output").
    technique_func : callable
        The actual profiling function to call.
    extra_params : Any
        Additional keyword arguments for technique_func.
    """
    # Step 1: Read split mode (default 'both')
    split_mode: str = technique_cfg.get("split", "both")

    # Step 2: Get base output path (without suffix)
    base_output: str | None = technique_cfg.get("output")
    if not base_output:
        raise ValueError(f"Technique {technique_func.__name__} missing 'output' in config")

    # Step 3: Define splits to execute
    if split_mode == "compare":
        # Only one output, use base path as is
        if ctx.df_train is not None and ctx.df_test is not None:
            result = technique_func(ctx.df_train, ctx.df_test, **extra_params)
            (ctx.phase2_dir / base_output).write_text(
                json.dumps(result, indent=2, default=str),
                encoding="utf-8"
            )
            log.info(f"[{technique_func.__name__}] compare done -> {base_output}")
    else:
        # Determine which splits to run
        splits_to_run: list[tuple[str, pd.DataFrame | None]] = []
        if split_mode in ("train", "both") and ctx.df_train is not None:
            splits_to_run.append(("train", ctx.df_train))
        if split_mode in ("test", "both") and ctx.df_test is not None and not ctx.df_test.empty:
            splits_to_run.append(("test", ctx.df_test))

        for split_name, df in splits_to_run:
            # Generate output path with suffix e.g. ".train.json"
            output_path = base_output.replace(".json", f".{split_name}.json")
            result = technique_func(df, **extra_params)
            (ctx.phase2_dir / output_path).write_text(
                json.dumps(result, indent=2, default=str),
                encoding="utf-8"
            )
            log.info(f"[{technique_func.__name__}] {split_name} done -> {output_path}")

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
    # CALL s2_cfg.dataset_input — paths already injected by _inject_runtime_vars()
    dataset_input: dict[str, Any] = s2_cfg.dataset_input or {}
    train_path_raw: str | None = s2_cfg.dataset_input.train_path if s2_cfg.dataset_input else None  # type: ignore[union-attr]
    test_path_raw: str | None = s2_cfg.dataset_input.test_path if s2_cfg.dataset_input else None  # type: ignore[union-attr]
    if not train_path_raw:
        raise ValueError("[2.1] train_path missing in dataset_input")

    # Resolve relative paths to absolute using project root
    train_path: str = str(resolve_path(train_path_raw))
    test_path: str | None = str(resolve_path(test_path_raw)) if test_path_raw else None

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

        output_file = load_train_cfg.get("output")
        if output_file:
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

        output_file = load_test_cfg.get("output")
        if output_file:
            (ctx.phase2_dir / output_file).write_text(
                json.dumps({"shape": list(df_test.shape), "columns": list(df_test.columns)}, indent=2),
                encoding="utf-8",
            )
        log.info("[2.1] loaded test shape=%s", df_test.shape)
    else:
        df_test = pd.DataFrame()

    # Store DataFrames in context BEFORE hierarchy profiling (needed by _run_technique_on_splits)
    ctx.df_train = df_train
    ctx.df_test = df_test

    # Step 6: Hierarchy profiling (respeta split)
    hierarchy_cfg = techniques.get("hierarchy_profiling") or {}
    if enabled(hierarchy_cfg, default=True):
        params = hierarchy_cfg.get("params") or {}
        _run_technique_on_splits(
            ctx,
            technique_cfg=hierarchy_cfg,
            technique_func=hierarchy_profiling_report,
            **params,
        )

    # Step 7: Write output_artifacts
    write_output_artifacts(
        ctx,
        step_key=StepsPhase.STEP_2_1.value,
        step_cfg=step21,
        df_train=df_train,
        df_test=df_test,
    )

    log.info("[2.1] done train=%s test=%s", df_train.shape, df_test.shape)
    return ctx


def run_data_description(ctx: RunContext) -> RunContext:
    """Phase 2.2 - Schema analysis, categorical profiling, special features."""
    # Step 1: Validate that data exists
    if ctx.df_train is None:
        raise RuntimeError("[2.2] df_train is None")
    s2_cfg = ctx.config.phases.phase2_data_understanding
    steps_cfg: dict[str, Any] = s2_cfg.steps or {}
    step22: dict[str, Any] = steps_cfg.get(StepsPhase.STEP_2_2.value) or {}

    if not enabled(step22, default=True):
        log.info("[2.2] step disabled")
        return ctx

    log.info("[2.2] start train=%s test=%s",
             ctx.df_train.shape,
             ctx.df_test.shape if ctx.df_test is not None else None)
    methods_cfg: dict[str, Any] = step22.get("methods") or {}

    # Step 2: Schema Analysis
    schema_analysis = methods_cfg.get("schema_analysis") or {}
    if enabled(schema_analysis, default=True):
        techniques = schema_analysis.get("techniques") or {}

        # column_metadata (split: both)
        tech_cfg = techniques.get("column_metadata") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}
            _run_technique_on_splits(
                ctx,
                tech_cfg,
                lambda df: column_metadata_report(df, **params).to_dict(orient="records"),
            )

        # schema_comparison (split: compare) - built-in handling in _run_technique_on_splits
        tech_cfg = techniques.get("schema_comparison") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}
            _run_technique_on_splits(
                ctx,
                tech_cfg,
                schema_comparison_report,
                **params,
            )

    # Step 3: Descriptive Statistics
    desc_stats = methods_cfg.get("descriptive_statistics") or {}
    if enabled(desc_stats, default=True):
        techniques = desc_stats.get("techniques") or {}

        # basic_stats (split: both)
        tech_cfg = techniques.get("basic_stats") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}
            _run_technique_on_splits(
                ctx,
                tech_cfg,
                lambda df: min_max_mean_std(df, **params).to_dict(orient="records"),
            )

    # Step 4: Categorical Analysis
    cat_analysis = methods_cfg.get("categorical_analysis") or {}
    if enabled(cat_analysis, default=True):
        techniques = cat_analysis.get("techniques") or {}

        # multi_value_parsing (split: both)
        tech_cfg = techniques.get("multi_value_parsing") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}
            _run_technique_on_splits(ctx, tech_cfg, multi_value_parser, **params)

        # cardinality_profiling (split: both)
        tech_cfg = techniques.get("cardinality_profiling") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}
            _run_technique_on_splits(ctx, tech_cfg, cardinality_profiler, **params)

        # target_distribution (split: both)
        tech_cfg = techniques.get("target_distribution") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}
            _run_technique_on_splits(ctx, tech_cfg, target_distribution_report, **params)

    # Step 5: Special Feature Detection
    special_feat = methods_cfg.get("special_feature_detection") or {}
    if enabled(special_feat, default=True):
        techniques = special_feat.get("techniques") or {}

        # id_column_detection (split: both)
        tech_cfg = techniques.get("id_column_detection") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}
            _run_technique_on_splits(
                ctx,
                tech_cfg,
                lambda df: {"id_columns": detect_id_columns(df, **params)},
            )

        # entity_conditional_sparsity (split: both)
        tech_cfg = techniques.get("entity_conditional_sparsity") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}
            _run_technique_on_splits(ctx, tech_cfg, entity_conditional_sparsity, **params)

    log.info("[2.2] done")
    return ctx


def run_data_quality_verification(ctx: RunContext) -> RunContext:
    """Phase 2.3 - Completeness, leakage detection, drift analysis."""
    # Step 1: Validate that data exists
    if ctx.df_train is None:
        raise RuntimeError("[2.3] df_train is None")
    s2_cfg = ctx.config.phases.phase2_data_understanding
    steps_cfg: dict[str, Any] = s2_cfg.steps or {}
    step23: dict[str, Any] = steps_cfg.get(StepsPhase.STEP_2_3.value) or {}

    if not enabled(step23, default=True):
        log.info("[2.3] step disabled")
        return ctx

    log.info("[2.3] start train=%s test=%s",
             ctx.df_train.shape,
             ctx.df_test.shape if ctx.df_test is not None else None)
    methods_cfg: dict[str, Any] = step23.get("methods") or {}
    dpi: int = int(dget(ctx.config.common_base_config.output_policy, "dpi", 150))

    # Step 2: Missing Data Profiling
    missing_prof = methods_cfg.get("missing_data_profiling") or {}
    if enabled(missing_prof, default=True):
        techniques = missing_prof.get("techniques") or {}

        # completeness_report (split: both)
        tech_cfg = techniques.get("completeness_report") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}
            _run_technique_on_splits(ctx, tech_cfg, completeness_report, **params)

            # Generate PNGs for each split
            if tech_cfg.get("split", "both") in ("train", "both") and ctx.df_train is not None:
                fig_train = plot_missingness_top(ctx.df_train, top_n=30, title="Phase 2.3 - Top Missing Columns (Train)")
                save_figure(fig_train, out_path=ctx.phase2_dir / "2.3.missing_data_profiling.missingness_top.train.png", dpi=dpi)
            if tech_cfg.get("split", "both") in ("test", "both") and ctx.df_test is not None and not ctx.df_test.empty:
                fig_test = plot_missingness_top(ctx.df_test, top_n=30, title="Phase 2.3 - Top Missing Columns (Test)")
                save_figure(fig_test, out_path=ctx.phase2_dir / "2.3.missing_data_profiling.missingness_top.test.png", dpi=dpi)
            log.info("[2.3] completeness_report done")

    # Step 3: Structural Integrity
    struct_int = methods_cfg.get("structural_integrity") or {}
    if enabled(struct_int, default=True):
        techniques = struct_int.get("techniques") or {}

        # duplicate_detection (split: both)
        tech_cfg = techniques.get("duplicate_detection") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}
            # duplicates_summary returns a DataFrame, convert to dict for JSON
            _run_technique_on_splits(
                ctx,
                tech_cfg,
                lambda df: duplicates_summary(df, subset=params.get("subset"), keep=params.get("keep", "first")).to_dict(orient="records"),
            )

            # Generate PNGs for each split
            if tech_cfg.get("split", "both") in ("train", "both") and ctx.df_train is not None:
                dup_train = duplicates_summary(ctx.df_train, subset=params.get("subset"), keep=params.get("keep", "first"))
                save_table_png(dup_train, out_path=ctx.phase2_dir / "2.3.structural_integrity.duplicates.train.png", title="Phase 2.3 - Duplicates (Train)", dpi=dpi)
            if tech_cfg.get("split", "both") in ("test", "both") and ctx.df_test is not None and not ctx.df_test.empty:
                dup_test = duplicates_summary(ctx.df_test, subset=params.get("subset"), keep=params.get("keep", "first"))
                save_table_png(dup_test, out_path=ctx.phase2_dir / "2.3.structural_integrity.duplicates.test.png", title="Phase 2.3 - Duplicates (Test)", dpi=dpi)
            log.info("[2.3] duplicate_detection done")

        # sentinel_detection (split: both)
        tech_cfg = techniques.get("sentinel_detection") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}
            _run_technique_on_splits(ctx, tech_cfg, detect_sentinel_values, **params)
            log.info("[2.3] sentinel_detection done")

    # Step 4: Potential Leakage Detection
    leakage_det = methods_cfg.get("potential_leakage_detection") or {}
    if enabled(leakage_det, default=True):
        techniques = leakage_det.get("techniques") or {}

        # crosstab_analysis (split: both)
        tech_cfg = techniques.get("crosstab_analysis") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}
            _run_technique_on_splits(ctx, tech_cfg, crosstab_leakage_analysis, **params)
            log.info("[2.3] crosstab_analysis done")

        # post_triage_detection (split: both)
        tech_cfg = techniques.get("post_triage_detection") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}
            _run_technique_on_splits(
                ctx,
                tech_cfg,
                lambda df: {"post_triage_columns": post_triage_detector(df, **params)},
            )
            log.info("[2.3] post_triage_detection done")

    # Step 5: Temporal Integrity
    temporal_int = methods_cfg.get("temporal_integrity") or {}
    if enabled(temporal_int, default=True):
        techniques = temporal_int.get("techniques") or {}

        # timestamp_range (split: both)
        tech_cfg = techniques.get("timestamp_range") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}
            _run_technique_on_splits(ctx, tech_cfg, timestamp_range_validator, **params)
            log.info("[2.3] timestamp_range done")

    # Step 6: Statistical Health (drift)
    stat_health = methods_cfg.get("statistical_health") or {}
    if enabled(stat_health, default=True):
        techniques = stat_health.get("techniques") or {}

        # data_drift (split: compare)
        tech_cfg = techniques.get("data_drift") or {}
        if enabled(tech_cfg, default=True) and ctx.df_train is not None and ctx.df_test is not None:
            from crispdm.data.profiling_utils_data import build_drift_report
            params = tech_cfg.get("params") or {}
            drift_cols = numeric_cols(ctx.df_train)  # or combined, but train is representative

            # For compare mode, the function recibe both DataFrames
            def _drift_func(train, test):
                return build_drift_report(
                    train, test, drift_cols, task=ctx.task,
                    target_col=ctx.config.get("target_col"),
                    psi_drift=0.20, ks_alpha=0.05, n_bins=10
                ).to_dict(orient="records")

            _run_technique_on_splits(ctx, tech_cfg, _drift_func)

            # Generate single PNG for drift (compare mode)
            drift_df = build_drift_report(
                ctx.df_train, ctx.df_test, drift_cols, task=ctx.task,
                target_col=ctx.config.get("target_col"),
                psi_drift=0.20, ks_alpha=0.05, n_bins=10
            )
            save_table_png(drift_df, out_path=ctx.phase2_dir / "2.3.statistical_health.drift_summary.png", title="Phase 2.3 - Drift Analysis", dpi=dpi)
            log.info("[2.3] data_drift done")

    log.info("[2.3] done")
    return ctx

def run_exploratory_analysis(ctx: RunContext) -> RunContext:
    """Phase 2.4 - Feature catalog and visual EDA."""
    # Step 1: Validate that data exists
    if ctx.df_train is None:
        raise RuntimeError("[2.4] df_train is None")
    s2_cfg = ctx.config.phases.phase2_data_understanding
    steps_cfg: dict[str, Any] = s2_cfg.steps or {}
    step24: dict[str, Any] = steps_cfg.get(StepsPhase.STEP_2_4.value) or {}

    if not enabled(step24, default=True):
        log.info("[2.4] step disabled")
        return ctx

    log.info("[2.4] start train=%s test=%s",
             ctx.df_train.shape,
             ctx.df_test.shape if ctx.df_test is not None else None)
    methods_cfg: dict[str, Any] = step24.get("methods") or {}
    dpi: int = int(dget(ctx.config.common_base_config.output_policy, "dpi", 150))

    # Step 2: Feature Inventory
    feat_inv = methods_cfg.get("feature_inventory") or {}
    if enabled(feat_inv, default=True):
        techniques = feat_inv.get("techniques") or {}

        # column_catalog (split: both)
        tech_cfg = techniques.get("column_catalog") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}
            _run_technique_on_splits(ctx, tech_cfg, column_catalog_by_roles, **params)
            log.info("[2.4] column_catalog done")

    # Step 3: Visual EDA
    visual_eda = methods_cfg.get("visual_eda") or {}
    if enabled(visual_eda, default=True):
        techniques = visual_eda.get("techniques") or {}

        # Helper to generate PNGs per split
        def _save_png_per_split(tech_cfg, plot_func, base_output_name, **plot_kwargs):
            split_mode = tech_cfg.get("split", "both")
            splits_to_plot = []
            if split_mode in ("train", "both") and ctx.df_train is not None:
                splits_to_plot.append(("train", ctx.df_train))
            if split_mode in ("test", "both") and ctx.df_test is not None and not ctx.df_test.empty:
                splits_to_plot.append(("test", ctx.df_test))

            for split_name, df in splits_to_plot:
                fig = plot_func(df, **plot_kwargs)
                # Generate output path with suffix, e.g. .train.png
                output_path = base_output_name.replace(".png", f".{split_name}.png")
                save_figure(fig, out_path=ctx.phase2_dir / output_path, dpi=dpi)
                log.info("[2.4] %s %s done -> %s", plot_func.__name__, split_name, output_path)

        # categorical_distributions
        tech_cfg = techniques.get("categorical_distributions") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}
            _save_png_per_split(
                tech_cfg,
                plot_categorical_distribution,
                tech_cfg.get("output", "2.4.visual_eda.categorical_distributions.png"),
                columns=params.get("columns", []),
                sample_rows=params.get("sample_rows"),
                title=params.get("title", "Phase 2.4 - Categorical Distributions"),
            )

        # temporal_overview
        tech_cfg = techniques.get("temporal_overview") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}
            _save_png_per_split(
                tech_cfg,
                plot_temporal_overview,
                tech_cfg.get("output", "2.4.visual_eda.temporal_overview.png"),
                time_column=params.get("time_column", "Timestamp"),
                resample_rule=params.get("resample_rule", "D"),
                sample_rows=params.get("sample_rows"),
                title=params.get("title", "Phase 2.4 - Temporal Overview"),
            )

        # target_by_category
        tech_cfg = techniques.get("target_by_category") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}
            _save_png_per_split(
                tech_cfg,
                plot_target_by_category,
                tech_cfg.get("output", "2.4.visual_eda.target_by_category.png"),
                target_column=params.get("target_column", "IncidentGrade"),
                group_by=params.get("group_by", "Category"),
                sample_rows=params.get("sample_rows"),
                title=params.get("title", "Phase 2.4 - Target by Category"),
            )

    log.info("[2.4] done")
    return ctx