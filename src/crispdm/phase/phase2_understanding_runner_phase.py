# src/crispdm/phase/phase2_understanding_runner_phase.py
from __future__ import annotations
"""
Why this module exists?


Objective:


Design Patterns:

"""

# ---------------------------------------------------------------------------
# SECTION 1 – Standard-library imports
# ---------------------------------------------------------------------------
from typing import Any

# ---------------------------------------------------------------------------
# SECTION 2 – Third-party imports
# ---------------------------------------------------------------------------
import pandas as pd

# ---------------------------------------------------------------------------
# SECTION 3 – Internal imports
# ---------------------------------------------------------------------------
from crispdm.common.context_facade_common import RunContext
from crispdm.common.dict_facade_common import dget, enabled, ensure_native_types
from crispdm.common.logging_adapter_common import get_logger
from crispdm.common.path_service_common import resolve_path
from crispdm.configuration.enum_registry_config import StepsPhase
from crispdm.configuration.read_strategy_repository_config import ReadStrategyContract
from crispdm.data.csv_loader_data import load_with_origin
from crispdm.data.persist_persister_data import save_json
from crispdm.data.profiling_profiler_data import (
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
    min_max_mean_std,
    multi_value_parser,
    numeric_cols,
    post_triage_detector,
    schema_comparison_report,
    target_distribution_report,
    timestamp_range_validator,
)
from crispdm.registry.generator_registry_registry import write_output_artifacts
from crispdm.reporting.artifact_persister_reporting import save_figure, save_table_png
from crispdm.reporting.plots_generator_reporting import (
    plot_categorical_distribution,
    plot_missingness_top,
    plot_target_by_category,
    plot_temporal_overview,
)

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Level logger
# ──────────────────────────────────────────────────────────────────────────────
log = get_logger(__name__)

# =============================================================================
# SECTION 5 — Constants
# =============================================================================


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
def _generate_png_for_splits(
        ctx: RunContext,
        tech_cfg: dict[str, Any],
        plot_func: callable,
        plot_title_prefix: str,
        dpi: int,
) -> None:
    """Generate one PNG per dataset split using a plot function.

    Iterates over train/test splits available in the context, calls
    the provided plot function for each non-empty DataFrame, and writes
    the resulting figure to the phase2 output directory.

    Output filenames are derived from the technique config ``output``
    key, replacing ``.json`` with ``.<split_name>.png``.

    Parameters
    ----------
    ctx : RunContext
        Pipeline run context holding DataFrames and output paths.
    tech_cfg : dict[str, Any]
        Technique configuration block; must contain an ``output`` key.
    plot_func : callable
        Plot function with signature ``(df, top_n, title) -> Figure``.
    plot_title_prefix : str
        Title prefix injected into each figure; split label is appended.
    dpi : int
        Resolution for saved PNG files.

    Returns
    -------
    None
        Figures are written to disk; nothing is returned.
    """
    # Step 1: Fetch base output name from technique config.
    output_base = tech_cfg.get("output")
    if not output_base:
        return

    # Step 2: Iterate over train/test splits available in context.
    for split_name, df, title_suffix in [
        ("train", ctx.df_train, "Train"),
        ("test", ctx.df_test, "Test"),
    ]:
        # Step 2.1: Skip split if DataFrame is missing or empty.
        if df is not None and not df.empty:

            # Step 2.2: -> CALL plot_func() to generate figure for split.
            fig = plot_func(
                df,
                top_n=30,
                title=f"{plot_title_prefix} ({title_suffix})",
            )

            # Step 2.3: Build output path replacing .json with .<split>.png.
            out_path = ctx.phase2_dir / output_base.replace(
                ".json", f".{split_name}.png"
            )

            # Step 2.4: -> CALL save_figure() to persist PNG to disk.
            save_figure(fig, out_path=out_path, dpi=dpi)


def _save_png_per_split(
        ctx: RunContext,
        tech_cfg: dict[str, Any],
        plot_func: callable,
        dpi: int,
        **plot_kwargs: Any,
) -> None:
    """Generate one PNG per dataset split using a configurable plot function.

    Unlike ``_generate_png_for_splits``, this function respects the
    ``split`` key in the technique config (``"train"``, ``"test"``, or
    ``"both"``), and forwards arbitrary keyword arguments to the plot
    function, making it reusable across all visual EDA techniques.

    Output filenames are derived from the technique config ``output``
    key, replacing ``.png`` with ``.<split_name>.png``.

    Parameters
    ----------
    ctx : RunContext
        Pipeline run context holding DataFrames and output paths.
    tech_cfg : dict[str, Any]
        Technique configuration block; must contain ``output`` and
        optionally ``split`` keys.
    plot_func : callable
        Plot function with signature ``(df, **kwargs) -> Figure``.
    dpi : int
        Resolution for saved PNG files.
    **plot_kwargs : Any
        Additional keyword arguments forwarded directly to ``plot_func``.

    Returns
    -------
    None
        Figures are written to disk; nothing is returned.

    Raises
    ------
    None
        Missing ``output`` key logs a warning and exits early.
    """
    # Step 1: Fetch base output filename from technique config.
    output_base = tech_cfg.get("output")
    if not output_base:
        log.warning("[_save_png_per_split] technique missing 'output' key")
        return

    # Step 2: Resolve which splits to plot based on config split mode.
    split_mode = tech_cfg.get("split", "both")
    splits_to_plot = []
    if split_mode in ("train", "both") and ctx.df_train is not None:
        splits_to_plot.append(("train", ctx.df_train))
    if (
            split_mode in ("test", "both")
            and ctx.df_test is not None
            and not ctx.df_test.empty
    ):
        splits_to_plot.append(("test", ctx.df_test))

    # Step 3: Iterate over resolved splits and generate one PNG each.
    for split_name, df in splits_to_plot:

        # Step 3.1: -> CALL plot_func() to generate figure for split.
        fig = plot_func(df, **plot_kwargs)

        # Step 3.2: Build output path appending .<split_name>.png suffix.
        output_path = output_base.replace(".png", f".{split_name}.png")

        # Step 3.3: -> CALL save_figure() to persist PNG to disk.
        save_figure(fig, out_path=ctx.phase2_dir / output_path, dpi=dpi)

        log.info(
            "[_save_png_per_split] %s %s done -> %s",
            plot_func.__name__,
            split_name,
            output_path,
        )

def _load_single_csv(
        path: str, csv_params: dict[str, Any], strategy: ReadStrategyContract
) -> pd.DataFrame:
    """Load a single CSV file into a DataFrame using a read strategy.

    Converts OmegaConf containers to native Python types before
    dispatching to ``load_with_origin``, which applies the sampling
    primitive (stratified, random, or head) defined in the strategy
    contract during chunked reading — no post-hoc down-sampling needed.

    Parameters
    ----------
    path : str
        Absolute path to the CSV file to load.
    csv_params : dict[str, Any]
        Raw CSV parameters from config; may contain OmegaConf
        ``ListConfig``/``DictConfig`` that pandas does not accept.
    strategy : ReadStrategyContract
        Sampling strategy contract defining how rows are selected
        during chunked reading.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame with an ``_origin`` column injected by
        ``load_with_origin``.

    Raises
    ------
    Exception
        Any exception raised by ``load_with_origin`` propagates
        to the caller unchanged.
    """
    # Step 1: -> CALL ensure_native_types() to convert OmegaConf
    # containers to native Python types that pandas accepts.
    params_native: dict[str, Any] = ensure_native_types(csv_params)

    # Step 2: -> CALL load_with_origin() to read CSV applying the
    # sampling primitive defined in the strategy contract.
    df, _ = load_with_origin(
        path,
        csv_params=params_native,
        strategy=strategy,
    )

    log.info("[_load_single_csv] loaded path=%s shape=%s", path, df.shape)
    return df

def _run_technique_on_splits(
        ctx: RunContext,
        technique_cfg: dict[str, Any],
        technique_func: callable,
        **extra_params: Any,
) -> None:
    """Execute a profiling technique on one or both dataset splits.

    Supports three split modes driven by the ``split`` key in
    ``technique_cfg``:

    - ``"train"`` / ``"test"`` / ``"both"``: runs ``technique_func``
      independently on each split, writing one JSON file per split
      with a ``.<split_name>.json`` suffix.
    - ``"compare"``: passes both DataFrames together to
      ``technique_func`` and writes a single JSON file at the base
      output path.

    Output filenames are resolved from the ``output`` key in
    ``technique_cfg``.

    Parameters
    ----------
    ctx : RunContext
        Pipeline run context holding DataFrames and output paths.
    technique_cfg : dict[str, Any]
        Technique configuration block; must contain ``output`` and
        optionally ``split`` keys.
    technique_func : callable
        Profiling function to execute. Signature varies by split mode:
        - per-split: ``(df, **extra_params) -> dict``
        - compare:   ``(df_train, df_test, **extra_params) -> dict``
    **extra_params : Any
        Additional keyword arguments forwarded directly to
        ``technique_func``.

    Returns
    -------
    None
        Results are written to disk as JSON; nothing is returned.

    Raises
    ------
    ValueError
        If ``output`` key is missing from ``technique_cfg``.
    """
    # Step 1: Read split mode from config; default to 'both'.
    split_mode: str = technique_cfg.get("split", "both")

    # Step 2: Fetch base output path; raise if missing.
    base_output: str | None = technique_cfg.get("output")
    if not base_output:
        raise ValueError(
            f"Technique {technique_func.__name__} missing 'output' in config"
        )

    # Step 3: Branch on split mode and execute technique accordingly.
    if split_mode == "compare":

        # Step 3.1: Validate both DataFrames are available for comparison.
        if ctx.df_train is not None and ctx.df_test is not None:

            # Step 3.2: -> CALL technique_func() with both splits.
            result = technique_func(ctx.df_train, ctx.df_test, **extra_params)

            # Step 3.3: -> CALL save_json() to persist result at base path.
            save_json(result, ctx.phase2_dir / base_output)

            log.info(
                "[_run_technique_on_splits] %s compare done -> %s",
                technique_func.__name__,
                base_output,
            )
    else:
        # Step 3.4: Resolve which splits to execute based on split mode.
        splits_to_run: list[tuple[str, pd.DataFrame | None]] = []
        if split_mode in ("train", "both") and ctx.df_train is not None:
            splits_to_run.append(("train", ctx.df_train))
        if (
                split_mode in ("test", "both")
                and ctx.df_test is not None
                and not ctx.df_test.empty
        ):
            splits_to_run.append(("test", ctx.df_test))

        # Step 3.5: Iterate over resolved splits and persist one JSON each.
        for split_name, df in splits_to_run:

            # Step 3.5.1: Build output path appending .<split_name>.json.
            output_path = base_output.replace(".json", f".{split_name}.json")

            # Step 3.5.2: -> CALL technique_func() on current split.
            result = technique_func(df, **extra_params)

            # Step 3.5.3: -> CALL save_json() to persist split result.
            save_json(result, ctx.phase2_dir / output_path)

            log.info(
                "[_run_technique_on_splits] %s %s done -> %s",
                technique_func.__name__,
                split_name,
                output_path,
            )
# =============================================================================
# SECTION 9 — Public functions
# =============================================================================


# ====================================================================
# SUB PHASE 2_1: DATA ACQUISITION
# ====================================================================
def run_initial_data_collection(ctx: RunContext) -> RunContext:
    """Execute CRISP-DM Phase 2.1 — Initial Data Collection.

    Loads train and test CSV files into the run context using the
    read strategy defined in configuration, persists shape metadata
    as JSON artifacts, and runs hierarchy profiling on both splits.

    The sampling primitive (stratified, random, or head) is applied
    during chunked reading via ``load_with_origin`` — no post-hoc
    down-sampling is performed.

    Parameters
    ----------
    ctx : RunContext
        Pipeline run context; ``df_train`` and ``df_test`` are
        populated by this function.

    Returns
    -------
    RunContext
        Updated context with ``df_train`` and ``df_test`` set.

    Raises
    ------
    ValueError
        If Phase 2 is disabled, ``step_2_1`` is disabled, or
        ``train_path`` is missing from configuration.
    """
    # Step 1: Validate Phase 2 is enabled in configuration.
    s2_cfg = ctx.config.phases.phase2_data_understanding
    if not s2_cfg.enabled:
        log.info("[2.1] Phase 2 disabled")
        return ctx

    # Step 2: Fetch step 2.1 config block; raise if step is disabled.
    steps_cfg: dict[str, Any] = s2_cfg.steps or {}
    step21: dict[str, Any] = steps_cfg.get(StepsPhase.STEP_2_1.value) or {}
    if not enabled(step21, default=True):
        raise ValueError("[2.1] step_2_1_data_acquisition disabled")

    # Step 3: Resolve train and test paths from dataset_input config.
    dataset_input: dict[str, Any] = s2_cfg.dataset_input or {}
    train_path_raw: str | None = (
        s2_cfg.dataset_input.train_path if s2_cfg.dataset_input else None
    )
    test_path_raw: str | None = (
        s2_cfg.dataset_input.test_path if s2_cfg.dataset_input else None
    )
    if not train_path_raw:
        raise ValueError("[2.1] train_path missing in dataset_input")

    # Step 4: -> CALL resolve_path() to convert relative paths to absolute.
    train_path: str = str(resolve_path(train_path_raw))
    test_path: str | None = (
        str(resolve_path(test_path_raw)) if test_path_raw else None
    )

    # Step 5: Fetch CSV params and build read strategy from config.
    csv_params: dict[str, Any] = dataset_input.get("csv_params") or {}
    read_strategy_dict: dict[str, Any] = s2_cfg.read_strategy or {}

    # Step 5.1: -> CALL ReadStrategyContract.from_dict() to build strategy.
    strategy = ReadStrategyContract.from_dict(read_strategy_dict)

    # Step 6: Navigate to technique config blocks under step 2.1.
    methods_cfg: dict[str, Any] = step21.get("methods") or {}
    data_acq: dict[str, Any] = methods_cfg.get("data_acquisition") or {}
    techniques: dict[str, Any] = data_acq.get("techniques") or {}

    # Step 7: Load train CSV if technique is enabled.
    load_train_cfg = techniques.get("load_csv_train") or {}
    if enabled(load_train_cfg, default=True):

        # Step 7.1: Merge technique-level params into csv_params.
        params = load_train_cfg.get("params") or {}
        csv_params.update(params)

        # Step 7.2: -> CALL _load_single_csv() to load train DataFrame.
        df_train = _load_single_csv(train_path, csv_params, strategy)

        # Step 7.3: Persist shape metadata JSON if output key is defined.
        output_file = load_train_cfg.get("output")
        if output_file:
            # Step 7.4: -> CALL save_json() to write train shape metadata.
            save_json(
                {"shape": list(df_train.shape), "columns": list(df_train.columns)},
                ctx.phase2_dir / output_file,
                )
        log.info("[2.1] loaded train shape=%s", df_train.shape)
    else:
        raise ValueError("[2.1] load_csv_train disabled")

    # Step 8: Load test CSV if technique is enabled and path exists.
    load_test_cfg = techniques.get("load_csv_test") or {}
    if enabled(load_test_cfg, default=True) and test_path:

        # Step 8.1: Merge technique-level params into csv_params.
        params = load_test_cfg.get("params") or {}
        csv_params.update(params)

        # Step 8.2: -> CALL _load_single_csv() to load test DataFrame.
        df_test = _load_single_csv(test_path, csv_params, strategy)

        # Step 8.3: -> CALL save_json() to write test shape metadata.
        output_file = load_test_cfg.get("output")
        save_json(
            {"shape": list(df_test.shape), "columns": list(df_test.columns)},
            ctx.phase2_dir / output_file,
            )
        log.info("[2.1] loaded test shape=%s", df_test.shape)
    else:
        df_test = pd.DataFrame()

    # Step 9: Inject loaded DataFrames into context before profiling.
    # Required by _run_technique_on_splits which reads from ctx directly.
    ctx.df_train = df_train
    ctx.df_test = df_test

    # Step 10: Run hierarchy profiling on both splits if enabled.
    hierarchy_cfg = techniques.get("hierarchy_profiling") or {}
    if enabled(hierarchy_cfg, default=True):
        params = hierarchy_cfg.get("params") or {}

        # Step 10.1: -> CALL _run_technique_on_splits() for hierarchy profiling.
        _run_technique_on_splits(
            ctx,
            technique_cfg=hierarchy_cfg,
            technique_func=hierarchy_profiling_report,
            **params,
        )

    # Step 11: -> CALL write_output_artifacts() to persist step artifacts.
    write_output_artifacts(
        ctx,
        step_key=StepsPhase.STEP_2_1.value,
        step_cfg=step21,
        df_train=ctx.df_train,
        df_test=ctx.df_test,
    )

    log.info("[2.1] done train=%s test=%s", df_train.shape, df_test.shape)
    return ctx

# ====================================================================
# SUB PHASE 2_2: DATA DESCRIPTION
# ====================================================================
def run_data_description(ctx: RunContext) -> RunContext:
    """Execute CRISP-DM Phase 2.2 — Data Description.

    Runs schema analysis, descriptive statistics, categorical profiling,
    and special feature detection on both train and test splits.

    Each technique is driven by its configuration block; disabled
    techniques are skipped silently. Results are persisted as JSON
    artifacts via ``_run_technique_on_splits``.

    Parameters
    ----------
    ctx : RunContext
        Pipeline run context; ``df_train`` must be populated before
        calling this function.

    Returns
    -------
    RunContext
        Updated context after all description techniques have run.

    Raises
    ------
    RuntimeError
        If ``df_train`` is None when the step is called.
    """
    # Step 1: Validate that df_train is available in context.
    if ctx.df_train is None:
        raise RuntimeError("[2.2] df_train is None")

    # Step 2: Fetch step 2.2 config block; skip if disabled.
    s2_cfg = ctx.config.phases.phase2_data_understanding
    steps_cfg: dict[str, Any] = s2_cfg.steps or {}
    step22: dict[str, Any] = steps_cfg.get(StepsPhase.STEP_2_2.value) or {}
    if not enabled(step22, default=True):
        log.info("[2.2] step disabled")
        return ctx

    log.info(
        "[2.2] start train=%s test=%s",
        ctx.df_train.shape,
        ctx.df_test.shape if ctx.df_test is not None else None,
    )

    # Step 3: Navigate to methods config block under step 2.2.
    methods_cfg: dict[str, Any] = step22.get("methods") or {}

    # Step 4: Run Schema Analysis techniques if method is enabled.
    schema_analysis = methods_cfg.get("schema_analysis") or {}
    if enabled(schema_analysis, default=True):
        techniques = schema_analysis.get("techniques") or {}

        # Step 4.1: Run column_metadata on both splits if enabled.
        tech_cfg = techniques.get("column_metadata") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}

            # Step 4.1.1: -> CALL _run_technique_on_splits() for
            # column_metadata; result converted to records dict.
            _run_technique_on_splits(
                ctx,
                tech_cfg,
                lambda df: column_metadata_report(df, **params).to_dict(
                    orient="records"
                ),
            )

        # Step 4.2: Run schema_comparison in compare mode if enabled.
        tech_cfg = techniques.get("schema_comparison") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}

            # Step 4.2.1: -> CALL _run_technique_on_splits() for
            # schema_comparison across train and test splits.
            _run_technique_on_splits(
                ctx,
                tech_cfg,
                schema_comparison_report,
                **params,
            )

    # Step 5: Run Descriptive Statistics techniques if method is enabled.
    desc_stats = methods_cfg.get("descriptive_statistics") or {}
    if enabled(desc_stats, default=True):
        techniques = desc_stats.get("techniques") or {}

        # Step 5.1: Run basic_stats on both splits if enabled.
        tech_cfg = techniques.get("basic_stats") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}

            # Step 5.1.1: -> CALL _run_technique_on_splits() for
            # min_max_mean_std; result converted to records dict.
            _run_technique_on_splits(
                ctx,
                tech_cfg,
                lambda df: min_max_mean_std(df, **params).to_dict(
                    orient="records"
                ),
            )

    # Step 6: Run Categorical Analysis techniques if method is enabled.
    cat_analysis = methods_cfg.get("categorical_analysis") or {}
    if enabled(cat_analysis, default=True):
        techniques = cat_analysis.get("techniques") or {}

        # Step 6.1: Run multi_value_parsing on both splits if enabled.
        tech_cfg = techniques.get("multi_value_parsing") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}

            # Step 6.1.1: -> CALL _run_technique_on_splits() for
            # multi_value_parser on each split.
            _run_technique_on_splits(ctx, tech_cfg, multi_value_parser, **params)

        # Step 6.2: Run cardinality_profiling on both splits if enabled.
        tech_cfg = techniques.get("cardinality_profiling") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}

            # Step 6.2.1: -> CALL _run_technique_on_splits() for
            # cardinality_profiler on each split.
            _run_technique_on_splits(ctx, tech_cfg, cardinality_profiler, **params)

        # Step 6.3: Run target_distribution on both splits if enabled.
        tech_cfg = techniques.get("target_distribution") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}

            # Step 6.3.1: -> CALL _run_technique_on_splits() for
            # target_distribution_report on each split.
            _run_technique_on_splits(
                ctx, tech_cfg, target_distribution_report, **params
            )

    # Step 7: Run Special Feature Detection techniques if method is enabled.
    special_feat = methods_cfg.get("special_feature_detection") or {}
    if enabled(special_feat, default=True):
        techniques = special_feat.get("techniques") or {}

        # Step 7.1: Run id_column_detection on both splits if enabled.
        tech_cfg = techniques.get("id_column_detection") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}

            # Step 7.1.1: -> CALL _run_technique_on_splits() for
            # detect_id_columns; result wrapped in dict for JSON.
            _run_technique_on_splits(
                ctx,
                tech_cfg,
                lambda df: {"id_columns": detect_id_columns(df, **params)},
            )

        # Step 7.2: Run entity_conditional_sparsity on both splits if enabled.
        tech_cfg = techniques.get("entity_conditional_sparsity") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}

            # Step 7.2.1: -> CALL _run_technique_on_splits() for
            # entity_conditional_sparsity on each split.
            _run_technique_on_splits(
                ctx, tech_cfg, entity_conditional_sparsity, **params
            )

    # step_results = {
    #     "schema_summary": {
    #         "column_count": len(ctx.df_train.columns),
    #         "dtypes": {col: str(dtype) for col, dtype in ctx.df_train.dtypes.items()},
    #         "has_test": ctx.df_test is not None and not ctx.df_test.empty,
    #     },
    #     "statistics_summary": {
    #         "numeric_columns": min_max_mean_std(ctx.df_train, numeric_only=True).to_dict(orient="records"),
    #     }
    # }
    # Step 8: -> CALL write_output_artifacts() to persist step artifacts.
    # write_output_artifacts(
    #     ctx,
    #     step_key=StepsPhase.STEP_2_2.value,
    #     step_cfg=step22,
    #     df_train=ctx.df_train,
    #     df_test=ctx.df_test,
    #     #step_results=step_results,
    # )

    log.info("[2.2] done")
    return ctx

# ====================================================================
# SUB PHASE 2_3: DATA QUALITY
# ====================================================================
def run_data_quality_verification(ctx: RunContext) -> RunContext:
    """Execute CRISP-DM Phase 2.3 — Data Quality Verification.

    Runs missing data profiling, structural integrity checks, leakage
    detection, temporal integrity validation, and statistical health
    analysis (drift) on both train and test splits.

    Each technique is driven by its configuration block; disabled
    techniques are skipped silently. JSON and PNG artifacts are
    persisted for each enabled technique.

    Parameters
    ----------
    ctx : RunContext
        Pipeline run context; ``df_train`` must be populated before
        calling this function.

    Returns
    -------
    RunContext
        Updated context after all quality techniques have run.

    Raises
    ------
    RuntimeError
        If ``df_train`` is None when the step is called.
    """
    # Step 1: Validate that df_train is available in context.
    if ctx.df_train is None:
        raise RuntimeError("[2.3] df_train is None")

    # Step 2: Fetch step 2.3 config block; skip if disabled.
    s2_cfg = ctx.config.phases.phase2_data_understanding
    steps_cfg: dict[str, Any] = s2_cfg.steps or {}
    step23: dict[str, Any] = steps_cfg.get(StepsPhase.STEP_2_3.value) or {}
    if not enabled(step23, default=True):
        log.info("[2.3] step disabled")
        return ctx

    log.info(
        "[2.3] start train=%s test=%s",
        ctx.df_train.shape,
        ctx.df_test.shape if ctx.df_test is not None else None,
    )

    # Step 3: Navigate to methods config block and resolve output DPI.
    methods_cfg: dict[str, Any] = step23.get("methods") or {}
    dpi: int = int(dget(ctx.config.common_base_config.output_policy, "dpi", 150))

    # Step 4: Run Missing Data Profiling techniques if method is enabled.
    missing_prof = methods_cfg.get("missing_data_profiling") or {}
    if enabled(missing_prof, default=True):
        techniques = missing_prof.get("techniques") or {}

        # Step 4.1: Run completeness_report on both splits if enabled.
        tech_cfg = techniques.get("completeness_report") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}

            # Step 4.1.1: -> CALL _run_technique_on_splits() for
            # completeness_report on each split.
            _run_technique_on_splits(ctx, tech_cfg, completeness_report, **params)

            # Step 4.1.2: -> CALL _generate_png_for_splits() to render
            # missingness bar chart PNG for each split.
            _generate_png_for_splits(
                ctx,
                tech_cfg,
                plot_missingness_top,
                plot_title_prefix="Phase 2.3 - Top Missing Columns",
                dpi=dpi,
            )
            log.info("[2.3] completeness_report done")

    # Step 5: Run Structural Integrity techniques if method is enabled.
    struct_int = methods_cfg.get("structural_integrity") or {}
    if enabled(struct_int, default=True):
        techniques = struct_int.get("techniques") or {}

        # Step 5.1: Run duplicate_detection on both splits if enabled.
        tech_cfg = techniques.get("duplicate_detection") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}

            # Step 5.1.1: -> CALL _run_technique_on_splits() for
            # duplicates_summary; result converted to records dict.
            _run_technique_on_splits(
                ctx,
                tech_cfg,
                lambda df: duplicates_summary(
                    df,
                    subset=params.get("subset"),
                    keep=params.get("keep", "first"),
                ).to_dict(orient="records"),
            )

            # Step 5.1.2: Iterate splits to generate duplicate summary PNGs.
            for split_name, df in [("train", ctx.df_train), ("test", ctx.df_test)]:
                if df is not None and not df.empty:

                    # Step 5.1.2.1: -> CALL duplicates_summary() to build
                    # DataFrame for PNG rendering.
                    dup_df = duplicates_summary(
                        df,
                        subset=params.get("subset"),
                        keep=params.get("keep", "first"),
                    )

                    # Step 5.1.2.2: Build output path replacing .json
                    # with .<split_name>.png suffix.
                    out_path = ctx.phase2_dir / tech_cfg.get("output", "").replace(
                        ".json", f".{split_name}.png"
                    )

                    # Step 5.1.2.3: -> CALL save_table_png() to persist
                    # duplicates summary as PNG artifact.
                    save_table_png(
                        dup_df,
                        out_path=out_path,
                        title="Phase 2.3 - Duplicates",
                        dpi=dpi,
                    )

            log.info("[2.3] duplicate_detection done")

        # Step 5.2: Run sentinel_detection on both splits if enabled.
        tech_cfg = techniques.get("sentinel_detection") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}

            # Step 5.2.1: -> CALL _run_technique_on_splits() for
            # detect_sentinel_values on each split.
            _run_technique_on_splits(ctx, tech_cfg, detect_sentinel_values, **params)
            log.info("[2.3] sentinel_detection done")

    # Step 6: Run Potential Leakage Detection techniques if method is enabled.
    leakage_det = methods_cfg.get("potential_leakage_detection") or {}
    if enabled(leakage_det, default=True):
        techniques = leakage_det.get("techniques") or {}

        # Step 6.1: Run crosstab_analysis on both splits if enabled.
        tech_cfg = techniques.get("crosstab_analysis") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}

            # Step 6.1.1: -> CALL _run_technique_on_splits() for
            # crosstab_leakage_analysis on each split.
            _run_technique_on_splits(
                ctx, tech_cfg, crosstab_leakage_analysis, **params
            )
            log.info("[2.3] crosstab_analysis done")

        # Step 6.2: Run post_triage_detection on both splits if enabled.
        tech_cfg = techniques.get("post_triage_detection") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}

            # Step 6.2.1: -> CALL _run_technique_on_splits() for
            # post_triage_detector; result wrapped in dict for JSON.
            _run_technique_on_splits(
                ctx,
                tech_cfg,
                lambda df: {"post_triage_columns": post_triage_detector(df, **params)},
            )
            log.info("[2.3] post_triage_detection done")

    # Step 7: Run Temporal Integrity techniques if method is enabled.
    temporal_int = methods_cfg.get("temporal_integrity") or {}
    if enabled(temporal_int, default=True):
        techniques = temporal_int.get("techniques") or {}

        # Step 7.1: Run timestamp_range on both splits if enabled.
        tech_cfg = techniques.get("timestamp_range") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}

            # Step 7.1.1: -> CALL _run_technique_on_splits() for
            # timestamp_range_validator on each split.
            _run_technique_on_splits(
                ctx, tech_cfg, timestamp_range_validator, **params
            )
            log.info("[2.3] timestamp_range done")

    # Step 8: Run Statistical Health (drift) techniques if method is enabled.
    stat_health = methods_cfg.get("statistical_health") or {}
    if enabled(stat_health, default=True):
        techniques = stat_health.get("techniques") or {}

        # Step 8.1: Run data_drift in compare mode if enabled and
        # both splits are available.
        tech_cfg = techniques.get("data_drift") or {}
        if (
                enabled(tech_cfg, default=True)
                and ctx.df_train is not None
                and ctx.df_test is not None
        ):
            from crispdm.data.profiling_profiler_data import build_drift_report
            params = tech_cfg.get("params") or {}

            # Step 8.1.1: -> CALL numeric_cols() to resolve columns
            # for drift analysis from train split.
            drift_cols = numeric_cols(ctx.df_train)

            # Step 8.1.2: Define compare function wrapping build_drift_report
            # to match the (train, test) signature of compare mode.
            def _drift_func(train, test):
                return build_drift_report(
                    train, test, drift_cols, task=ctx.task,
                    target_col=ctx.config.get("target_col"),
                    psi_drift=0.20, ks_alpha=0.05, n_bins=10,
                ).to_dict(orient="records")

            # Step 8.1.3: -> CALL _run_technique_on_splits() for
            # drift comparison across train and test splits.
            _run_technique_on_splits(ctx, tech_cfg, _drift_func)

            # Step 8.1.4: Generate drift summary PNG if output key exists.
            base_output = tech_cfg.get("output")
            if base_output:

                # Step 8.1.4.1: -> CALL build_drift_report() to rebuild
                # drift DataFrame for PNG rendering.
                drift_df = build_drift_report(
                    ctx.df_train, ctx.df_test, drift_cols, task=ctx.task,
                    target_col=ctx.config.get("target_col"),
                    psi_drift=0.20, ks_alpha=0.05, n_bins=10,
                )

                # Step 8.1.4.2: Build output path replacing .json with .png.
                drift_png_path = ctx.phase2_dir / base_output.replace(
                    ".json", ".png"
                )

                # Step 8.1.4.3: -> CALL save_table_png() to persist
                # drift summary as PNG artifact.
                save_table_png(
                    drift_df,
                    out_path=drift_png_path,
                    title="Phase 2.3 - Drift Analysis",
                    dpi=dpi,
                )

            log.info("[2.3] data_drift done")

    # Step 9: -> CALL write_output_artifacts() to persist step artifacts.
    # write_output_artifacts(
    #     ctx,
    #     step_key=StepsPhase.STEP_2_3.value,
    #     step_cfg=step23,
    #     df_train=ctx.df_train,
    #     df_test=ctx.df_test,
    # )

    log.info("[2.3] done")
    return ctx


# ====================================================================
# SUB PHASE 2_4: DATA QUEXPLORATION
# ====================================================================
def run_exploratory_analysis(ctx: RunContext) -> RunContext:
    """
.
    """
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
    # ====================================================================
    # Step 2: Feature Inventory
    feat_inv = methods_cfg.get("feature_inventory") or {}
    if enabled(feat_inv, default=True):
        techniques = feat_inv.get("techniques") or {}
        # Tech 1 ====================================================================
        # column_catalog (split: both)
        tech_cfg = techniques.get("column_catalog") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}
            _run_technique_on_splits(ctx, tech_cfg, column_catalog_by_roles, **params)
            log.info("[2.4] column_catalog done")
    # ====================================================================
    # Step 3: Visual EDA
    visual_eda = methods_cfg.get("visual_eda") or {}
    if enabled(visual_eda, default=True):
        techniques = visual_eda.get("techniques") or {}

        # Tech 1 ====================================================================
        # categorical_distributions
        tech_cfg = techniques.get("categorical_distributions") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}
            _save_png_per_split(
                ctx, tech_cfg, plot_categorical_distribution, dpi=dpi,
                columns=params.get("columns", []),
                sample_rows=params.get("sample_rows"),
                title=params.get("title", "Phase 2.4 - Categorical Distributions"),
            )
        # Tech 2 ====================================================================
        # temporal_overview
        tech_cfg = techniques.get("temporal_overview") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}
            _save_png_per_split(
                ctx, tech_cfg, plot_temporal_overview, dpi=dpi,
                time_column=params.get("time_column", "Timestamp"),
                resample_rule=params.get("resample_rule", "D"),
                sample_rows=params.get("sample_rows"),
                title=params.get("title", "Phase 2.4 - Temporal Overview"),
            )
        # Tech 3 ====================================================================
        # target_by_category
        tech_cfg = techniques.get("target_by_category") or {}
        if enabled(tech_cfg, default=True):
            params = tech_cfg.get("params") or {}
            _save_png_per_split(
                ctx, tech_cfg, plot_target_by_category, dpi=dpi,
                target_column=params.get("target_column", "IncidentGrade"),
                group_by=params.get("group_by", "Category"),
                sample_rows=params.get("sample_rows"),
                title=params.get("title", "Phase 2.4 - Target by Category"),
            )
    # ====================================================================
    # Step 7: Write output_artifacts
    # write_output_artifacts(
    #     ctx,
    #     step_key=StepsPhase.STEP_2_4.value,
    #     step_cfg=step24,
    #     df_train=ctx.df_train,
    #     df_test=ctx.df_test,
    # )
    log.info("[2.4] done")
    return ctx

