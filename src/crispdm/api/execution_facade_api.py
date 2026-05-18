from __future__ import annotations

from typing import Any, Optional

from crispdm.configuration.build_factory_config import build_config, BuiltConfig
from crispdm.configuration.enum_registry_config import ProblemType, normalize_problem_type
from crispdm.configuration.yml_repository_config import YmlRepository
from crispdm.common.context_facade_common import RunContext, create_run_context
from crispdm.common.logging_adapter_common import config_run_logging, get_logger
from crispdm.data.download_extractor_data import download_microsoft_dataset
from crispdm.pipeline.clustering_runner_pipeline import (
    ClusteringRunContext,
    create_clustering_context,
    run_clustering_pipeline_phase2_1,
    run_clustering_pipeline_phase3_1,   # <--- NEW
)
from crispdm.pipeline.clustering_runner_pipeline import (
    ClusteringRunContext,
    create_clustering_context,
    run_clustering_pipeline_phase2_1,
)
from crispdm.phase.phase2_understanding_runner_phase import (
    run_data_description,
    run_data_quality_verification,
    run_exploratory_analysis,
)

log = get_logger(__name__)


def _detect_task_from_config(
        pipeline_name: str,
) -> ProblemType:
    """Detect the ProblemType from a loaded pipeline config's metadata."""
    pipeline_cfg = YmlRepository.load_pipeline_config(pipeline_name)

    task_value = pipeline_cfg.metadata.get("pipeline_key", {}).get("task")
    if not task_value:
        raise KeyError(f"Task key not found in pipeline config for '{pipeline_name}'")

    task = normalize_problem_type(str(task_value).strip().lower())
    log.debug("Detected task: %s", task.value)
    return task


def init_run_phase2(
        pipeline_name: str,
        dataset_key: str,
        notebook_vars: Optional[dict[str, Any]] = None,
) -> RunContext:
    """Initialize a phase2 run context for a given pipeline and dataset.

    Parameters
    ----------
    pipeline_name : str
        Pipeline name (e.g. ``"clustering"``, ``"classification"``).
        Corresponds to ``<pipeline_name>_pipeline_config.yml``.
    dataset_key : str
        Dataset key in ``dataset_config.yml``.
    notebook_vars : dict[str, Any] or None, optional
        Runtime variables from the notebook context.

    Returns
    -------
    RunContext
        Initialized run context ready for phase 2 execution.
    """
    notebook_vars = notebook_vars or {}

    log.info(
        "Initializing phase2 run: pipeline=%s dataset_key=%s",
        pipeline_name,
        dataset_key,
    )

    # Step 1: Build pipeline config using ConfigBuilder via build_config()
    #         This internally:
    #           - Loads pipeline + dataset configs via YmlRepository
    #           - Injects runtime variables (train/test paths, notebook vars)
    #           - Applies dataset defaults to phase2
    #           - Validates config structure with Pydantic DTO (PipelineConfig)
    download_microsoft_dataset()

    built: BuiltConfig = build_config(
        pipeline_name=pipeline_name,
        dataset_key=dataset_key,
        notebook_vars=notebook_vars,
    )

    # Step 2: Create run context using the factory helper
    ctx = create_run_context(
        config=built.config,
        dataset_key=dataset_key,
    )

    # Step 3: Configure logging for this run
    task = ctx.task  # str from config metadata
    run_name = f"run_{task}_{dataset_key}_{ctx.run_id}"
    log_file = config_run_logging(
        output_root=built.config.common_base_config.runtime.output_root,
        run_name=run_name,
        log_level=built.config.common_base_config.runtime.log_level,
    )

    log.info("Logging to file: %s", log_file)
    log.info("Phase2 context initialized: run_id=%s run_dir=%s", ctx.run_id, ctx.run_dir)
    return ctx


def run_phase2_1(ctx: RunContext) -> RunContext:
    """Run Phase 2.1 - Data Acquisition.

    Dispatches to the pipeline-specific Phase 2.1 runner based on
    the task type stored in the config metadata.
    """
    log.info("Running phase2.1: run_id=%s", ctx.run_id)

    log.info(
        "[run_phase2_1] start task=%s run_id=%s",
        ctx.task,
        ctx.run_id,
    )

    # Dispatch to pipeline-specific runner
    ctx = _dispatch_pipeline_phase2_1(ctx)

    log.info(
        "[run_phase2_1] done df_train_shape=%s",
        ctx.df_train.shape if ctx.df_train is not None else None,
    )

    log.info("Phase2.1 complete")
    return ctx


def run_phase2_2(ctx: RunContext) -> RunContext:
    """Run Phase 2.2 - Data Description."""
    log.info("Running phase2.2: run_id=%s", ctx.run_id)
    log.info(
        "[run_phase2_2] start task=%s run_id=%s",
        ctx.task,
        ctx.run_id,
    )
    ctx = _dispatch_pipeline_phase2_2(ctx)
    log.info("Phase2.2 complete")
    return ctx


def run_phase2_3(ctx: RunContext) -> RunContext:
    """Run Phase 2.3 - Data Quality Assessment."""
    log.info("Running phase2.3: run_id=%s", ctx.run_id)
    log.info(
        "[run_phase2_3] start task=%s run_id=%s",
        ctx.task,
        ctx.run_id,
    )
    ctx = _dispatch_pipeline_phase2_3(ctx)

    log.info("Phase2.3 complete")
    return ctx


def run_phase2_4(ctx: RunContext) -> RunContext:
    """Run Phase 2.4 - Exploratory Analysis."""
    log.info("Running phase2.4: run_id=%s", ctx.run_id)

    log.info(
        "[run_phase2_4] start task=%s run_id=%s",
        ctx.task,
        ctx.run_id,
    )
    ctx = _dispatch_pipeline_phase2_4(ctx)

    log.info("Phase2.4 complete")
    return ctx


def _dispatch_pipeline_phase2_1(ctx: RunContext) -> RunContext:
    """Dispatch Phase 2.1 to the correct pipeline runner by ``ctx.task``.

    ``ctx.task`` is a ``str`` from config metadata (e.g. ``"clustering"``).
    Each known task value maps to its pipeline-specific Phase 2.1 runner.

    Parameters
    ----------
    ctx : RunContext
        Run context with task set in config metadata.

    Returns
    -------
    RunContext
        Same ``ctx`` enriched by Phase 2.1.

    Raises
    ------
    NotImplementedError
        If a runner does not exist yet for the detected task.
    """
    # Route by task string — compare to ProblemType enum values as strings.
    if ctx.task == ProblemType.CLUSTERING.value:
        # Wrap generic RunContext into ClusteringRunContext for the runner
        # (copies all fields from the generic context)
        clustering_ctx = ClusteringRunContext(
            config=ctx.config,
            run_dir=ctx.run_dir,
            run_id=ctx.run_id,
            dataset_key=ctx.dataset_key,
        )
        return run_clustering_pipeline_phase2_1(clustering_ctx)

    raise NotImplementedError(
        f"[_dispatch_pipeline_phase2_1] task={ctx.task!r} pipeline not yet implemented. "
        f"Valid tasks: {[m.value for m in ProblemType]}"
    )

def _dispatch_pipeline_phase2_2(ctx: RunContext) -> RunContext:
    """Dispatch Phase 2.2 to the correct pipeline runner by ``ctx.task``.

    REUSES the *same* ``ctx`` (already populated by Phase 2.1) — does NOT
    create a new empty context.  The downstream runners (e.g. ``run_data_description``)
    accept a plain ``RunContext``, so no ClusteringRunContext wrapper is needed.

    Parameters
    ----------
    ctx : RunContext
        Run context with task set in config metadata.  Must already have
        ``df_train`` and ``df_test`` set by Phase 2.1.

    Returns
    -------
    RunContext
        Same ``ctx`` enriched by Phase 2.2.

    Raises
    ------
    RuntimeError
        If ``ctx.df_train`` is ``None`` (Phase 2.1 not run first).
    NotImplementedError
        If a runner does not exist yet for the detected task.
    """
    if ctx.task == ProblemType.CLUSTERING.value:
        if ctx.df_train is None:
            raise RuntimeError("[2.2] no data loaded - run Phase 2.1 first")
        return run_data_description(ctx)

    raise NotImplementedError(
        f"[_dispatch_pipeline_phase2_2] task={ctx.task!r} pipeline not yet "
        f"implemented. "
        f"Valid tasks: {[m.value for m in ProblemType]}"
    )

def _dispatch_pipeline_phase2_3(ctx: RunContext) -> RunContext:
    """Dispatch Phase 2.3 to the correct pipeline runner by ``ctx.task``.

    REUSES the *same* ``ctx`` (already populated by Phase 2.1) — does NOT
    create a new empty context.

    Parameters
    ----------
    ctx : RunContext
        Run context with task set in config metadata.  Must already have
        ``df_train`` and ``df_test`` set by Phase 2.1.

    Returns
    -------
    RunContext
        Same ``ctx`` enriched by Phase 2.3.

    Raises
    ------
    RuntimeError
        If ``ctx.df_train`` is ``None`` (Phase 2.1 not run first).
    NotImplementedError
        If a runner does not exist yet for the detected task.
    """
    if ctx.task == ProblemType.CLUSTERING.value:
        if ctx.df_train is None:
            raise RuntimeError("[2.3] no data loaded - run Phase 2.1 first")
        return run_data_quality_verification(ctx)

    raise NotImplementedError(
        f"[_dispatch_pipeline_phase2_3] task={ctx.task!r} pipeline not yet "
        f"implemented. "
        f"Valid tasks: {[m.value for m in ProblemType]}"
    )

def _dispatch_pipeline_phase2_4(ctx: RunContext) -> RunContext:
    """Dispatch Phase 2.4 to the correct pipeline runner by ``ctx.task``.

    REUSES the *same* ``ctx`` (already populated by Phase 2.1) — does NOT
    create a new empty context.

    Parameters
    ----------
    ctx : RunContext
        Run context with task set in config metadata.  Must already have
        ``df_train`` and ``df_test`` set by Phase 2.1.

    Returns
    -------
    RunContext
        Same ``ctx`` enriched by Phase 2.4.

    Raises
    ------
    RuntimeError
        If ``ctx.df_train`` is ``None`` (Phase 2.1 not run first).
    NotImplementedError
        If a runner does not exist yet for the detected task.
    """
    if ctx.task == ProblemType.CLUSTERING.value:
        if ctx.df_train is None:
            raise RuntimeError("[2.4] no data loaded - run Phase 2.1 first")
        return run_exploratory_analysis(ctx)

    raise NotImplementedError(
        f"[_dispatch_pipeline_phase2_4] task={ctx.task!r} pipeline not yet "
        f"implemented. "
        f"Valid tasks: {[m.value for m in ProblemType]}"
    )

# =============================================================================
# PHASE 3 DISPATCHERS
# =============================================================================

def run_phase3_1(ctx: RunContext) -> RunContext:
    """Run Phase 3.1 - Data Selection (sentinel removal).

    Reads sample_train.parquet from Phase 2 and applies sentinel removal
    (bigint_cleanup).  Dispatches to the pipeline-specific runner.

    Parameters
    ----------
    ctx : RunContext
        Run context with task set in config metadata. Must have
        ``run_dir`` pointing to the completed Phase 2 output.

    Returns
    -------
    RunContext
        Same ``ctx`` enriched by Phase 3.1.

    Raises
    ------
    NotImplementedError
        If a runner does not exist yet for the detected task.
    """
    log.info("[run_phase3_1] start task=%s run_id=%s", ctx.task, ctx.run_id)
    ctx = _dispatch_pipeline_phase3_1(ctx)
    log.info("[run_phase3_1] done df_train_shape=%s",
             ctx.df_train.shape if ctx.df_train is not None else None)
    return ctx


def _dispatch_pipeline_phase3_1(ctx: RunContext) -> RunContext:
    """Dispatch Phase 3.1 to the correct pipeline runner by ``ctx.task``.

    For clustering, wraps the generic ``RunContext`` into a
    ``ClusteringRunContext`` and calls ``run_clustering_pipeline_phase3_1``.

    Parameters
    ----------
    ctx : RunContext
        Run context with task set in config metadata.

    Returns
    -------
    RunContext
        Same ``ctx`` enriched by Phase 3.1.

    Raises
    ------
    NotImplementedError
        If a runner does not exist yet for the detected task.
    """
    if ctx.task == ProblemType.CLUSTERING.value:
        # Wrap generic RunContext into ClusteringRunContext for consistency
        clustering_ctx = ClusteringRunContext(
            config=ctx.config,
            run_dir=ctx.run_dir,
            run_id=ctx.run_id,
            dataset_key=ctx.dataset_key,
            df_train=ctx.df_train,
            df_test=ctx.df_test,
            artifacts=ctx.artifacts,
            phase_results=ctx.phase_results,
            errors=ctx.errors,
        )
        return run_clustering_pipeline_phase3_1(clustering_ctx)

    raise NotImplementedError(
        f"[_dispatch_pipeline_phase3_1] task={ctx.task!r} pipeline not yet implemented. "
        f"Valid tasks: {[m.value for m in ProblemType]}"
    )