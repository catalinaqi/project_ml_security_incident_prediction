# src/crispdm/pipeline/clustering_runner_pipeline.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from crispdm.configuration.build_factory_config import build_config, BuiltConfig
from crispdm.configuration.enum_registry_config import PhaseDir, StepsPhase
from crispdm.common.context_facade_common import RunContext, create_run_context
from crispdm.common.logging_adapter_common import get_logger
from crispdm.phase.phase2_understanding_runner_stage import (
    run_data_description,
    run_data_quality_verification,
    run_exploratory_analysis,
    run_initial_data_collection,
)

log = get_logger(__name__)


@dataclass
class ClusteringRunContext(RunContext):
    """Mutable state for clustering pipeline run."""
    cluster_labels: Optional[pd.Series] = field(default=None, repr=False)


def create_clustering_context(
        *,
        pipeline_name: str,
        dataset_key: str,
        notebook_vars: Optional[dict[str, Any]] = None,
) -> ClusteringRunContext:
    """Build ClusteringRunContext ready for Phase 2."""
    log.info("[create_clustering_context] start dataset_key=%s", dataset_key)

    # Step 1: Build pipeline config via build_config()
    built: BuiltConfig = build_config(
        pipeline_name=pipeline_name,
        dataset_key=dataset_key,
        notebook_vars=notebook_vars,
    )

    # Step 2: Create run context via factory helper
    ctx_generic = create_run_context(
        config=built.config,
        dataset_key=dataset_key,
    )

    # Step 3: Create clustering-specific context, copying all fields from generic
    ctx = ClusteringRunContext(
        config=ctx_generic.config,
        run_dir=ctx_generic.run_dir,
        run_id=ctx_generic.run_id,
        dataset_key=ctx_generic.dataset_key,
        df_train=ctx_generic.df_train,
        df_test=ctx_generic.df_test,
        artifacts=ctx_generic.artifacts,
        phase_results=ctx_generic.phase_results,
        errors=ctx_generic.errors,
    )

    log.info("[create_clustering_context] done run_id=%s", ctx.run_id)
    return ctx


def run_clustering_pipeline(ctx: ClusteringRunContext) -> ClusteringRunContext:
    """Execute full CRISP-DM clustering pipeline."""
    log.info("[run_clustering_pipeline] START run_id=%s task=%s", ctx.run_id, ctx.task)

    # Phase 2 - Data Understanding
    log.info("[run_clustering_pipeline] >>> PHASE 2")
    ctx = run_clustering_pipeline_phase2_1(ctx)
    ctx = run_clustering_pipeline_phase2_2(ctx)
    ctx = run_clustering_pipeline_phase2_3(ctx)
    ctx = run_clustering_pipeline_phase2_4(ctx)

    # Phase 3 - Data Preparation
    log.info("[run_clustering_pipeline] >>> PHASE 3")
    ctx = run_clustering_pipeline_phase3_1(ctx)
    ctx = run_clustering_pipeline_phase3_2(ctx)
    ctx = run_clustering_pipeline_phase3_3(ctx)
    ctx = run_clustering_pipeline_phase3_5(ctx)

    # Phase 4 - Modeling
    log.info("[run_clustering_pipeline] >>> PHASE 4")
    ctx = run_clustering_pipeline_phase4_1(ctx)
    ctx = run_clustering_pipeline_phase4_2(ctx)
    ctx = run_clustering_pipeline_phase4_3(ctx)
    ctx = run_clustering_pipeline_phase4_4(ctx)

    # Phase 5 - Evaluation
    log.info("[run_clustering_pipeline] >>> PHASE 5")
    ctx = run_clustering_pipeline_phase5_1(ctx)
    ctx = run_clustering_pipeline_phase5_2(ctx)
    ctx = run_clustering_pipeline_phase5_3(ctx)
    ctx = run_clustering_pipeline_phase5_4(ctx)

    log.info("[run_clustering_pipeline] END run_id=%s artifacts=%d", ctx.run_id, len(ctx.artifacts))
    return ctx


# =============================================================================
# PHASE 2 ORCHESTRATORS
# =============================================================================


def run_clustering_pipeline_phase2_1(ctx: ClusteringRunContext) -> ClusteringRunContext:
    """Phase 2.1 - Load train/test CSVs separately."""
    if ctx.df_train is not None:
        log.warning("[2.1] df_train already set shape=%s - skipping", ctx.df_train.shape)
        return ctx

    log.info("[2.1] start run_id=%s", ctx.run_id)
    ctx = run_initial_data_collection(ctx)
    log.info(
        "[2.1] done train=%s test=%s",
        ctx.df_train.shape if ctx.df_train is not None else None,
        ctx.df_test.shape if ctx.df_test is not None else None,
    )
    return ctx


def run_clustering_pipeline_phase2_2(ctx: ClusteringRunContext) -> ClusteringRunContext:
    """Phase 2.2 - Data profiling and description."""
    if ctx.df_train is None:
        raise RuntimeError("[2.2] no data loaded - run Phase 2.1 first")

    log.info("[2.2] start run_id=%s", ctx.run_id)
    ctx = run_data_description(ctx)
    log.info("[2.2] done")
    return ctx


def run_clustering_pipeline_phase2_3(ctx: ClusteringRunContext) -> ClusteringRunContext:
    """Phase 2.3 - Quality verification and drift detection."""
    if ctx.df_train is None:
        raise RuntimeError("[2.3] no data loaded - run Phase 2.1 first")

    log.info("[2.3] start run_id=%s", ctx.run_id)
    ctx = run_data_quality_verification(ctx)
    drift_detected = ctx.phase_results.get(StepsPhase.STEP_2_3.value, {}).get("drift_analyzed", False)
    log.info("[2.3] done drift_detected=%s", drift_detected)
    return ctx


def run_clustering_pipeline_phase2_4(ctx: ClusteringRunContext) -> ClusteringRunContext:
    """Phase 2.4 - Exploratory Data Analysis."""
    if ctx.df_train is None:
        raise RuntimeError("[2.4] no data loaded - run Phase 2.1 first")

    log.info("[2.4] start run_id=%s", ctx.run_id)
    ctx = run_exploratory_analysis(ctx)
    log.info("[2.4] done")
    return ctx


# =============================================================================
# PHASE 3 ORCHESTRATORS (STUBS)
# =============================================================================


def run_clustering_pipeline_phase3_1(ctx: ClusteringRunContext) -> ClusteringRunContext:
    """Phase 3.1 - Data Selection (STUB)."""
    log.info("[3.1] START run_id=%s", ctx.run_id)
    # ctx = run_data_selection(ctx)  # TODO: implement
    log.info("[3.1] DONE (stub)")
    return ctx


def run_clustering_pipeline_phase3_2(ctx: ClusteringRunContext) -> ClusteringRunContext:
    """Phase 3.2 - Data Cleaning (STUB)."""
    log.info("[3.2] START run_id=%s", ctx.run_id)
    # ctx = run_data_cleaning(ctx)  # TODO: implement
    log.info("[3.2] DONE (stub)")
    return ctx


def run_clustering_pipeline_phase3_3(ctx: ClusteringRunContext) -> ClusteringRunContext:
    """Phase 3.3 - Data Transformation (STUB)."""
    log.info("[3.3] START run_id=%s", ctx.run_id)
    # ctx = run_data_transformation(ctx)  # TODO: implement
    log.info("[3.3] DONE (stub)")
    return ctx


def run_clustering_pipeline_phase3_5(ctx: ClusteringRunContext) -> ClusteringRunContext:
    """Phase 3.5 - Data Formatting (STUB)."""
    log.info("[3.5] START run_id=%s", ctx.run_id)
    # ctx = run_data_formatting(ctx)  # TODO: implement
    log.info("[3.5] DONE (stub)")
    return ctx


# =============================================================================
# PHASE 4 ORCHESTRATORS (STUBS)
# =============================================================================


def run_clustering_pipeline_phase4_1(ctx: ClusteringRunContext) -> ClusteringRunContext:
    """Phase 4.1 - Algorithm Selection (STUB)."""
    log.info("[4.1] START run_id=%s", ctx.run_id)
    # ctx = run_algorithm_selection(ctx)  # TODO: implement
    log.info("[4.1] DONE (stub)")
    return ctx


def run_clustering_pipeline_phase4_2(ctx: ClusteringRunContext) -> ClusteringRunContext:
    """Phase 4.2 - Model Training (STUB)."""
    log.info("[4.2] START run_id=%s", ctx.run_id)
    # ctx = run_model_training(ctx)  # TODO: implement
    log.info("[4.2] DONE (stub)")
    return ctx


def run_clustering_pipeline_phase4_3(ctx: ClusteringRunContext) -> ClusteringRunContext:
    """Phase 4.3 - Test Design (STUB)."""
    log.info("[4.3] START run_id=%s", ctx.run_id)
    # ctx = run_test_design(ctx)  # TODO: implement
    log.info("[4.3] DONE (stub)")
    return ctx


def run_clustering_pipeline_phase4_4(ctx: ClusteringRunContext) -> ClusteringRunContext:
    """Phase 4.4 - Model Evaluation (STUB)."""
    log.info("[4.4] START run_id=%s", ctx.run_id)
    # ctx = run_model_evaluation(ctx)  # TODO: implement
    log.info("[4.4] DONE (stub)")
    return ctx


# =============================================================================
# PHASE 5 ORCHESTRATORS (STUBS)
# =============================================================================


def run_clustering_pipeline_phase5_1(ctx: ClusteringRunContext) -> ClusteringRunContext:
    """Phase 5.1 - Interpretation (STUB)."""
    log.info("[5.1] START run_id=%s", ctx.run_id)
    # ctx = run_interpretation(ctx)  # TODO: implement
    log.info("[5.1] DONE (stub)")
    return ctx


def run_clustering_pipeline_phase5_2(ctx: ClusteringRunContext) -> ClusteringRunContext:
    """Phase 5.2 - Business Evaluation (STUB)."""
    log.info("[5.2] START run_id=%s", ctx.run_id)
    # ctx = run_business_evaluation(ctx)  # TODO: implement
    log.info("[5.2] DONE (stub)")
    return ctx


def run_clustering_pipeline_phase5_3(ctx: ClusteringRunContext) -> ClusteringRunContext:
    """Phase 5.3 - Process Audit (STUB)."""
    log.info("[5.3] START run_id=%s", ctx.run_id)
    # ctx = run_process_audit(ctx)  # TODO: implement
    log.info("[5.3] DONE (stub)")
    return ctx


def run_clustering_pipeline_phase5_4(ctx: ClusteringRunContext) -> ClusteringRunContext:
    """Phase 5.4 - Decision Making (STUB)."""
    log.info("[5.4] START run_id=%s", ctx.run_id)
    # ctx = run_decision_making(ctx)  # TODO: implement
    log.info("[5.4] DONE (stub)")
    return ctx