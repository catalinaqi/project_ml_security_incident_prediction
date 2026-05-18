# src/crispdm/registry/generators/phase2_generator_registry.py
from __future__ import annotations
import json
from crispdm.common.context_facade_common import RunContext
from crispdm.common.logging_adapter_common import get_logger
from crispdm.configuration.enum_registry_config import StepsPhase, StepOutputArtifact
from crispdm.data.persist_persister_data import save_parquet, save_json
from crispdm.registry.generator_registry_registry import register_artifact

log = get_logger(__name__)

# =============================================================================
# Phase 2 — Data Understanding artifact generators
# Steps: 2.1 Data Acquisition, 2.2 Data Description,
#        2.3 Data Quality Assessment, 2.4 Data Exploration
# =============================================================================

# -----------------------------------------------------------------------------
# Step 2.1 — Data Acquisition
# -----------------------------------------------------------------------------

@register_artifact(StepsPhase.STEP_2_1.value, StepOutputArtifact.SAMPLE_TRAIN_PARQUET)
def _write_sample_train_parquet(ctx: RunContext, path: str, **data) -> None:
    save_parquet(data["df_train"], ctx.phase2_dir / path, compression="snappy")
    log.info("[2.1] saved sample_train_parquet → %s", path)


@register_artifact(StepsPhase.STEP_2_1.value, StepOutputArtifact.SAMPLE_TEST_PARQUET)
def _write_sample_test_parquet(ctx: RunContext, path: str, **data) -> None:
    df_test = data.get("df_test")
    if df_test is not None and not df_test.empty:
        save_parquet(df_test, ctx.phase2_dir / path, compression="snappy")
        log.info("[2.1] saved sample_test_parquet → %s", path)


@register_artifact(StepsPhase.STEP_2_1.value, StepOutputArtifact.SAMPLE_STATS)
def _write_sample_stats(ctx: RunContext, path: str, **data) -> None:
    df_train = data["df_train"]
    df_test  = data.get("df_test")
    stats = {
        "train_shape":   list(df_train.shape),
        "test_shape":    list(df_test.shape) if df_test is not None and not df_test.empty else None,
        "train_columns": list(df_train.columns),
        "test_columns":  list(df_test.columns) if df_test is not None and not df_test.empty else None,
    }
    #(ctx.phase2_dir / path).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    save_json( stats,ctx.phase2_dir / path)
    log.info("[2.1] saved sample_stats → %s", path)


# -----------------------------------------------------------------------------
# Step 2.2 — Data Description
# -----------------------------------------------------------------------------

@register_artifact(StepsPhase.STEP_2_2.value, StepOutputArtifact.SCHEMA_SUMMARY)
def _write_schema_summary(ctx: RunContext, path: str, **data) -> None:
    #results = data.get("step_results", {})
    #summary = results.get("schema_summary", {})
    save_json({}, ctx.phase2_dir / path)
    log.info("[2.2] saved schema_summary → %s", path)

@register_artifact(StepsPhase.STEP_2_2.value, StepOutputArtifact.STATISTICS_SUMMARY)
def _write_statistics_summary(ctx: RunContext, path: str, **data) -> None:
    save_json({}, ctx.phase2_dir / path)
    log.info("[2.2] saved statistics_summary → %s", path)


# -----------------------------------------------------------------------------
# Step 2.3 — Data Quality Assessment
# -----------------------------------------------------------------------------

@register_artifact(StepsPhase.STEP_2_3.value, StepOutputArtifact.QUALITY_SUMMARY)
def _write_quality_summary(ctx: RunContext, path: str, **data) -> None:
    save_json({}, ctx.phase2_dir / path)
    log.info("[2.3] saved quality_summary → %s", path)

@register_artifact(StepsPhase.STEP_2_3.value, StepOutputArtifact.CRITICAL_ISSUES)
def _write_critical_issues(ctx: RunContext, path: str, **data) -> None:
    save_json({}, ctx.phase2_dir / path)
    log.info("[2.3] saved critical_issues → %s", path)


# -----------------------------------------------------------------------------
# Step 2.4 — Data Exploration
# -----------------------------------------------------------------------------

@register_artifact(StepsPhase.STEP_2_4.value, StepOutputArtifact.EXPLORATORY_SUMMARY)
def _write_exploratory_summary(ctx: RunContext, path: str, **data) -> None:
    save_json({}, ctx.phase2_dir / path)
    log.info("[2.4] saved exploratory_summary → %s", path)

@register_artifact(StepsPhase.STEP_2_4.value, StepOutputArtifact.PHASE2_REPORT)
def _write_phase2_report(ctx: RunContext, path: str, **data) -> None:
    save_json({}, ctx.phase2_dir / path)
    log.info("[2.4] saved phase2_report → %s", path)