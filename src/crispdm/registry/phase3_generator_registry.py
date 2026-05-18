# src/crispdm/registry/generators/phase3_generator_registry.py
from __future__ import annotations
from crispdm.common.context_facade_common import RunContext
from crispdm.common.logging_adapter_common import get_logger
from crispdm.configuration.enum_registry_config import StepsPhase, StepOutputArtifact
from crispdm.data.persist_persister_data import save_json, save_parquet, save_pickle
from crispdm.registry.generator_registry_registry import register_artifact

log = get_logger(__name__)

# =============================================================================
# Phase 3 — Data Preparation artifact generators
# Steps: 3.1 Data Selection, 3.2 Data Cleaning,
#        3.3 Data Transformation, 3.5 Data Formatting
# =============================================================================

# -----------------------------------------------------------------------------
# Step 3.1 — Data Selection
# -----------------------------------------------------------------------------

@register_artifact(StepsPhase.STEP_3_1.value, StepOutputArtifact.FINAL_FEATURES)
def _write_final_features(ctx: RunContext, path: str, **data) -> None:
    df_train = data.get("df_train")
    if df_train is not None:
        report = {"final_features": sorted(df_train.columns.tolist()),
                  "count": len(df_train.columns)}
        save_json(report, ctx.phase3_dir / path)
        log.info("[3.1] saved final_features → %s", path)
    else:
        log.warning("[3.1] df_train is None, cannot save final_features")


@register_artifact(StepsPhase.STEP_3_1.value, StepOutputArtifact.OVERVIEW_PLOT)
def _write_overview_plot(ctx: RunContext, path: str, **data) -> None:
    log.debug("[3.1] overview_plot not yet implemented, path=%s", path)


# -----------------------------------------------------------------------------
# Step 3.2 — Data Cleaning
# -----------------------------------------------------------------------------

@register_artifact(StepsPhase.STEP_3_2.value, StepOutputArtifact.IMPUTATION_CONSOLIDATED)
def _write_imputation_consolidated(ctx: RunContext, path: str, **data) -> None:
    log.debug("[3.2] imputation_consolidated not yet implemented, path=%s", path)


@register_artifact(StepsPhase.STEP_3_2.value, StepOutputArtifact.CLEANING_SUMMARY)
def _write_cleaning_summary(ctx: RunContext, path: str, **data) -> None:
    log.debug("[3.2] cleaning_summary not yet implemented, path=%s", path)


# -----------------------------------------------------------------------------
# Step 3.3 — Data Transformation
# -----------------------------------------------------------------------------

@register_artifact(StepsPhase.STEP_3_3.value, StepOutputArtifact.TRANSFORMATION_SUMMARY)
def _write_transformation_summary(ctx: RunContext, path: str, **data) -> None:
    log.debug("[3.3] transformation_summary not yet implemented, path=%s", path)


# -----------------------------------------------------------------------------
# Step 3.5 — Data Formatting
# -----------------------------------------------------------------------------

@register_artifact(StepsPhase.STEP_3_5.value, StepOutputArtifact.TRAIN_PREPARED)
def _write_train_prepared(ctx: RunContext, path: str, **data) -> None:
    df_train = data.get("df_train")
    if df_train is not None:
        save_parquet(df_train, ctx.phase3_dir / path, compression="snappy")
        log.info("[3.5] saved train_prepared → %s", path)


@register_artifact(StepsPhase.STEP_3_5.value, StepOutputArtifact.TEST_PREPARED)
def _write_test_prepared(ctx: RunContext, path: str, **data) -> None:
    df_test = data.get("df_test")
    if df_test is not None and not df_test.empty:
        save_parquet(df_test, ctx.phase3_dir / path, compression="snappy")
        log.info("[3.5] saved test_prepared → %s", path)


@register_artifact(StepsPhase.STEP_3_5.value, StepOutputArtifact.TRANSFORMERS_PIPELINE)
def _write_transformers_pipeline(ctx: RunContext, path: str, **data) -> None:
    transformer = data.get("fitted_transformer")
    if transformer is not None:
        save_pickle(transformer, ctx.phase3_dir / path)
        log.info("[3.5] saved transformers_pipeline → %s", path)