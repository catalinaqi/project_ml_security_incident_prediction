# src/crispdm/registry/artifact_registry_registry.py
from __future__ import annotations
import json

from crispdm.common.context_facade_common import RunContext
from crispdm.common.logging_adapter_common import get_logger
from crispdm.configuration.enum_registry_config import StepsPhase, StepOutputArtifact
from crispdm.data.persist_utils_data import save_parquet

log = get_logger(__name__)
"""
=============================================================================
Why this module exists
-----------------------------------------------------------------------------
Central registry for the "output_artifacts" dispatch pattern used across all
CRISP-DM phases (2–5).

Every phase runner (phase2_understanding_runner_stage, phase3_preparation_runner_stage, …)
needs to write step-level derived artifacts at the end of each step.  Previously each
module had its own copy of the registry dict and dispatch loop — this module eliminates
that duplication.

Design
------
Each phase module registers its own generator functions using the @register_artifact
decorator.  The generators live close to the logic that produces the data they write.
The dispatch function ``write_output_artifacts()`` iterates the YAML ``output_artifacts``
block for the current step, looks up the registered generator, and calls it.

Flow
----
1. Phase runner finishes its technique-level work (which writes technique outputs).
2. Phase runner calls ``write_output_artifacts(ctx, step_key, step_cfg, **data)``.
3. The function reads ``step_cfg.output_artifacts`` (from YAML).
4. For each (artifact_key, artifact_path) it looks up the generator.
5. The generator receives ``(ctx, artifact_path, **data)`` and writes the file.

Key properties
--------------
- YAML is single source of truth for artifact paths.
- No file is written unless the key exists in YAML.
- No hardcoded paths in Python — paths come from the config.
=============================================================================
"""

from typing import Any, Callable

from crispdm.common.context_facade_common import RunContext
from crispdm.common.logging_adapter_common import get_logger

log = get_logger(__name__)

# Registry: (step_key, artifact_key) → callable(ctx, artifact_path, **data)
_ARTIFACT_GENERATORS: dict[tuple[str, str], Callable] = {}


def register_artifact(step_key: str, artifact_key: str):
    """Decorator to register an artifact generator function.

    Parameters
    ----------
    step_key : str
        StepsPhase value (e.g. ``"step_2_1_data_acquisition"``,
        ``"step_3_1_data_selection"``).
    artifact_key : str
        Key from the YAML ``output_artifacts`` dict (e.g. ``"sample_train_parquet"``,
        ``"final_features"``).

    Example
    -------
    .. code-block:: python

        @register_artifact(StepsPhase.STEP_2_1.value, StepOutputArtifact.SAMPLE_TRAIN_PARQUET)
        def _write_sample_train_parquet(ctx, path, **data):
            save_parquet(data["df_train"], ctx.phase2_dir / path)
    """
    def decorator(func: Callable) -> Callable:
        key = (step_key, artifact_key)
        if key in _ARTIFACT_GENERATORS:
            log.warning(
                "[register_artifact] overwriting existing generator for step='%s' "
                "artifact_key='%s'", step_key, artifact_key,
            )
        _ARTIFACT_GENERATORS[key] = func
        return func
    return decorator


def write_output_artifacts(
        ctx: RunContext,
        step_key: str,
        step_cfg: dict[str, Any],
        **context_data: Any,
) -> None:
    """Write all output_artifacts defined in YAML config for a given step.

    Iterates the ``output_artifacts`` dict from *step_cfg*, looks up each key in
    the global registry, and calls the registered generator function.

    Parameters
    ----------
    ctx : RunContext
        Run context with the appropriate phase directory set
        (e.g. ``ctx.phase2_dir``, ``ctx.phase3_dir``, etc.).
    step_key : str
        StepsPhase value (e.g. ``"step_2_1_data_acquisition"``).
    step_cfg : dict[str, Any]
        Raw step config dict from YAML (contains ``output_artifacts``).
    context_data : Any
        Extra data passed down to each generator (e.g. ``df_train``, ``df_test``,
        ``fitted_transformer``, ``model``, etc.).
    """
    output_artifacts: dict[str, Any] = step_cfg.get("output_artifacts") or {}
    for artifact_key, artifact_path in output_artifacts.items():
        generator = _ARTIFACT_GENERATORS.get((step_key, artifact_key))
        if generator:
            log.debug(
                "[write_output_artifacts] dispatching step='%s' key='%s' path='%s'",
                step_key, artifact_key, artifact_path,
            )
            generator(ctx, artifact_path, **context_data)
        else:
            log.warning(
                "[write_output_artifacts] no generator for step='%s' artifact_key='%s' — "
                "define a generator with @register_artifact or remove the key from YAML",
                step_key, artifact_key,
            )



# =============================================================================
# Step 2.1 — Initial Data Collection
# =============================================================================

@register_artifact(StepsPhase.STEP_2_1.value, StepOutputArtifact.SAMPLE_TRAIN_PARQUET)
def _write_sample_train_parquet(ctx: RunContext, path: str, **data) -> None:
    save_parquet(data["df_train"], ctx.phase2_dir / path, compression="snappy")
    log.info("[2.1] saved sample_train_parquet to %s", path)


@register_artifact(StepsPhase.STEP_2_1.value, StepOutputArtifact.SAMPLE_TEST_PARQUET)
def _write_sample_test_parquet(ctx: RunContext, path: str, **data) -> None:
    df_test = data.get("df_test")
    if df_test is not None and not df_test.empty:
        save_parquet(df_test, ctx.phase2_dir / path, compression="snappy")
        log.info("[2.1] saved sample_test_parquet to %s", path)


@register_artifact(StepsPhase.STEP_2_1.value, StepOutputArtifact.SAMPLE_STATS)
def _write_sample_stats(ctx: RunContext, path: str, **data) -> None:
    df_train = data["df_train"]
    df_test = data.get("df_test")
    stats = {
        "train_shape": list(df_train.shape),
        "test_shape": (
            list(df_test.shape)
            if df_test is not None and not df_test.empty
            else None
        ),
        "train_columns": list(df_train.columns),
        "test_columns": (
            list(df_test.columns)
            if df_test is not None and not df_test.empty
            else None
        ),
    }
    (ctx.phase2_dir / path).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log.info("[2.1] saved stats_comparison to %s", path)





# =============================================================================
# Step 2.2 — Data Description
# =============================================================================

@register_artifact(StepsPhase.STEP_2_2.value, StepOutputArtifact.SCHEMA_SUMMARY)
def _write_schema_summary(ctx: RunContext, path: str, **data) -> None:
    log.debug("[2.2] schema_summary artifact not yet implemented, path=%s", path)


@register_artifact(StepsPhase.STEP_2_2.value, StepOutputArtifact.STATISTICS_SUMMARY)
def _write_statistics_summary(ctx: RunContext, path: str, **data) -> None:
    log.debug("[2.2] statistics_summary artifact not yet implemented, path=%s", path)


# =============================================================================
# Step 2.3 — Data Quality Assessment
# =============================================================================

@register_artifact(StepsPhase.STEP_2_3.value, StepOutputArtifact.QUALITY_SUMMARY)
def _write_quality_summary(ctx: RunContext, path: str, **data) -> None:
    log.debug("[2.3] quality_summary artifact not yet implemented, path=%s", path)


@register_artifact(StepsPhase.STEP_2_3.value, StepOutputArtifact.CRITICAL_ISSUES)
def _write_critical_issues(ctx: RunContext, path: str, **data) -> None:
    log.debug("[2.3] critical_issues artifact not yet implemented, path=%s", path)


# =============================================================================
# Step 2.4 — Data Exploration
# =============================================================================

@register_artifact(StepsPhase.STEP_2_4.value, StepOutputArtifact.EXPLORATORY_SUMMARY)
def _write_exploratory_summary(ctx: RunContext, path: str, **data) -> None:
    log.debug("[2.4] exploratory_summary artifact not yet implemented, path=%s", path)


@register_artifact(StepsPhase.STEP_2_4.value, StepOutputArtifact.PHASE2_REPORT)
def _write_phase2_report(ctx: RunContext, path: str, **data) -> None:
    log.debug("[2.4] phase2_report artifact not yet implemented, path=%s", path)
