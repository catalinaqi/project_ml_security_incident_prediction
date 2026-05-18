# src/crispdm/registry/generator_registry_registry.py
from __future__ import annotations
from typing import Any, Callable
from crispdm.common.logging_adapter_common import get_logger
from crispdm.common.context_facade_common import RunContext

log = get_logger(__name__)

# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# Central dispatch engine for the output_artifacts pattern used across all
# CRISP-DM phases (2-5).
#
# Each phase registers its generator functions via @register_artifact in
# registry/generators/phase{N}_generator_registry.py.
# write_output_artifacts() dispatches to them at runtime.
#
# Design patterns
# -----------------------------------------------------------------------------
# - GoF: Registry — central dict mapping (step_key, artifact_key) → callable.
# =============================================================================

_ARTIFACT_GENERATORS: dict[tuple[str, str], Callable] = {}


def register_artifact(step_key: str, artifact_key: str):
    """Decorator to register an artifact generator into the central registry.

    Parameters
    ----------
    step_key : str
        StepsPhase value (e.g. ``"step_2_1_data_acquisition"``).
    artifact_key : str
        Key from the YAML ``output_artifacts`` block
        (e.g. ``"sample_train_parquet"``).
    """
    def decorator(func: Callable) -> Callable:
        key = (step_key, artifact_key)
        if key in _ARTIFACT_GENERATORS:
            log.warning(
                "[register_artifact] overwriting generator step='%s' key='%s'",
                step_key, artifact_key,
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
    """Dispatch all output_artifacts defined in YAML for a given step.

    Parameters
    ----------
    ctx : RunContext
        Run context carrying phase directories.
    step_key : str
        StepsPhase value (e.g. ``"step_2_1_data_acquisition"``).
    step_cfg : dict[str, Any]
        Step config dict from YAML — must contain ``output_artifacts``.
    context_data : Any
        Data passed to each generator (e.g. ``df_train``, ``df_test``).
    """
    output_artifacts: dict[str, Any] = step_cfg.get("output_artifacts") or {}
    for artifact_key, artifact_path in output_artifacts.items():
        generator = _ARTIFACT_GENERATORS.get((step_key, artifact_key))
        if generator:
            log.debug(
                "[write_output_artifacts] step='%s' key='%s' path='%s'",
                step_key, artifact_key, artifact_path,
            )
            generator(ctx, artifact_path, **context_data)
        else:
            log.warning(
                "[write_output_artifacts] no generator for step='%s' key='%s' "
                "— add @register_artifact or remove key from YAML",
                step_key, artifact_key,
            )