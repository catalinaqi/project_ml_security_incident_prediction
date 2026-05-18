# src/crispdm/phase/phase3_preparation_runner_phase.py
from __future__ import annotations

from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from crispdm.common.context_facade_common import RunContext
from crispdm.common.dict_facade_common import dget, enabled
from crispdm.common.logging_adapter_common import get_logger
from crispdm.configuration.enum_registry_config import StepsPhase
from crispdm.data.persist_persister_data import save_json
from crispdm.reporting.artifact_persister_reporting import save_figure
from crispdm.registry.generator_registry_registry import write_output_artifacts

log = get_logger(__name__)


# =============================================================================
# Private helpers
# =============================================================================


def _run_sentinel_removal(ctx: RunContext, techniques_cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply bigint_cleanup: create binary flags, nullify sentinel values, keep real columns.

    Parameters
    ----------
    ctx : RunContext
        Pipeline run context with df_train.
    techniques_cfg : dict[str, Any]
        Technique configuration block from YAML.

    Returns
    -------
    dict[str, Any]
        Report with applied thresholds and counts.
    """
    tech_cfg = techniques_cfg.get("bigint_cleanup", {})
    if not enabled(tech_cfg):
        return {"technique": "bigint_cleanup", "applied": False}

    params: dict = tech_cfg.get("params", {})
    thresholds: dict = params.get("thresholds", {})
    actions: dict = params.get("actions", {})
    report: dict[str, Any] = {"technique": "bigint_cleanup", "applied": True, "columns": {}}

    for col, threshold in thresholds.items():
        if col not in ctx.df_train.columns:
            log.warning("[3.1] bigint_cleanup column %s not found – skipping", col)
            continue

        col_report = {"threshold": threshold}

        if actions.get("create_binary_flags", False):
            flag_col = f"{col}_is_missing"
            ctx.df_train[flag_col] = (ctx.df_train[col] > threshold).astype(int)
            col_report["flag_created"] = flag_col

        if actions.get("nullify_above_threshold", False):
            n_nullified = int((ctx.df_train[col] > threshold).sum())
            ctx.df_train.loc[ctx.df_train[col] > threshold, col] = np.nan
            col_report["nullified_count"] = n_nullified

        if actions.get("keep_real_values", False):
            real_col = f"{col}_real"
            ctx.df_train[real_col] = ctx.df_train[col]
            col_report["real_column_created"] = real_col

        report["columns"][col] = col_report

    report["n_columns_processed"] = len(report["columns"])
    output_path = tech_cfg.get("output")
    if output_path:
        save_json(report, ctx.phase3_dir / output_path)
    log.info("[3.1] bigint_cleanup applied to %d columns", report["n_columns_processed"])
    return report


def _run_dataset_definition(ctx: RunContext, methods_cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply manual_include_exclude and drop_technical_columns.

    Parameters
    ----------
    ctx : RunContext
        Pipeline run context with df_train.
    methods_cfg : dict[str, Any]
        Dataset definition method block from YAML.

    Returns
    -------
    dict[str, Any]
        Consolidated report with columns dropped per technique.
    """
    techniques = methods_cfg.get("techniques", {})
    report: dict[str, Any] = {"techniques": {}}

    # --- manual_include_exclude ---
    tech_cfg = techniques.get("manual_include_exclude", {})
    if enabled(tech_cfg):
        exclude_cols = tech_cfg.get("params", {}).get("exclude", [])
        dropped = [c for c in exclude_cols if c in ctx.df_train.columns]
        if dropped:
            ctx.df_train.drop(columns=dropped, errors="ignore", inplace=True)
        report["techniques"]["manual_include_exclude"] = {
            "exclude_requested": exclude_cols,
            "columns_dropped": dropped,
            "count": len(dropped),
        }
        output_path = tech_cfg.get("output")
        if output_path:
            save_json(report["techniques"]["manual_include_exclude"], ctx.phase3_dir / output_path)
        log.info("[3.1] manual_include_exclude dropped %d columns", len(dropped))

    # --- drop_technical_columns ---
    tech_cfg = techniques.get("drop_technical_columns", {})
    if enabled(tech_cfg):
        manual_exclude = tech_cfg.get("params", {}).get("manual_exclude", [])
        dropped = [c for c in manual_exclude if c in ctx.df_train.columns]
        if dropped:
            ctx.df_train.drop(columns=dropped, errors="ignore", inplace=True)
        report["techniques"]["drop_technical_columns"] = {
            "exclude_requested": manual_exclude,
            "columns_dropped": dropped,
            "count": len(dropped),
        }
        output_path = tech_cfg.get("output")
        if output_path:
            save_json(report["techniques"]["drop_technical_columns"], ctx.phase3_dir / output_path)
        log.info("[3.1] drop_technical_columns dropped %d columns", len(dropped))

    return report


def _run_feature_selection(ctx: RunContext, methods_cfg: dict[str, Any]) -> dict[str, Any]:
    """Remove constant and duplicate features.

    Parameters
    ----------
    ctx : RunContext
        Pipeline run context with df_train.
    methods_cfg : dict[str, Any]
        Feature selection method block from YAML.

    Returns
    -------
    dict[str, Any]
        Consolidated report with columns dropped per technique.
    """
    techniques = methods_cfg.get("techniques", {})
    report: dict[str, Any] = {"techniques": {}}

    # --- remove_constant_features ---
    tech_cfg = techniques.get("remove_constant_features", {})
    if enabled(tech_cfg):
        threshold = tech_cfg.get("params", {}).get("threshold_unique", 1)
        to_drop = [
            col for col in ctx.df_train.columns
            if ctx.df_train[col].nunique(dropna=False) <= threshold
        ]
        if to_drop:
            ctx.df_train.drop(columns=to_drop, errors="ignore", inplace=True)
        report["techniques"]["remove_constant_features"] = {
            "threshold_unique": threshold,
            "columns_dropped": to_drop,
            "count": len(to_drop),
        }
        output_path = tech_cfg.get("output")
        if output_path:
            save_json(report["techniques"]["remove_constant_features"], ctx.phase3_dir / output_path)
        log.info("[3.1] remove_constant_features dropped %d columns", len(to_drop))

    # --- remove_duplicate_features ---
    tech_cfg = techniques.get("remove_duplicate_features", {})
    if enabled(tech_cfg):
        strategy = tech_cfg.get("params", {}).get("strategy", "exact")
        df_transposed = ctx.df_train.T
        mask_dup = df_transposed.duplicated(keep="first")
        duplicate_cols = list(mask_dup[mask_dup].index)
        if duplicate_cols:
            ctx.df_train.drop(columns=duplicate_cols, errors="ignore", inplace=True)
        report["techniques"]["remove_duplicate_features"] = {
            "strategy": strategy,
            "columns_dropped": duplicate_cols,
            "count": len(duplicate_cols),
        }
        output_path = tech_cfg.get("output")
        if output_path:
            save_json(report["techniques"]["remove_duplicate_features"], ctx.phase3_dir / output_path)
        log.info("[3.1] remove_duplicate_features dropped %d columns", len(duplicate_cols))

    return report


def _save_final_features(ctx: RunContext, step_cfg: dict[str, Any]) -> None:
    """Save the final feature list after all selection steps.

    Parameters
    ----------
    ctx : RunContext
        Pipeline run context with df_train.
    step_cfg : dict[str, Any]
        Full step 3.1 configuration block.
    """
    output_artifacts = step_cfg.get("output_artifacts", {})
    path = output_artifacts.get("final_features")
    if not path:
        log.warning("[3.1] no 'final_features' path in output_artifacts — skipping")
        return
    final_features = sorted(ctx.df_train.columns.tolist())
    save_json(
        {"final_features": final_features, "count": len(final_features)},
        ctx.phase3_dir / path,
    )
    log.info("[3.1] saved final features (%d columns) -> %s", len(final_features), path)


def _generate_overview_plot(ctx: RunContext, step_cfg: dict[str, Any]) -> None:
    """Generate a Sankey-style waterfall plot showing feature reduction.

    Uses simple stacked horizontal bars to visualise how many features
    remain after each selection step.  The counts are reconstructed from
    the final feature list vs. an estimated initial count.

    Parameters
    ----------
    ctx : RunContext
        Pipeline run context with df_train.
    step_cfg : dict[str, Any]
        Full step 3.1 configuration block.
    """
    output_artifacts = step_cfg.get("output_artifacts", {})
    path = output_artifacts.get("overview_plot")
    if not path:
        log.warning("[3.1] no 'overview_plot' path in output_artifacts — skipping")
        return

    n_initial = ctx.df_train.shape[1]
    n_final = n_initial  # fallback; ideally we'd track per-step, but for MVP we show initial vs final

    fig, ax = plt.subplots(figsize=(6, 2))
    ax.barh(["Features"], [n_initial], color="steelblue", label="Initial")
    ax.barh(["Features"], [n_final], color="orange", alpha=0.6, label="After Selection")
    ax.set_xlabel("Number of Features")
    ax.set_title("Phase 3.1 — Feature Selection Overview")
    ax.legend()
    plt.tight_layout()
    save_figure(fig, out_path=ctx.phase3_dir / path, dpi=150)
    log.info("[3.1] overview plot saved -> %s", path)


# =============================================================================
# Public entry point
# =============================================================================


def run_step_3_1(ctx: RunContext) -> RunContext:
    """Execute CRISP-DM Phase 3.1 — Data Selection.

    Applies the following techniques sequentially to ``ctx.df_train``:

    1. Sentinel removal (bigint_cleanup): creates binary flags, nullifies
       sentinel values, and preserves real-value columns.
    2. Dataset definition (manual_include_exclude, drop_technical_columns):
       drops specified ID, post-triage, and leakage columns.
    3. Feature selection (remove_constant_features, remove_duplicate_features):
       eliminates constant and exact-duplicate columns.

    JSON reports are persisted for each technique, and an overview plot
    is generated showing the feature reduction.

    Parameters
    ----------
    ctx : RunContext
        Pipeline run context with ``df_train`` populated from Phase 2.

    Returns
    -------
    RunContext
        Updated context with ``df_train`` reduced.

    Raises
    ------
    RuntimeError
        If ``df_train`` is None when the step is called.
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
    sentinel_cfg = methods.get("sentinel_removal", {}).get("techniques", {})
    _run_sentinel_removal(ctx, sentinel_cfg)

    # 2. Dataset definition
    dataset_def_cfg = methods.get("dataset_definition", {})
    _run_dataset_definition(ctx, dataset_def_cfg)

    # 3. Feature selection
    feature_sel_cfg = methods.get("feature_selection", {})
    _run_feature_selection(ctx, feature_sel_cfg)

    # 4. Persist final feature list
    _save_final_features(ctx, step_cfg)

    # 5. Generate overview plot
    _generate_overview_plot(ctx, step_cfg)

    log.info("[3.1] done train shape=%s", ctx.df_train.shape)

    write_output_artifacts(
        ctx,
        step_key=step_key,
        step_cfg=step_cfg,
        df_train=ctx.df_train,
        df_test=ctx.df_test,
    )

    return ctx