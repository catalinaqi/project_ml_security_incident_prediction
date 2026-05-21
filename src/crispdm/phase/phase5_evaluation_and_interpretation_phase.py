# src/crispdm/phase/phase5_evaluation_and_interpretation_phase.py
"""CRISP-DM Phase 5 orchestrators – Evaluation & Interpretation.

Each function corresponds to a step defined in the pipeline config.
Follows the same pattern as :mod:`crispdm.phase.phase4_modeling_runner_phase`.
"""

from __future__ import annotations

from typing import Any,Dict,Optional,List

from crispdm.common.context_facade_common import RunContext
from crispdm.common.dict_facade_common import dget, enabled
from crispdm.common.logging_adapter_common import get_logger
from crispdm.configuration.enum_registry_config import PhaseDir, StepsPhase
from crispdm.data.persist_persister_data import save_json
from crispdm.interpretation.cluster_interpreter_interpretation import (
    interpret_cluster_profiles,
    load_feature_names,
    load_json,
)
from crispdm.registry.generator_registry_registry import write_output_artifacts

from pathlib import Path

# --- NEW IMPORTS for step 5.2 ---
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crispdm.interpretation.business_alignment_evaluator_interpretation import (
    compute_confusion_matrix,
    generate_alignment_plot,
)
from crispdm.reporting.artifact_persister_reporting import save_figure

# --- NEW IMPORT for step 5.3 ---
from crispdm.interpretation.pipeline_auditor_interpretation import (
    check_leakage_sanity,
    check_reproducibility,
)

# --- NEW IMPORT for step 5.4 ---
from crispdm.interpretation.deployment_reporter_interpretation import (
    evaluate_deployment_readiness,
    generate_recommendations,
)

log = get_logger(__name__)


def run_step_5_1(ctx: RunContext) -> RunContext:
    """Execute CRISP-DM Phase 5.1 – Interpretation of cluster results.

    Reads per‑algorithm profiling JSON (generated in Phase 4.5),
    extracts top features per cluster, maps feature indices to names
    (using the final schema from Phase 3.5), and produces interpretable
    profiles.

    Parameters
    ----------
    ctx : RunContext
        Run context containing the pipeline configuration under
        ``phases.phase5_evaluation_and_interpretation.steps.step_5_1_interpretation``.

    Returns
    -------
    RunContext
        Enriched with interpreted profiles in ``ctx.artifacts``.
    """
    step_key = StepsPhase.STEP_5_1.value
    step_cfg: dict[str, Any] = dget(
        ctx.config.phases.phase5_evaluation_and_interpretation.steps,
        step_key,
        {},
    )
    if not enabled(step_cfg, default=True):
        log.info("[5.1] step disabled – skipping")
        return ctx

    log.info("[5.1] start – run_id=%s", ctx.run_id)
    log.debug("[5.1] step_cfg keys: %s", list(step_cfg.keys()))

    # ------------------------------------------------------------------
    # Determine output directory (Phase 5)
    # ------------------------------------------------------------------
    phase5_dir = ctx.run_dir / PhaseDir.PHASE5.value
    phase5_dir.mkdir(parents=True, exist_ok=True)
    log.debug("[5.1] output directory: %s", phase5_dir)

    # ------------------------------------------------------------------
    # Extract technique configurations
    # ------------------------------------------------------------------
    methods_cfg: dict[str, Any] = step_cfg.get("methods", {}).get("cluster_interpretation", {})
    techniques: dict[str, Any] = methods_cfg.get("techniques", {})

    # We expect 'kmeans_n2_profiling' and 'kmeans_n3_profiling'
    # For each, we read the source profiling JSON, load feature names,
    # call interpret_cluster_profiles, and save the result.
    # We'll also build a consolidated knowledge base.

    # Load feature names once (shared by both techniques)
    feature_names: list[str] | None = None
    feature_names_source: str | None = None
    # Determine feature_names_source from the first technique that provides it
    for technique_name, technique_cfg in techniques.items():
        params = technique_cfg.get("params", {})
        if params.get("feature_names_source"):
            feature_names_source = params["feature_names_source"]
            break

    if feature_names_source:
        # Resolve path – could be under phase3 or phase4 directories
        feature_names_path = _resolve_artifact_path(ctx, feature_names_source)
        if feature_names_path and feature_names_path.exists():
            feature_names = load_feature_names(feature_names_path)
            log.info("[5.1] loaded %d feature names from %s", len(feature_names), feature_names_path)
        else:
            log.error(
                "[5.1] feature names source not found: %s. "
                "Will use fallback indices.",
                feature_names_source,
            )
    else:
        log.warning("[5.1] no 'feature_names_source' configured – using index names")

    # ------------------------------------------------------------------
    # Process each technique
    # ------------------------------------------------------------------
    interpreted_profiles: dict[str, Any] = {}

    for technique_name, technique_cfg in techniques.items():
        if not enabled(technique_cfg, default=True):
            log.info("[5.1] technique '%s' disabled – skipping", technique_name)
            continue

        params = technique_cfg.get("params", {})
        source_path = params.get("source", "")
        top_n = params.get("top_features", 10)
        output_path = technique_cfg.get("output", "")

        if not source_path:
            log.warning("[5.1] technique '%s' has no 'source' param – skipping", technique_name)
            continue

        # Locate the profiling JSON (should be under phase4 output)
        profiling_path = _resolve_artifact_path(ctx, source_path)
        if profiling_path is None or not profiling_path.exists():
            log.error(
                "[5.1] technique '%s' source file not found: %s",
                technique_name,
                source_path,
            )
            continue

        log.info("[5.1] loading profiling data from %s", profiling_path)
        profiling_data = load_json(profiling_path)

        # Interpret profiles
        try:
            profiles = interpret_cluster_profiles(
                profiling_data=profiling_data,
                feature_names=feature_names or _fallback_names(profiling_data),
                top_n=top_n,
            )
        except ValueError as e:
            log.error("[5.1] interpretation failed for '%s': %s", technique_name, e)
            continue

        interpreted_profiles[technique_name] = profiles

        # Save per‑technique output
        if output_path:
            save_json(profiles, phase5_dir / output_path)
            log.info("[5.1] saved '%s' profiles to %s", technique_name, output_path)

    # ------------------------------------------------------------------
    # Build consolidated knowledge base (threat_knowledge_base)
    # ------------------------------------------------------------------
    output_artifacts: dict[str, str] = step_cfg.get("output_artifacts", {})
    knowledge_base_path = output_artifacts.get("threat_knowledge_base", "")
    if knowledge_base_path and interpreted_profiles:
        knowledge_base = {
            "description": "Cluster interpretation profiles for all trained models",
            "models": interpreted_profiles,
            "feature_names_used": feature_names,
        }
        save_json(knowledge_base, phase5_dir / knowledge_base_path)
        log.info("[5.1] saved threat knowledge base to %s", knowledge_base_path)

    # ------------------------------------------------------------------
    # Store in context artifacts for downstream steps (5.2, 5.3, 5.4)
    # ------------------------------------------------------------------
    ctx.artifacts["interpreted_profiles"] = interpreted_profiles
    #ctx.artifacts["threat_knowledge_base"] = knowledge_base if knowledge_base_path else None

    knowledge_base = None
    if knowledge_base_path and interpreted_profiles:
        knowledge_base = {
            "description": "Cluster interpretation profiles for all trained models",
            "models": interpreted_profiles,
            "feature_names_used": feature_names,
        }
        save_json(knowledge_base, phase5_dir / knowledge_base_path)
        log.info("[5.1] saved threat knowledge base to %s", knowledge_base_path)

    ctx.artifacts["threat_knowledge_base"] = knowledge_base

    # ------------------------------------------------------------------
    # Write output artifacts for registry
    # ------------------------------------------------------------------
    write_output_artifacts(
        ctx,
        step_key=step_key,
        step_cfg=step_cfg,
        df_train=None,
        df_test=None,
        extra={
            "techniques_interpreted": list(interpreted_profiles.keys()),
            "feature_names_source": feature_names_source,
            "n_features": len(feature_names) if feature_names else 0,
        },
    )

    log.info("[5.1] done – %d technique(s) interpreted", len(interpreted_profiles))
    return ctx


# ---------------------------------------------------------------------------
# Helper: resolve artifact path across phase directories
# ---------------------------------------------------------------------------
def _resolve_artifact_path(ctx: RunContext, relative_path: str) -> Path | None:
    """Resolve a relative artifact path to an absolute file path.

    Tries candidate directories in order:
    1. The run root (``ctx.run_dir``).
    2. Phase 3 output directory (``PhaseDir.PHASE3.value``).
    3. Phase 4 output directory (``PhaseDir.PHASE4.value``).
    4. Phase 5 output directory (``PhaseDir.PHASE5.value``).

    Parameters
    ----------
    ctx : RunContext
        Current run context.
    relative_path : str
        Relative path as configured in YAML (e.g.
        ``"4.5.model_evaluation.profiling.kmeans_n2.json"``).

    Returns
    -------
    Path or None
        Resolved absolute path, or ``None`` if not found.
    """
    run_root = ctx.run_dir
    candidates = [
        run_root / relative_path,
        run_root / PhaseDir.PHASE3.value / relative_path,
        run_root / PhaseDir.PHASE4.value / relative_path,
        run_root / PhaseDir.PHASE5.value / relative_path,
    ]

    for candidate in candidates:
        if candidate.exists():
            log.debug("[_resolve_artifact_path] found at %s", candidate)
            return candidate.resolve()

    log.warning("[_resolve_artifact_path] file not found in any candidate path: %s", relative_path)
    return None



# =============================================================================
# STEP 5.2 — BUSINESS EVALUATION (Confusion matrices & alignment plot)
# =============================================================================


def run_step_5_2(ctx: RunContext) -> RunContext:
    """Execute CRISP-DM Phase 5.2 – Business Evaluation.

    Computes confusion matrices between cluster assignments and ground truth
    labels (IncidentGrade), and generates a stacked bar alignment plot.

    Parameters
    ----------
    ctx : RunContext
        Run context containing the pipeline configuration under
        ``phases.phase5_evaluation_and_interpretation.steps.step_5_2_business_evaluation``.

    Returns
    -------
    RunContext
        Enriched with confusion matrices and alignment plot in artifacts.
    """
    step_key = StepsPhase.STEP_5_2.value
    step_cfg: dict[str, Any] = dget(
        ctx.config.phases.phase5_evaluation_and_interpretation.steps,
        step_key,
        {},
    )
    if not enabled(step_cfg, default=True):
        log.info("[5.2] step disabled – skipping")
        return ctx

    log.info("[5.2] start – run_id=%s", ctx.run_id)

    # ------------------------------------------------------------------
    # Determine output directory (Phase 5)
    # ------------------------------------------------------------------
    phase5_dir = ctx.run_dir / PhaseDir.PHASE5.value
    phase5_dir.mkdir(parents=True, exist_ok=True)
    log.debug("[5.2] output directory: %s", phase5_dir)

    # ------------------------------------------------------------------
    # Load ground truth labels
    # ------------------------------------------------------------------
    read_strategy: dict[str, Any] = dget(ctx.config.phases.phase5_evaluation_and_interpretation, "read_strategy", {})
    train_labels_path_rel: str = read_strategy.get("train_labels", "")
    if not train_labels_path_rel:
        log.error("[5.2] 'train_labels' not configured in phase5 read_strategy")
        raise RuntimeError("[5.2] Missing train_labels path")

    gt_path = _resolve_artifact_path(ctx, train_labels_path_rel)
    if gt_path is None or not gt_path.exists():
        log.error("[5.2] ground truth file not found: %s", train_labels_path_rel)
        raise FileNotFoundError(f"[5.2] ground truth not found: {train_labels_path_rel}")

    log.info("[5.2] loading ground truth from %s", gt_path)
    df_gt = pd.read_parquet(gt_path)
    # Expect column 'label' per config (encoded)
    if "label" not in df_gt.columns:
        log.error("[5.2] ground truth file missing 'label' column – columns: %s", df_gt.columns.tolist())
        raise ValueError("[5.2] Missing 'label' column in ground truth")
    true_labels: np.ndarray = df_gt["label"].values.astype(int)
    log.info("[5.2] ground truth loaded – n=%d, classes=%s", len(true_labels), np.unique(true_labels))

    # ------------------------------------------------------------------
    # Load cluster assignments (sample from step 4.3)
    # ------------------------------------------------------------------
    cluster_assign_path_rel: str = read_strategy.get("cluster_assignments", "")
    if not cluster_assign_path_rel:
        log.error("[5.2] 'cluster_assignments' not configured in phase5 read_strategy")
        raise RuntimeError("[5.2] Missing cluster_assignments path")

    ca_path = _resolve_artifact_path(ctx, cluster_assign_path_rel)
    if ca_path is None or not ca_path.exists():
        log.error("[5.2] cluster assignments file not found: %s", cluster_assign_path_rel)
        raise FileNotFoundError(f"[5.2] cluster assignments not found: {cluster_assign_path_rel}")

    log.info("[5.2] loading cluster assignments from %s", ca_path)
    df_ca = pd.read_parquet(ca_path)
    # The parquet has one column per model (e.g., "kmeans_n2", "kmeans_n3")
    available_models = df_ca.columns.tolist()
    log.info("[5.2] available models in assignments: %s", available_models)

    # ------------------------------------------------------------------
    # Extract technique configurations
    # ------------------------------------------------------------------
    methods_cfg: dict[str, Any] = step_cfg.get("methods", {}).get("business_alignment", {})
    techniques: dict[str, Any] = methods_cfg.get("techniques", {})

    # We'll collect all computed confusion matrices for consolidated output
    confusion_results: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Process each confusion matrix technique
    # ------------------------------------------------------------------
    for technique_name, technique_cfg in techniques.items():
        if not enabled(technique_cfg, default=True):
            log.info("[5.2] technique '%s' disabled – skipping", technique_name)
            continue

        params = technique_cfg.get("params", {})
        output_path = technique_cfg.get("output", "")

        # Determine model name from cluster_source param (e.g., "kmeans_n2")
        cluster_source: str = params.get("cluster_source", "")
        if not cluster_source:
            log.warning("[5.2] technique '%s' has no 'cluster_source' param – skipping", technique_name)
            continue

        if cluster_source not in df_ca.columns:
            log.warning("[5.2] cluster_source '%s' not found in assignments – skipping", cluster_source)
            continue

        cluster_labels: np.ndarray = df_ca[cluster_source].values.astype(int)
        log.info("[5.2] computing %s with cluster_source=%s (n=%d, clusters=%s)",
                 technique_name, cluster_source, len(cluster_labels), np.unique(cluster_labels))

        # Determine parameters
        normalize: Optional[str] = params.get("normalize", None)
        collapse_top_n: Optional[int] = params.get("collapse_top_n", None)

        # Compute confusion matrix
        try:
            cm_result = compute_confusion_matrix(
                cluster_labels=cluster_labels,
                true_labels=true_labels,
                normalize=normalize,
                collapse_top_n=collapse_top_n,
            )
        except ValueError as e:
            log.error("[5.2] compute_confusion_matrix failed for '%s': %s", technique_name, e)
            continue

        confusion_results[technique_name] = cm_result

        # Save individual output
        if output_path:
            save_json(cm_result, phase5_dir / output_path)
            log.info("[5.2] saved confusion matrix for '%s' to %s", technique_name, output_path)

    # ------------------------------------------------------------------
    # Generate alignment plot (crosstab_distribution)
    # ------------------------------------------------------------------
    crosstab_cfg: dict[str, Any] = techniques.get("crosstab_distribution", {})
    if enabled(crosstab_cfg, default=True):
        plot_params = crosstab_cfg.get("params", {})
        targets: List[str] = plot_params.get("targets", [])
        plot_type: str = plot_params.get("plot_type", "stacked_bar")
        output_plot_path: str = crosstab_cfg.get("output", "")

        if targets:
            # Extract cluster labels for each target model
            cluster_labels_dict: Dict[str, np.ndarray] = {}
            missing_targets = []
            for target in targets:
                if target in df_ca.columns:
                    cluster_labels_dict[target] = df_ca[target].values.astype(int)
                else:
                    missing_targets.append(target)
                    log.warning("[5.2] target '%s' not in assignments – skipping in plot", target)
            if missing_targets:
                log.warning("[5.2] targets missing from data: %s", missing_targets)

            if cluster_labels_dict:
                fig = generate_alignment_plot(
                    cluster_labels_dict=cluster_labels_dict,
                    true_labels=true_labels,
                    targets=[t for t in targets if t in cluster_labels_dict],
                    plot_type=plot_type,
                )

                # Save plot
                if output_plot_path:
                    #save_figure(fig, phase5_dir / output_plot_path, dpi=150)
                    save_figure(fig, out_path=phase5_dir / output_plot_path, dpi=150)
                    log.info("[5.2] saved alignment plot to %s", output_plot_path)
                plt.close(fig)
            else:
                log.warning("[5.2] no valid targets for alignment plot")

    # ------------------------------------------------------------------
    # Save consolidated output artifacts
    # ------------------------------------------------------------------
    output_artifacts: dict[str, str] = step_cfg.get("output_artifacts", {})

    # confusion_matrices (consolidated)
    cm_consolidated_path: str = output_artifacts.get("confusion_matrices", "")
    if cm_consolidated_path and confusion_results:
        consolidated = {
            "description": "Confusion matrices for all evaluated models",
            "models": confusion_results,
            "ground_truth_column": "IncidentGrade",
        }
        save_json(consolidated, phase5_dir / cm_consolidated_path)
        log.info("[5.2] saved consolidated confusion matrices to %s", cm_consolidated_path)

    # alignment_plot is already saved above

    # ------------------------------------------------------------------
    # Store in context artifacts
    # ------------------------------------------------------------------
    ctx.artifacts["confusion_matrices"] = confusion_results
    ctx.artifacts["alignment_plot_path"] = output_plot_path if crosstab_cfg.get("output") else None

    # ------------------------------------------------------------------
    # Write output artifacts for registry
    # ------------------------------------------------------------------
    write_output_artifacts(
        ctx,
        step_key=step_key,
        step_cfg=step_cfg,
        df_train=df_gt,  # optional, for registry
        df_test=None,
        extra={
            "models_evaluated": list(confusion_results.keys()),
            "ground_truth_source": train_labels_path_rel,
            "cluster_assignments_source": cluster_assign_path_rel,
        },
    )

    log.info("[5.2] done – %d confusion matrix(es) computed", len(confusion_results))
    return ctx


# =============================================================================
# STEP 5.3 — PROCESS AUDIT (leakage check & reproducibility)
# =============================================================================


def run_step_5_3(ctx: RunContext) -> RunContext:
    """Execute CRISP-DM Phase 5.3 – Process Audit.

    Performs leakage sanity checks on selected features and verifies
    reproducibility of key pipeline artifacts.

    Parameters
    ----------
    ctx : RunContext
        Run context with config under
        ``phases.phase5_evaluation_and_interpretation.steps.step_5_3_process_audit``.

    Returns
    -------
    RunContext
        Enriched with audit report in artifacts.
    """
    step_key = StepsPhase.STEP_5_3.value
    step_cfg: dict[str, Any] = dget(
        ctx.config.phases.phase5_evaluation_and_interpretation.steps,
        step_key,
        {},
    )
    if not enabled(step_cfg, default=True):
        log.info("[5.3] step disabled – skipping")
        return ctx

    log.info("[5.3] start – run_id=%s", ctx.run_id)

    # ------------------------------------------------------------------
    # Determine output directory
    # ------------------------------------------------------------------
    phase5_dir = ctx.run_dir / PhaseDir.PHASE5.value
    phase5_dir.mkdir(parents=True, exist_ok=True)
    log.debug("[5.3] output directory: %s", phase5_dir)

    # ------------------------------------------------------------------
    # Extract technique configurations
    # ------------------------------------------------------------------
    methods_cfg: dict[str, Any] = step_cfg.get("methods", {}).get("reproducibility_audit", {})
    techniques: dict[str, Any] = methods_cfg.get("techniques", {})

    audit_results: Dict[str, Any] = {}

    # ---- Technique 1: leakage_sanity_check ----
    leakage_cfg: dict[str, Any] = techniques.get("leakage_sanity_check", {})
    if enabled(leakage_cfg, default=True):
        params = leakage_cfg.get("params", {})
        checked_features: List[str] = params.get("checked_features", [])
        leakage_risk: str = params.get("leakage_risk", "suspect")
        output_path: str = leakage_cfg.get("output", "")

        if checked_features:
            log.info("[5.3] running leakage check on features: %s", checked_features)
            leakage_report = check_leakage_sanity(
                checked_features=checked_features,
                leakage_risk=leakage_risk,
            )
            audit_results["leakage_sanity_check"] = leakage_report

            if output_path:
                save_json(leakage_report, phase5_dir / output_path)
                log.info("[5.3] saved leakage check to %s", output_path)
        else:
            log.warning("[5.3] leakage_sanity_check enabled but no checked_features configured")
    else:
        log.info("[5.3] leakage_sanity_check disabled – skipping")

    # ---- Technique 2: reproducibility_check ----
    repro_cfg: dict[str, Any] = techniques.get("reproducibility_check", {})
    if enabled(repro_cfg, default=True):
        params = repro_cfg.get("params", {})
        artifacts_to_verify: List[str] = params.get("artifacts_to_verify", [])
        target_seed: Optional[int] = params.get("target_seed")
        output_path: str = repro_cfg.get("output", "")

        if artifacts_to_verify:
            log.info("[5.3] running reproducibility check on %d artifacts", len(artifacts_to_verify))
            repro_report = check_reproducibility(
                artifacts_to_verify=artifacts_to_verify,
                target_seed=target_seed,
                run_dir=ctx.run_dir,  # base directory for resolution
            )
            audit_results["reproducibility_check"] = repro_report

            if output_path:
                save_json(repro_report, phase5_dir / output_path)
                log.info("[5.3] saved reproducibility check to %s", output_path)
        else:
            log.warning("[5.3] reproducibility_check enabled but no artifacts_to_verify configured")
    else:
        log.info("[5.3] reproducibility_check disabled – skipping")

    # ------------------------------------------------------------------
    # Save consolidated audit report (reproducibility_certificate)
    # ------------------------------------------------------------------
    output_artifacts: dict[str, str] = step_cfg.get("output_artifacts", {})
    cert_path: str = output_artifacts.get("reproducibility_certificate", "")
    if cert_path and audit_results:
        certificate = {
            "description": "Process audit report (leakage + reproducibility)",
            "timestamp": ctx.run_id,  # use run_id as timestamp
            "results": audit_results,
            "overall_status": (
                "PASS"
                if all(r.get("all_passed", True) for r in audit_results.values()
                       if isinstance(r, dict))
                else "WARNING"
            ),
        }
        save_json(certificate, phase5_dir / cert_path)
        log.info("[5.3] saved reproducibility certificate to %s", cert_path)

    # ------------------------------------------------------------------
    # Store in context artifacts
    # ------------------------------------------------------------------
    ctx.artifacts["audit_results"] = audit_results
    ctx.artifacts["reproducibility_certificate"] = certificate if cert_path else None

    # ------------------------------------------------------------------
    # Write output artifacts for registry
    # ------------------------------------------------------------------
    write_output_artifacts(
        ctx,
        step_key=step_key,
        step_cfg=step_cfg,
        df_train=None,
        df_test=None,
        extra={
            "leakage_checked": bool(audit_results.get("leakage_sanity_check")),
            "reproducibility_checked": bool(audit_results.get("reproducibility_check")),
            "audit_techniques": list(audit_results.keys()),
        },
    )

    log.info("[5.3] done – audit results saved")
    return ctx


# =============================================================================
# STEP 5.4 — DECISION MAKING (Deployment readiness & recommendations)
# =============================================================================

def run_step_5_4(ctx: RunContext) -> RunContext:
    """Execute CRISP-DM Phase 5.4 – Decision Making.

    Evaluates deployment readiness based on evaluation metrics (ARI,
    silhouette, etc.) and generates recommendations for the next phase.

    Parameters
    ----------
    ctx : RunContext
        Run context with config under
        ``phases.phase5_evaluation_and_interpretation.steps.step_5_4_decision_making``.

    Returns
    -------
    RunContext
        Enriched with readiness metrics and recommendations in artifacts.
    """
    step_key = StepsPhase.STEP_5_4.value
    step_cfg: dict[str, Any] = dget(
        ctx.config.phases.phase5_evaluation_and_interpretation.steps,
        step_key,
        {},
    )
    if not enabled(step_cfg, default=True):
        log.info("[5.4] step disabled – skipping")
        return ctx

    log.info("[5.4] start – run_id=%s", ctx.run_id)

    # ------------------------------------------------------------------
    # Determine output directory
    # ------------------------------------------------------------------
    phase5_dir = ctx.run_dir / PhaseDir.PHASE5.value
    phase5_dir.mkdir(parents=True, exist_ok=True)
    log.debug("[5.4] output directory: %s", phase5_dir)

    # ------------------------------------------------------------------
    # Load required input data from Phase 4.5 and Phase 5 read_strategy
    # ------------------------------------------------------------------
    read_strategy: dict[str, Any] = dget(
        ctx.config.phases.phase5_evaluation_and_interpretation, "read_strategy", {}
    )

    # 1. Evaluation summary (consolidated from 4.5)
    eval_summary_path_rel: str = read_strategy.get("evaluation_summary", "")
    evaluation_summary: dict = {}
    if eval_summary_path_rel:
        eval_path = _resolve_artifact_path(ctx, eval_summary_path_rel)
        if eval_path and eval_path.exists():
            evaluation_summary = load_json(eval_path)
            log.info("[5.4] loaded evaluation summary from %s", eval_path)
        else:
            log.warning("[5.4] evaluation_summary not found at %s – using empty", eval_summary_path_rel)
    else:
        log.warning("[5.4] 'evaluation_summary' not configured in read_strategy – using empty")

    # 2. ARI consolidated
    ari_consolidated_path_rel: str = read_strategy.get("ari_consolidated", "")
    ari_consolidated: dict = {}
    if ari_consolidated_path_rel:
        ari_path = _resolve_artifact_path(ctx, ari_consolidated_path_rel)
        if ari_path and ari_path.exists():
            ari_consolidated = load_json(ari_path)
            log.info("[5.4] loaded ARI consolidated from %s", ari_path)
        else:
            log.warning("[5.4] ari_consolidated not found at %s – using empty", ari_consolidated_path_rel)
    else:
        log.warning("[5.4] 'ari_consolidated' not configured in read_strategy – using empty")

    # ------------------------------------------------------------------
    # Extract technique configurations
    # ------------------------------------------------------------------
    methods_cfg: dict[str, Any] = step_cfg.get("methods", {}).get("downstream_strategy", {})
    techniques: dict[str, Any] = methods_cfg.get("techniques", {})

    # ---- Technique 1: evaluate_deployment_readiness ----
    readiness_cfg: dict[str, Any] = techniques.get("evaluate_deployment_readiness", {})
    readiness_metrics: dict = {}
    if enabled(readiness_cfg, default=True):
        params = readiness_cfg.get("params", {})
        output_path: str = readiness_cfg.get("output", "")

        log.info("[5.4] evaluating deployment readiness...")
        readiness_metrics = evaluate_deployment_readiness(
            evaluation_summary=evaluation_summary,
            ari_consolidated=ari_consolidated,
            params=params,
        )

        if output_path:
            save_json(readiness_metrics, phase5_dir / output_path)
            log.info("[5.4] saved readiness metrics to %s", output_path)
    else:
        log.info("[5.4] evaluate_deployment_readiness disabled – skipping")

    # ---- Technique 2: recommendations ----
    recommendations_cfg: dict[str, Any] = techniques.get("recommendations", {})
    recommendations: dict = {}
    if enabled(recommendations_cfg, default=True):
        params = recommendations_cfg.get("params", {})
        output_path: str = recommendations_cfg.get("output", "")

        log.info("[5.4] generating recommendations...")
        recommendations = generate_recommendations(
            readiness_metrics=readiness_metrics,
            params=params,
        )

        if output_path:
            save_json(recommendations, phase5_dir / output_path)
            log.info("[5.4] saved recommendations to %s", output_path)
    else:
        log.info("[5.4] recommendations disabled – skipping")

    # ------------------------------------------------------------------
    # Save consolidated output artifacts
    # ------------------------------------------------------------------
    output_artifacts: dict[str, str] = step_cfg.get("output_artifacts", {})

    # deployment_readiness (may be same as readiness_metrics path)
    deploy_readiness_path: str = output_artifacts.get("deployment_readiness", "")
    if deploy_readiness_path and readiness_metrics:
        save_json(readiness_metrics, phase5_dir / deploy_readiness_path)
        log.info("[5.4] saved deployment_readiness to %s", deploy_readiness_path)

    # recommendations (may be same as recommendations path)
    rec_path: str = output_artifacts.get("recommendations", "")
    if rec_path and recommendations:
        save_json(recommendations, phase5_dir / rec_path)
        log.info("[5.4] saved recommendations to %s", rec_path)

    # ------------------------------------------------------------------
    # Store in context artifacts
    # ------------------------------------------------------------------
    ctx.artifacts["readiness_metrics"] = readiness_metrics
    ctx.artifacts["recommendations"] = recommendations

    # ------------------------------------------------------------------
    # Write output artifacts for registry
    # ------------------------------------------------------------------
    write_output_artifacts(
        ctx,
        step_key=step_key,
        step_cfg=step_cfg,
        df_train=None,
        df_test=None,
        extra={
            "readiness_evaluated": bool(readiness_metrics),
            "recommendations_generated": bool(recommendations),
            "winning_model": readiness_metrics.get("winning_model", ""),
            "threshold_met": readiness_metrics.get("threshold_met", False),
        },
    )

    log.info("[5.4] done – deployment readiness evaluated, recommendations generated")
    return ctx

# =============================================================================
# (Keep the helpers _resolve_artifact_path and _fallback_names from above)
# =============================================================================

def _fallback_names(profiling_data: dict) -> list[str]:
    """Generate fallback feature names based on indices found in profiling data.

    Parameters
    ----------
    profiling_data : dict
        First cluster entry is used to determine the maximum index.

    Returns
    -------
    list[str]
        List of names like ``"feature_0"``, ``"feature_1"``, etc.
    """
    max_idx = 0
    for cluster_data in profiling_data.values():
        if isinstance(cluster_data, dict) and "features" in cluster_data:
            for idx_str in cluster_data["features"].keys():
                try:
                    max_idx = max(max_idx, int(idx_str))
                except ValueError:
                    continue
    if max_idx == 0:
        return ["feature_0"]
    return [f"feature_{i}" for i in range(max_idx + 1)]