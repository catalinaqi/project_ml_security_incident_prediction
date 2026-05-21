# src/crispdm/phase/phase4_modeling_runner_phase.py
"""CRISP-DM Phase 4 orchestrators.

Each function corresponds to a step defined in the pipeline config.
Follows the same pattern as :mod:`crispdm.phase.phase3_preparation_runner_phase`.
"""

from __future__ import annotations

from typing import Any

from crispdm.common.context_facade_common import RunContext
from crispdm.common.dict_facade_common import dget, enabled
from crispdm.common.logging_adapter_common import get_logger
from crispdm.configuration.enum_registry_config import PhaseDir, StepsPhase
from crispdm.data.persist_persister_data import save_json
from crispdm.model.algorithm_selector_model import select_algorithms
from crispdm.model.knn_distance_analyzer_model import knn_distance_analysis
from crispdm.reporting.artifact_persister_reporting import save_figure
from crispdm.registry.generator_registry_registry import write_output_artifacts
from crispdm.model.clustering_trainer_model import train_clustering_models
from crispdm.model.test_design_reporter_model import generate_test_design
#from crispdm.model.clustering_evaluator_model import evaluate_all_models
from crispdm.model.clustering_evaluator_model import evaluate_all_models
from pathlib import Path
import pandas as pd
import numpy as np


log = get_logger(__name__)


def run_step_4_1(ctx: RunContext) -> RunContext:
    """Execute CRISP-DM Phase 4.1 – Algorithm Selection.

    Reads the step configuration, calls :func:`select_algorithms`,
    persists JSON reports for each selected algorithm, and writes
    output artifacts.

    Parameters
    ----------
    ctx : RunContext
        Run context with ``config`` containing
        ``phase4_data_modeling.steps.step_4_1_algorithm_selection``.

    Returns
    -------
    RunContext
        Same ``ctx`` (may be extended with phase4 artifacts later).
    """
    step_key = StepsPhase.STEP_4_1.value
    step_cfg: dict[str, Any] = dget(
        ctx.config.phases.phase4_data_modeling.steps, step_key, {}
    )
    if not enabled(step_cfg, default=True):
        log.info("[4.1] step disabled – skipping")
        return ctx

    log.info("[4.1] start – run_id=%s", ctx.run_id)

    # ------------------------------------------------------------------
    # Determine output directory for phase4 artifacts
    # ------------------------------------------------------------------
    # PhaseDir.PHASE4 is expected to be "phase4" (or similar)
    phase4_dir = ctx.run_dir / PhaseDir.PHASE4.value
    phase4_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Call pure selector logic
    # ------------------------------------------------------------------
    methods_cfg: dict[str, Any] = step_cfg.get("methods", {}).get("algorithm_selection", {})
    selected = select_algorithms(methods_cfg)

    # ------------------------------------------------------------------
    # Persist per‑algorithm reports
    # ------------------------------------------------------------------
    #artifacts_saved = []
    for algo in selected:
        output_path = algo.get("output", "")
        if not output_path:
            log.warning("[4.1] algorithm '%s' has no output path – skipping persist", algo["name"])
            continue

        # Build a report that includes the selection metadata
        report = {
            "algorithm": algo["name"],
            "enabled": algo["enabled"],
            "priority": algo["priority"],
            "params": algo.get("params", {}),
            "note": "Algorithm selected for hyperparameter tuning in step 4.3",
        }
        save_json(report, phase4_dir / output_path)
        #artifacts_saved.append(algo["name"])
        log.info("[4.1] saved algorithm report for '%s' → %s", algo["name"], output_path)

    # ------------------------------------------------------------------
    # Save a consolidated selection summary
    # ------------------------------------------------------------------
    output_artifacts: dict[str, str] = step_cfg.get("output_artifacts", {})
    if output_artifacts and selected:
        # Save algorithms_selected consolidated summary
        algo_summary_path = output_artifacts.get("algorithms_selected", "")
        if algo_summary_path:
            consolidated = {
                "selected_algorithms": [a["name"] for a in selected],
                "algorithm_details": selected,
                "count": len(selected),
            }
            save_json(consolidated, phase4_dir / algo_summary_path)
            log.info("[4.1] saved consolidated selection summary → %s", algo_summary_path)

    # ------------------------------------------------------------------
    # Write output artifacts for registry / downstream steps
    # ------------------------------------------------------------------
    write_output_artifacts(
        ctx,
        step_key=step_key,
        step_cfg=step_cfg,
        df_train=None,
        df_test=None,
        extra={"selected_algorithms": [a["name"] for a in selected]},
    )

    log.info("[4.1] done – selected %d algorithm(s): %s",
             len(selected), [a["name"] for a in selected])
    return ctx


# =============================================================================
# STEP 4.2 — PRETRAIN ANALYSIS (k‑NN distance for DBSCAN eps)
# =============================================================================


def run_step_4_2(ctx: RunContext) -> RunContext:
    """Execute CRISP-DM Phase 4.2 – Pretrain Analysis (k‑NN distance).

    Computes k‑nearest neighbour distances on the training data, persists
    statistics, an elbow plot, and a suggested eps value for DBSCAN.

    Parameters
    ----------
    ctx : RunContext
        Run context with ``config`` containing
        ``phase4_data_modeling.steps.step_4_2_pretrain_analysis``.
        Also requires ``ctx.artifacts["X_train"]`` from Phase 3.5.

    Returns
    -------
    RunContext
        Same ``ctx`` enriched with step 4.2 results (stored in artifacts).
    """
    step_key = StepsPhase.STEP_4_2.value
    step_cfg: dict[str, Any] = dget(
        ctx.config.phases.phase4_data_modeling.steps, step_key, {}
    )
    if not enabled(step_cfg, default=True):
        log.info("[4.2] step disabled – skipping")
        return ctx

    log.info("[4.2] start – run_id=%s", ctx.run_id)

    # ------------------------------------------------------------------
    # Validate required input data
    # ------------------------------------------------------------------
    X_train = ctx.artifacts.get("X_train")
    if X_train is None:
        error_msg = (
            "[4.2] X_train not found in ctx.artifacts. "
            "Ensure Phase 3.5 (data formatting) completed and stored the numpy array."
        )
        log.error(error_msg)
        raise RuntimeError(error_msg)

    if not hasattr(X_train, "shape") or X_train.ndim != 2:
        raise RuntimeError(
            f"[4.2] X_train must be a 2D numpy array, got type={type(X_train)}, ndim={getattr(X_train, 'ndim', '?')}"
        )

    log.info("[4.2] input shape=%s", X_train.shape)

    # ------------------------------------------------------------------
    # Determine output directory for phase4 artifacts
    # ------------------------------------------------------------------
    phase4_dir = ctx.run_dir / PhaseDir.PHASE4.value
    phase4_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Extract technique configuration
    # ------------------------------------------------------------------
    methods_cfg: dict[str, Any] = step_cfg.get("methods", {}).get("knn_distance_analysis", {})
    techniques: dict[str, Any] = methods_cfg.get("techniques", {})
    knn_cfg: dict[str, Any] = techniques.get("knn_distances", {})

    if not enabled(knn_cfg):
        log.warning("[4.2] knn_distances technique disabled – no analysis performed")
        # Still write output artifacts to mark the step as done
        write_output_artifacts(
            ctx, step_key=step_key, step_cfg=step_cfg, df_train=None, df_test=None,
            extra={"knn_enabled": False},
        )
        return ctx

    # ------------------------------------------------------------------
    # Call pure analysis function
    # ------------------------------------------------------------------
    params: dict[str, Any] = knn_cfg.get("params", {})
    log.debug("[4.2] calling knn_distance_analysis with params=%s", params)
    try:
        stats, eps_suggested, fig = knn_distance_analysis(X_train, params)
    except ValueError as e:
        log.error("[4.2] knn_distance_analysis failed: %s", e)
        raise

    # ------------------------------------------------------------------
    # Persist artifacts
    # ------------------------------------------------------------------
    # 1. Statistics JSON
    output_stats = knn_cfg.get("output_stats", "")
    if output_stats:
        save_json(stats, phase4_dir / output_stats)
        log.info("[4.2] saved statistics to %s", output_stats)
    else:
        log.warning("[4.2] no output_stats path configured – skipping save")

    # 2. Elbow plot
    output_plot = knn_cfg.get("output_plot", "")
    if output_plot and fig is not None:
        save_figure(fig, out_path=phase4_dir / output_plot, dpi=150)
        log.info("[4.2] saved elbow plot to %s", output_plot)
    elif output_plot:
        log.warning("[4.2] figure is None – cannot save plot")
    else:
        log.debug("[4.2] no output_plot path configured – skipping save")

    # 3. eps suggestion
    output_eps = knn_cfg.get("output_eps_suggested", "")
    if output_eps:
        save_json(eps_suggested, phase4_dir / output_eps)
        log.info("[4.2] saved eps suggestion to %s", output_eps)

    # ------------------------------------------------------------------
    # Store results in context artifacts for downstream steps
    # ------------------------------------------------------------------
    ctx.artifacts["knn_stats"] = stats
    ctx.artifacts["eps_suggested"] = eps_suggested
    ctx.artifacts["knn_figure"] = fig  # optional, for later reuse


    # ------------------------------------------------------------------
    # Save output_artifacts (from step_cfg)
    # ------------------------------------------------------------------
    output_artifacts: dict[str, str] = step_cfg.get("output_artifacts", {})
    if output_artifacts:
        # 1. eps_recommendation (reuse eps_suggested)
        eps_rec_path = output_artifacts.get("eps_recommendation", "")
        if eps_rec_path:
            save_json(eps_suggested, phase4_dir / eps_rec_path)
            log.info("[4.2] saved eps_recommendation → %s", eps_rec_path)

        # 2. knn_distance_summary (reuse stats)
        summary_path = output_artifacts.get("knn_distance_summary", "")
        if summary_path:
            save_json(stats, phase4_dir / summary_path)
            log.info("[4.2] saved knn_distance_summary → %s", summary_path)

        # 3. sample_metadata (extract from stats)
        sample_meta_path = output_artifacts.get("sample_metadata", "")
        if sample_meta_path:
            sample_meta = {
                "n_samples_used": stats.get("n_samples_used"),
                "k": stats.get("k"),
                "metric": stats.get("metric"),
                "subsample_size": params.get("sample_size"),
                "random_state": params.get("random_state"),
            }
            save_json(sample_meta, phase4_dir / sample_meta_path)
            log.info("[4.2] saved sample_metadata → %s", sample_meta_path)

        # 4. eps_validation_preview (percentiles table)
        eps_val_path = output_artifacts.get("eps_validation_preview", "")
        if eps_val_path:
            validation_preview = {
                "percentiles": eps_suggested.get("all_percentiles", {}),
                "suggested_eps": eps_suggested.get("eps_suggested"),
                "recommendation": eps_suggested.get("recommendation"),
            }
            save_json(validation_preview, phase4_dir / eps_val_path)
            log.info("[4.2] saved eps_validation_preview → %s", eps_val_path)

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
            "knn_enabled": True,
            "eps_suggested_value": eps_suggested.get("eps_suggested"),
            "n_samples_used": stats.get("n_samples_used"),
        },
    )

    log.info("[4.2] done – eps_suggested=%s", eps_suggested.get("eps_suggested", "N/A"))
    return ctx

# =============================================================================
# STEP 4.3 — MODEL TRAINING WITH HYPERPARAMETER TUNING
# =============================================================================


def run_step_4_3(ctx: RunContext) -> RunContext:
    """Execute CRISP-DM Phase 4.3 – Model Training with hyperparameter tuning.

    Trains clustering models (KMeans, DBSCAN) via grid search over
    hyperparameter grids defined in the config.  Persists the best model,
    cluster assignments, and evaluation summaries.

    Parameters
    ----------
    ctx : RunContext
        Run context with ``config`` containing
        ``phase4_data_modeling.steps.step_4_3_model_training``.
        Requires ``ctx.artifacts["X_train"]`` from Phase 3.5.

    Returns
    -------
    RunContext
        Same ``ctx`` enriched with trained models and cluster labels in artifacts.
    """
    step_key = StepsPhase.STEP_4_3.value
    step_cfg: dict[str, Any] = dget(
        ctx.config.phases.phase4_data_modeling.steps, step_key, {}
    )
    if not enabled(step_cfg, default=True):
        log.info("[4.3] step disabled – skipping")
        return ctx

    log.info("[4.3] start – run_id=%s", ctx.run_id)

    # ------------------------------------------------------------------
    # Validate required input data
    # ------------------------------------------------------------------
    X_train = ctx.artifacts.get("X_train")
    if X_train is None:
        error_msg = (
            "[4.3] X_train not found in ctx.artifacts. "
            "Ensure Phase 3.5 completed and stored the numpy array."
        )
        log.error(error_msg)
        raise RuntimeError(error_msg)

    if not hasattr(X_train, "shape") or X_train.ndim != 2:
        raise RuntimeError(
            f"[4.3] X_train must be a 2D numpy array, got type={type(X_train)}, "
            f"ndim={getattr(X_train, 'ndim', '?')}"
        )

    log.info("[4.3] input shape=%s", X_train.shape)

    # ------------------------------------------------------------------
    # Ensure output directory exists
    # ------------------------------------------------------------------
    phase4_dir = ctx.run_dir / PhaseDir.PHASE4.value
    phase4_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Extract configuration for model_training (corrected)
    # ------------------------------------------------------------------
    methods_cfg: dict[str, Any] = step_cfg.get("methods", {}).get("model_training", {})

    # Correct path: techniques.fit.params / techniques.hyperparameter_tuning.params
    fit_technique: dict[str, Any] = methods_cfg.get("techniques", {}).get("fit", {})
    fit_params: dict[str, Any] = fit_technique.get("params", {})  # fit_best_only, store_all_models

    tune_technique: dict[str, Any] = methods_cfg.get("techniques", {}).get("hyperparameter_tuning", {})
    tune_params: dict[str, Any] = tune_technique.get("params", {})  # strategy, scoring, grids, ...
    # Extraemos el nuevo parámetro
    max_training_rows = fit_params.get("max_training_rows")

    log.debug("[4.3] tune_params keys: %s", list(tune_params.keys()))
    log.debug("[4.3] fit_params: %s", fit_params)
    log.debug("[4.3] grids found: %s", list(tune_params.get("grids", {}).keys()))

    if not enabled(tune_technique):
        log.warning("[4.3] hyperparameter_tuning disabled – no models will be trained")
        write_output_artifacts(
            ctx, step_key=step_key, step_cfg=step_cfg,
            df_train=None, df_test=None,
            extra={"models_trained": False},
        )
        return ctx

    # ------------------------------------------------------------------
    # NEW: Extraction and Validation of Stratification Vector (SOC Ratios)
    # ------------------------------------------------------------------
    stratify_labels: np.ndarray | None = None
    sample_method = tune_params.get("sample_method", "random")

    if sample_method == "stratified":
        log.info("[4.3] 'stratified' sample_method detected. Resolving auxiliary labels for SOC balance.")

        # Intentar extraer del contexto original si fluye aguas arriba
        df_gt_source = ctx.df_train if hasattr(ctx, "df_train") and ctx.df_train is not None else None

        # Enfoque defensivo secundario: si df_train no contiene la columna 'label', leer de disco
        if df_gt_source is not None and "label" in df_gt_source.columns:
            stratify_labels = df_gt_source["label"].to_numpy()
            log.info("[4.3] Stratification vector successfully extracted from memory Context.")
        else:
            # Reutilizamos la función resolutora existente o la ruta estandarizada por CRISP-DM
            phase3_dir = ctx.run_dir / PhaseDir.PHASE3.value
            gt_filename = "3.5.dataset_formatting.save_auxiliary_labels.c_incident_grade_labels_train.parquet"
            gt_path = phase3_dir / gt_filename

            if gt_path.exists():
                df_gt = pd.read_parquet(gt_path)
                if "label" in df_gt.columns:
                    stratify_labels = df_gt["label"].to_numpy()
                    log.info("[4.3] Stratification vector successfully loaded from disk: %s", gt_filename)
                else:
                    log.warning("[4.3] 'label' column missing in auxiliary Parquet file. Falling back.")
            else:
                log.warning("[4.3] Auxiliary labels Parquet not found at %s. Grid search may skip stratification.", gt_path)

        # Validación crítica de consistencia dimensional antes de entrar a la grilla
        if stratify_labels is not None and len(stratify_labels) != X_train.shape[0]:
            error_dim = f"[4.3] Dimension mismatch: X_train has {X_train.shape[0]} rows but stratify_labels has {len(stratify_labels)} elements."
            log.error(error_dim)
            raise RuntimeError(error_dim)


    # ------------------------------------------------------------------
    # Call the pure training function (pass tune_params and stratify_labels)
    # ------------------------------------------------------------------
    try:
        training_result = train_clustering_models(
            X_train=X_train,
            tuning_cfg=tune_params,       # Now contains 'grids'
            fit_cfg=fit_params,           # now contains 'fit_best_only'
            problem_type="clustering",
            stratify_labels=stratify_labels,  #  Inyectado de forma limpia
            max_training_rows=max_training_rows,
        )
    except ValueError as e:
        log.error("[4.3] train_clustering_models failed: %s", e)
        raise

    best_models: dict = training_result["best_models"]
    best_params: dict = training_result["best_params"]
    results: dict = training_result["results"]
    cluster_labels: dict = training_result["cluster_labels"]
    metadata: dict = training_result["metadata"]

    log.info(
        "[4.3] training completed – %d algorithm(s) trained: %s",
        len(best_models), list(best_models.keys()),
    )

    # ------------------------------------------------------------------
    # Persist per‑algorithm hyperparameter search results (JSON)
    # ------------------------------------------------------------------
    # The config defines output paths inside each grid's dictionary.
    # We'll save the combined results for each algorithm.
    grids_cfg: dict[str, Any] = tune_params.get("grids", {})   # correct
    for algo_name, algo_result in results.items():
        algo_grid_cfg = grids_cfg.get(algo_name, {})
        output_path = algo_grid_cfg.get("output", "")
        if output_path:
            save_json(algo_result, phase4_dir / output_path)
            log.info("[4.3] saved grid results for '%s' → %s", algo_name, output_path)

    # ------------------------------------------------------------------
    # Persist best model pickle (both output_artifacts and fit technique)
    # ------------------------------------------------------------------
    output_artifacts = step_cfg.get("output_artifacts", {})

    # a) Save as best_model.pkl (output_artifacts)
    best_model_path = output_artifacts.get("best_model", "")
    if best_model_path:
        import joblib
        joblib.dump(best_models, phase4_dir / best_model_path)
        log.info("[4.3] saved best models to %s (%d model(s))", best_model_path, len(best_models))

    # b) Save as fit.pkl (from fit technique output)
    fit_output = fit_technique.get("output", "")
    if fit_output and best_models:
        joblib.dump(best_models, phase4_dir / fit_output)
        log.info("[4.3] saved best models (via fit technique) to %s", fit_output)

    # ------------------------------------------------------------------
    # Persist hyperparameter search summary (using best_params from result)
    # ------------------------------------------------------------------
    hp_summary_path = output_artifacts.get("hp_search_summary", "")
    if hp_summary_path:
        hp_summary = {
            "best_params": best_params,   # will now be populated
            "n_combinations_evaluated": {k: len(v) for k, v in results.items()},
            "scoring": metadata.get("scoring"),
            "grid_search_sample_size": metadata.get("grid_search_sample_size"),
            "refit": metadata.get("refit"),
            "sample_method": metadata.get("sample_method"),  # Almacenar trazabilidad en metadatos
        }
        save_json(hp_summary, phase4_dir / hp_summary_path)
        log.info("[4.3] saved hp search summary to %s", hp_summary_path)

    # ------------------------------------------------------------------
    # Persist cluster assignments (full train) as Parquet
    # ------------------------------------------------------------------
    cluster_assign_path = output_artifacts.get("cluster_assignments_sample", "")
    if cluster_assign_path and cluster_labels:
        # For MVP, save all labels for the first algorithm; or store dict as separate columns
        # We'll create a DataFrame with one column per algorithm.
        df_assign = pd.DataFrame(cluster_labels)
        df_assign.to_parquet(phase4_dir / cluster_assign_path, index=False)
        log.info("[4.3] saved cluster assignments to %s (shape=%s)",
                 cluster_assign_path, df_assign.shape)
    elif cluster_assign_path:
        log.warning("[4.3] no cluster labels to save – skipping assignments")

    # ------------------------------------------------------------------
    # Persist cluster sizes JSON
    # ------------------------------------------------------------------
    sizes_path = output_artifacts.get("cluster_sizes", "")
    if sizes_path and cluster_labels:
        sizes = {}
        for algo_name, labels in cluster_labels.items():
            unique, counts = np.unique(labels, return_counts=True)
            sizes[algo_name] = dict(zip(map(str, unique), counts.tolist()))
        save_json(sizes, phase4_dir / sizes_path)
        log.info("[4.3] saved cluster sizes to %s", sizes_path)

    # ------------------------------------------------------------------
    # Persist model card (minimal)
    # ------------------------------------------------------------------
    model_card_path = output_artifacts.get("model_card", "")
    if model_card_path:
        model_card = {
            "description": "Clustering models trained via grid search",
            "algorithms": list(best_models.keys()),
            "best_params": best_params,
            "metadata": metadata,
            "n_features": X_train.shape[1],
            "n_samples": X_train.shape[0],
        }
        save_json(model_card, phase4_dir / model_card_path)
        log.info("[4.3] saved model card to %s", model_card_path)

    # ------------------------------------------------------------------
    # (Optional) Cluster centroids – only for KMeans-like models
    # ------------------------------------------------------------------
    centroids_path = output_artifacts.get("cluster_centroids", "")
    if centroids_path and best_models:
        centroids = {}
        for algo_name, model in best_models.items():
            if hasattr(model, "cluster_centers_"):
                centroids[algo_name] = model.cluster_centers_.tolist()
        if centroids:
            # Save as JSON (not Parquet, since centroids are small)
            save_json(centroids, phase4_dir / centroids_path)
            log.info("[4.3] saved centroids to %s", centroids_path)
        else:
            log.debug("[4.3] no models with centroids – skipping centroids file")

    # ------------------------------------------------------------------
    # Store in context artifacts for Phase 4.4/4.5
    # ------------------------------------------------------------------
    ctx.artifacts["best_models"] = best_models
    ctx.artifacts["best_params"] = best_params
    ctx.artifacts["cluster_labels"] = cluster_labels
    ctx.artifacts["training_metadata"] = metadata

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
            "models_trained": len(best_models) > 0,
            "algorithms": list(best_models.keys()),
            "refit": metadata.get("refit"),
            "scoring": metadata.get("scoring"),
            "sample_method": metadata.get("sample_method"),
        },
    )

    log.info("[4.3] done – %d model(s) trained and persisted", len(best_models))
    return ctx

# =============================================================================
# STEP 4.4 — TEST DESIGN GENERATION
# =============================================================================


def run_step_4_4(ctx: RunContext) -> RunContext:
    """Execute CRISP-DM Phase 4.4 – Test Design Generation.

    Reads the step configuration, calls :func:`generate_test_design`,
    persists the evaluation plan as JSON, and writes output artifacts.

    Parameters
    ----------
    ctx : RunContext
        Run context with ``config`` containing
        ``phase4_data_modeling.steps.step_4_4_test_design``.

    Returns
    -------
    RunContext
        Same ``ctx`` (enriched with the evaluation plan in artifacts).
    """
    step_key = StepsPhase.STEP_4_4.value
    step_cfg: dict[str, Any] = dget(
        ctx.config.phases.phase4_data_modeling.steps, step_key, {}
    )
    if not enabled(step_cfg, default=True):
        log.info("[4.4] step disabled – skipping")
        return ctx

    log.info("[4.4] start – run_id=%s", ctx.run_id)
    log.debug("[4.4] step_cfg received: %s", step_cfg)

    # ------------------------------------------------------------------
    # Validate that the step configuration has output path information
    # ------------------------------------------------------------------
    output_artifacts: dict[str, str] = step_cfg.get("output_artifacts", {})
    output_path = output_artifacts.get("evaluation_plan") or step_cfg.get("output")
    if not output_path:
        log.error("[4.4] no output path found in config (output_artifacts.evaluation_plan or output) – cannot persist plan")
        raise RuntimeError(
            "[4.4] Missing output path for evaluation plan. "
            "Ensure step_4_4_test_design.output_artifacts.evaluation_plan or output is defined."
        )

    # ------------------------------------------------------------------
    # Determine output directory
    # ------------------------------------------------------------------
    phase4_dir = ctx.run_dir / PhaseDir.PHASE4.value
    phase4_dir.mkdir(parents=True, exist_ok=True)
    log.debug("[4.4] phase4 output directory: %s", phase4_dir)

    # ------------------------------------------------------------------
    # Call pure model function
    # ------------------------------------------------------------------
    plan = generate_test_design(step_cfg)
    log.info("[4.4] evaluation plan generated – metrics=%s, approach=%s",
             plan.get("metrics"), plan.get("validation_approach"))

    # ------------------------------------------------------------------
    # Persist the evaluation plan as JSON
    # ------------------------------------------------------------------
    save_json(plan, phase4_dir / output_path)
    log.info("[4.4] persisted evaluation plan to %s", output_path)

    # ------------------------------------------------------------------
    # Store in context artifacts for downstream steps (optional)
    # ------------------------------------------------------------------
    ctx.artifacts["evaluation_plan"] = plan

    # ------------------------------------------------------------------
    # Write output artifacts for registry
    # ------------------------------------------------------------------
    write_output_artifacts(
        ctx,
        step_key=step_key,
        step_cfg=step_cfg,
        df_train=None,
        df_test=None,
        extra={"evaluation_plan": plan},
    )

    log.info("[4.4] done – evaluation plan persisted")
    return ctx


# =============================================================================
# STEP 4.5 — MODEL EVALUATION
# =============================================================================


def run_step_4_5(ctx: RunContext) -> RunContext:
    """Execute CRISP-DM Phase 4.5 – Model Evaluation.

    Computes evaluation metrics (silhouette, davies_bouldin, calinski_harabasz,
    adjusted_rand_index, cluster_profiling) on trained clustering models.

    Parameters
    ----------
    ctx : RunContext
        Run context with ``config`` containing
        ``phase4_data_modeling.steps.step_4_5_model_evaluation``.
        Requires ``ctx.artifacts["X_train"]``, ``ctx.artifacts["cluster_labels"]``,
        ``ctx.artifacts["best_models"]``.

    Returns
    -------
    RunContext
        Same ``ctx`` enriched with evaluation results in artifacts.
    """
    step_key = StepsPhase.STEP_4_5.value
    step_cfg: dict[str, Any] = dget(
        ctx.config.phases.phase4_data_modeling.steps, step_key, {}
    )
    if not enabled(step_cfg, default=True):
        log.info("[4.5] step disabled – skipping")
        return ctx

    log.info("[4.5] start – run_id=%s", ctx.run_id)

    # ------------------------------------------------------------------
    # Validate required input data
    # ------------------------------------------------------------------
    X_train = ctx.artifacts.get("X_train")
    cluster_labels = ctx.artifacts.get("cluster_labels")
    best_models = ctx.artifacts.get("best_models")

    if X_train is None:
        log.error("[4.5] X_train not found in ctx.artifacts – aborting")
        raise RuntimeError("[4.5] Missing X_train")
    if cluster_labels is None:
        log.error("[4.5] cluster_labels not found in ctx.artifacts – aborting")
        raise RuntimeError("[4.5] Missing cluster_labels")
    if best_models is None:
        log.warning("[4.5] best_models not found – profiling may be limited")

    log.info("[4.5] input shape=%s, models=%s", X_train.shape, list(cluster_labels.keys()))

    # ------------------------------------------------------------------
    # Extract global random seed (inherited from base_pipeline_config)
    # ------------------------------------------------------------------
    global_seed = dget(ctx.config.runtime, "random_seed", 7)
    log.debug("[4.5] global random_seed=%d", global_seed)

    # ------------------------------------------------------------------
    # Load ground truth labels if adjusted_rand_index is enabled
    # ------------------------------------------------------------------
    techniques: dict[str, Any] = step_cfg.get("methods", {}).get("model_evaluation", {}).get("techniques", {})
    ari_cfg: dict[str, Any] = techniques.get("adjusted_rand_index", {})
    y_true: np.ndarray | None = None
    if ari_cfg.get("enabled", False):
        ground_truth_source: str = ari_cfg.get("params", {}).get("ground_truth_source", "")
        ground_truth_column: str = ari_cfg.get("params", {}).get("ground_truth_column", "IncidentGrade")
        encoding: dict[str, int] = ari_cfg.get("params", {}).get("encoding", {})

        if ground_truth_source:
            # Resolve path relative to run root
            from pathlib import Path
            gt_path = _resolve_ground_truth_path(ctx, ground_truth_source)
            if gt_path and gt_path.exists():
                import pandas as pd
                df_gt = pd.read_parquet(gt_path)
                if ground_truth_column in df_gt.columns:
                    # Apply encoding if provided
                    if encoding:
                        # Map string labels to integers using encoding
                        y_true = df_gt[ground_truth_column].map(encoding).values
                    else:
                        y_true = df_gt[ground_truth_column].values
                    log.info("[4.5] loaded ground truth from %s (n=%d, source=%s)",
                             gt_path, len(y_true), ground_truth_source)
                else:
                    log.warning("[4.5] column '%s' not found in ground truth file – skipping ARI",
                                ground_truth_column)
            else:
                log.warning("[4.5] ground truth file not found: %s – skipping ARI",
                            ground_truth_source)
        else:
            log.warning("[4.5] adjusted_rand_index enabled but no ground_truth_source configured")

    # ------------------------------------------------------------------
    # Ensure output directory exists
    # ------------------------------------------------------------------
    phase4_dir = ctx.run_dir / PhaseDir.PHASE4.value
    phase4_dir.mkdir(parents=True, exist_ok=True)
    log.debug("[4.5] output directory: %s", phase4_dir)

    # ------------------------------------------------------------------
    # Call pure evaluation function
    # ------------------------------------------------------------------
    log.debug("[4.5] techniques config keys: %s", list(techniques.keys()))
    try:
        evaluation_results: dict = evaluate_all_models(
            techniques=techniques,
            X_train=X_train,
            cluster_labels=cluster_labels,
            best_models=best_models or {},
            y_true=y_true,
            global_seed=global_seed,
        )
    except Exception as e:
        log.error("[4.5] evaluate_all_models failed: %s", e)
        raise

    log.info("[4.5] evaluation completed – %d technique(s) processed",
             len(evaluation_results) - 2)  # exclude consolidated keys

    # ------------------------------------------------------------------
    # Persist per‑technique outputs (individual JSON files)
    # ------------------------------------------------------------------
    for technique_name, technique_cfg in techniques.items():
        if not technique_cfg.get("enabled", False):
            continue
        targets: list[str] = technique_cfg.get("targets", [])
        output_config: dict = technique_cfg.get("output", {})
        technique_results: dict = evaluation_results.get(technique_name, {})

        for target in targets:
            output_path: str = output_config.get(target, "")
            if output_path and target in technique_results:
                save_json(technique_results[target], phase4_dir / output_path)
                log.info("[4.5] saved '%s' for '%s' → %s", technique_name, target, output_path)
            elif not output_path:
                log.debug("[4.5] no output path for technique '%s', target '%s'", technique_name, target)

    # ------------------------------------------------------------------
    # Persist output_artifacts (consolidated files)
    # ------------------------------------------------------------------
    output_artifacts: dict[str, str] = step_cfg.get("output_artifacts", {})

    # cluster_labels.parquet – final cluster assignments (reuse from 4.3)
    cluster_labels_path: str = output_artifacts.get("cluster_labels", "")
    if cluster_labels_path and cluster_labels:
        df_labels = pd.DataFrame(cluster_labels)
        df_labels.to_parquet(phase4_dir / cluster_labels_path, index=False)
        log.info("[4.5] saved cluster labels → %s", cluster_labels_path)

    # summary_comparison.json
    summary_path: str = output_artifacts.get("summary_comparison", "")
    if summary_path and "consolidated_summary" in evaluation_results:
        save_json(evaluation_results["consolidated_summary"], phase4_dir / summary_path)
        log.info("[4.5] saved summary comparison → %s", summary_path)

    # consolidated_ari.json – only if ARI was computed
    ari_path: str = output_artifacts.get("consolidated_ari", "")
    if ari_path and "adjusted_rand_index" in evaluation_results:
        save_json(evaluation_results["adjusted_rand_index"], phase4_dir / ari_path)
        log.info("[4.5] saved consolidated ARI → %s", ari_path)

    # consolidated_profiling.json
    profiling_path: str = output_artifacts.get("consolidated_profiling", "")
    if profiling_path and "consolidated_profiling" in evaluation_results:
        save_json(evaluation_results["consolidated_profiling"], phase4_dir / profiling_path)
        log.info("[4.5] saved consolidated profiling → %s", profiling_path)

    # ------------------------------------------------------------------
    # Handle cluster_profiling export of cluster subsets (if enabled)
    # ------------------------------------------------------------------
    cluster_profiling_cfg: dict[str, Any] = techniques.get("cluster_profiling", {})
    if cluster_profiling_cfg.get("enabled", False):
        profiling_params: dict = cluster_profiling_cfg.get("params", {})
        export_subsets: bool = profiling_params.get("export_cluster_subsets", False)
        if export_subsets:
            subset_sample_size: int = profiling_params.get("subset_sample_size", 5000)
            subset_format: str = profiling_params.get("subset_format", "parquet")
            targets_profiling: list[str] = cluster_profiling_cfg.get("targets", [])

            for target in targets_profiling:
                labels = cluster_labels.get(target)
                if labels is None:
                    continue
                unique_labels = np.unique(labels)
                subsets = {}
                for cluster_id in unique_labels:
                    mask = labels == cluster_id
                    cluster_X = X_train[mask]
                    # Subsample if needed
                    if cluster_X.shape[0] > subset_sample_size:
                        rng = np.random.RandomState(global_seed)
                        idx = rng.choice(cluster_X.shape[0], subset_sample_size, replace=False)
                        cluster_X_sub = cluster_X[idx]
                    else:
                        cluster_X_sub = cluster_X
                    subsets[int(cluster_id)] = cluster_X_sub.tolist()  # or save as separate files

                # Save as JSON (or Parquet if you want, but JSON is simpler for inspection)
                subset_output_key = f"cluster_subsets_{target}"
                subset_path: str = output_artifacts.get(subset_output_key, "")
                if subset_path:
                    save_json(subsets, phase4_dir / subset_path)
                    log.info("[4.5] saved cluster subset for '%s' → %s", target, subset_path)

    # ------------------------------------------------------------------
    # Store in context artifacts for downstream steps (Phase 5)
    # ------------------------------------------------------------------
    ctx.artifacts["evaluation_results"] = evaluation_results
    ctx.artifacts["cluster_labels_final"] = cluster_labels  # preserve

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
            "techniques_evaluated": [k for k, v in techniques.items() if v.get("enabled")],
            "models_evaluated": list(cluster_labels.keys()),
            "has_ground_truth": y_true is not None,
            "global_seed_used": global_seed,
        },
    )

    log.info("[4.5] done – evaluation completed successfully")
    return ctx


# ---------------------------------------------------------------------------
# Helper: resolve ground truth file path (relative to run root)
# ---------------------------------------------------------------------------


def _resolve_ground_truth_path(ctx: RunContext, relative_path: str) -> Path | None:
    """Resolve ``ground_truth_source`` relative path to an absolute file path.

    Tries:
    1. Directly under the run root (``ctx.run_dir``).
    2. Under the phase3 output directory (if exists).
    3. Under the phase4 directory (for cases where path references earlier phase).

    Parameters
    ----------
    ctx : RunContext
        Current run context.
    relative_path : str
        Relative path as configured in YAML (e.g., ``"3.5.dataset_formatting.save_auxiliary_labels.c_incident_grade_labels_train.parquet"``).

    Returns
    -------
    Path or None
        Resolved absolute path, or ``None`` if not found.
    """
    from pathlib import Path

    run_root = ctx.run_dir
    candidates = [
        run_root / relative_path,
        run_root / PhaseDir.PHASE3.value / relative_path,
        run_root / PhaseDir.PHASE4.value / relative_path,
    ]

    for candidate in candidates:
        if candidate.exists():
            log.debug("[_resolve_ground_truth_path] found at %s", candidate)
            return candidate.resolve()

    log.warning("[_resolve_ground_truth_path] file not found in any candidate path: %s", relative_path)
    return None