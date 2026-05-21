"""
Dynamic executor for Step 4.4 and 4.5.
Refactored with file shadowing to force dimensional synchronization.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from crispdm.phase.phase4_modeling_runner_phase import run_step_4_4, run_step_4_5
from crispdm.common.context_facade_common import RunContext
from crispdm.configuration.enum_registry_config import PhaseDir
from crispdm.common.path_service_common import find_project_root
from src.crispdm.configuration.yml_repository_config import YmlRepository

# =============================================================================
# 1. Configuration and Run Detection
# =============================================================================
PROJECT_ROOT = find_project_root()
BASE_RUNS_DIR = PROJECT_ROOT / "outputs" / "runs" / "clustering" / "ms_sec_inc_pre"

all_runs = sorted([d for d in BASE_RUNS_DIR.iterdir() if d.is_dir()], key=lambda p: p.stat().st_mtime)
RUN_DIR = all_runs[-1]
RUN_ID = RUN_DIR.name

cfg_node = YmlRepository.load_pipeline_config("clustering")
if "runtime" not in cfg_node:
    cfg_node = OmegaConf.merge(cfg_node, OmegaConf.create({"runtime": {"random_seed": 7}}))

print(f"📁 Root: {PROJECT_ROOT}")
print(f"🔑 Detected run: {RUN_ID}")

# =============================================================================
# 2. Strict Synchronization and Shadowing (Forced on disk)
# =============================================================================
# Source file paths
path_x = RUN_DIR / PhaseDir.PHASE3.value / "3.5.data_formatting.a_train_prepared.parquet"
path_y = RUN_DIR / PhaseDir.PHASE3.value / "3.5.dataset_formatting.save_auxiliary_labels.c_incident_grade_labels_train.parquet"
path_lbl = RUN_DIR / PhaseDir.PHASE4.value / "4.3.model_training.cluster_assignments_sample.parquet"

# Determine limit
limit = cfg_node.phases.phase4_data_modeling.steps.step_4_3_model_training.methods.model_training.techniques.fit.params.get("max_training_rows", 30000)

print(f"⚙️ Synchronizing artifacts to limit: {limit}")

# Apply Shadowing: Overwrite files on disk if they exceed the limit
for path in [path_x, path_y, path_lbl]:
    if path.exists():
        df = pd.read_parquet(path)
        if len(df) > limit:
            print(f"⚠️ Applying Shadowing on: {path.name} (truncating to {limit})")
            df.iloc[:limit].to_parquet(path)

# Load artifacts for context
artifacts = {
    "X_train": pd.read_parquet(path_x).values.astype(np.float32),
    "cluster_labels": {"kmeans_n2": pd.read_parquet(path_lbl)["kmeans"].values},
    "best_models": {"kmeans_n2": joblib.load(RUN_DIR / PhaseDir.PHASE4.value / "4.3.model_training.best_model.pkl")["kmeans"]},
    "y_true": pd.read_parquet(path_y)["label"].values.astype(np.int32) if path_y.exists() else None
}

# Ensure y_true is available for the evaluator
if artifacts.get("y_true") is None:
    print("⚠️ Attention! y_true is None. Ensuring loading...")
    gt_path = RUN_DIR / PhaseDir.PHASE3.value / "3.5.dataset_formatting.save_auxiliary_labels.c_incident_grade_labels_train.parquet"
    if gt_path.exists():
        artifacts["y_true"] = pd.read_parquet(gt_path)["label"].values.astype(np.int32)[:limit]

# =============================================================================
# 3. Context and Execution
# =============================================================================
ctx = RunContext(config=cfg_node, run_dir=RUN_DIR, run_id=RUN_ID, dataset_key="ms_sec_inc_pre")
ctx.artifacts = artifacts

# Dynamic configuration adjustment
eval_techs = cfg_node.phases.phase4_data_modeling.steps.step_4_5_model_evaluation.methods.model_evaluation.techniques
for tech_name, tech_cfg in eval_techs.items():
    tech_cfg.targets = ["kmeans_n2"]
    tech_cfg.output = {"kmeans_n2": f"4.5.model_evaluation.{tech_name}.kmeans_n2.json"}

print("🚀 Starting Pipeline 4.4 -> 4.5...")
ctx = run_step_4_4(ctx)
ctx = run_step_4_5(ctx)

print(f"\n✅ Pipeline completed successfully. Results in: {RUN_DIR / PhaseDir.PHASE4.value}")