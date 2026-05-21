"""
Executor for Steps 4.4 and 4.5.
Evaluates kmeans_n2 and kmeans_n3. No disk shadowing.
"""
from __future__ import annotations

import logging
import joblib
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from crispdm.phase.phase4_modeling_runner_phase import run_step_4_4, run_step_4_5
from crispdm.common.context_facade_common import RunContext
from crispdm.configuration.enum_registry_config import PhaseDir
from crispdm.common.path_service_common import find_project_root
from src.crispdm.configuration.yml_repository_config import YmlRepository

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("run_4_4_4_5")

# =============================================================================
# 1. Run Detection
# =============================================================================
PROJECT_ROOT = find_project_root()
BASE_RUNS_DIR = PROJECT_ROOT / "outputs" / "runs" / "clustering" / "ms_sec_inc_pre"

all_runs = sorted([d for d in BASE_RUNS_DIR.iterdir() if d.is_dir()], key=lambda p: p.stat().st_mtime)
if not all_runs:
    log.error("No execution folders found."); exit(1)

RUN_DIR = all_runs[-1]
RUN_ID  = RUN_DIR.name
log.info("🎯 Detected run: %s", RUN_ID)

# =============================================================================
# 2. Config
# =============================================================================
cfg_node = YmlRepository.load_pipeline_config("clustering")
if "runtime" not in cfg_node:
    cfg_node = OmegaConf.merge(cfg_node, OmegaConf.create({"runtime": {"random_seed": 7}}))

limit = cfg_node.phases.phase4_data_modeling.steps \
    .step_4_3_model_training.methods.model_training \
    .techniques.fit.params.get("max_training_rows", 30000)
log.info("⚙️ max_training_rows: %d", limit)

# =============================================================================
# 3. Load artifacts IN MEMORY — no disk overwrite
# =============================================================================
path_x   = RUN_DIR / PhaseDir.PHASE3.value / "3.5.data_formatting.a_train_prepared.parquet"
path_y   = RUN_DIR / PhaseDir.PHASE3.value / "3.5.dataset_formatting.save_auxiliary_labels.c_incident_grade_labels_train.parquet"
path_lbl = RUN_DIR / PhaseDir.PHASE4.value / "4.3.model_training.cluster_assignments_sample.parquet"
path_mdl = RUN_DIR / PhaseDir.PHASE4.value / "4.3.model_training.best_model.pkl"

# X_train — truncate in memory only
X_full  = pd.read_parquet(path_x).values.astype(np.float32)
X_train = X_full[:limit] if limit and limit < len(X_full) else X_full
log.info("📊 X_train shape: %s", X_train.shape)

# y_true
if path_y.exists():
    y_df   = pd.read_parquet(path_y)
    col    = "label" if "label" in y_df.columns else y_df.columns[0]
    y_true = y_df[col].values.astype(np.int32)[:limit]
    log.info("✅ y_true loaded: %d rows, col='%s'", len(y_true), col)
else:
    y_true = None
    log.warning("⚠️ y_true not found — ARI will be skipped")

# cluster_labels — kmeans_n2 y kmeans_n3 desde cluster_assignments_sample
if not path_lbl.exists():
    log.error("❌ cluster_assignments not found: %s", path_lbl); exit(1)

lbl_df = pd.read_parquet(path_lbl).iloc[:limit]
log.info("📋 cluster_assignments columns: %s", list(lbl_df.columns))

cluster_labels = {}
for variant in ["kmeans_n2", "kmeans_n3"]:
    if variant in lbl_df.columns:
        cluster_labels[variant] = lbl_df[variant].values
        log.info("✅ cluster labels loaded for '%s'", variant)
    else:
        log.warning("⚠️ column '%s' not in cluster_assignments — variant will be skipped", variant)

if not cluster_labels:
    log.error("❌ No valid cluster label variants found. Re-run 4.3 with named_variants config."); exit(1)

# best_models — kmeans_n2 y kmeans_n3 desde best_model.pkl
if not path_mdl.exists():
    log.error("❌ best_model.pkl not found: %s", path_mdl); exit(1)

models_dict = joblib.load(path_mdl)
log.info("📦 best_model.pkl keys: %s", list(models_dict.keys()))

best_models = {}
for variant in ["kmeans_n2", "kmeans_n3"]:
    if variant in models_dict:
        best_models[variant] = models_dict[variant]
        log.info("✅ model loaded for '%s'", variant)
    else:
        log.warning("⚠️ model key '%s' not in pkl — variant will be skipped", variant)

# =============================================================================
# 4. Context — sin override de targets, el YAML controla
# =============================================================================
ctx = RunContext(
    config=cfg_node,
    run_dir=RUN_DIR,
    run_id=RUN_ID,
    dataset_key="ms_sec_inc_pre"
)
ctx.artifacts = {
    "X_train":       X_train,
    "y_true":        y_true,
    "cluster_labels": cluster_labels,
    "best_models":   best_models,
}

# =============================================================================
# 5. Execute 4.4 → 4.5
# =============================================================================
log.info("🚀 Starting 4.4 → 4.5...")
try:
    ctx = run_step_4_4(ctx)
    ctx = run_step_4_5(ctx)
    log.info("✅ Done. Results in: %s", RUN_DIR / PhaseDir.PHASE4.value)
except Exception as e:
    log.error("❌ Failed: %s", e, exc_info=True); exit(1)