"""
Robust execution script for Step 4.3.
- Uses a constant for the Step key (avoids IndexError).
- Deduces paths dynamically (eliminates hardcoding of dates).
"""
from __future__ import annotations

import logging
from pathlib import Path
import numpy as np
import pandas as pd
from types import SimpleNamespace

from crispdm.configuration.enum_registry_config import PhaseDir
from crispdm.common.path_service_common import find_project_root
from src.crispdm.configuration.yml_repository_config import YmlRepository
from src.crispdm.phase.phase4_modeling_runner_phase import run_step_4_3

STEP_4_3_KEY = "step_4_3_model_training"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("run_4_3_final")

PROJECT_ROOT = find_project_root()
BASE_RUNS_DIR = PROJECT_ROOT / "outputs" / "runs" / "clustering" / "ms_sec_inc_pre"

# 1. Dynamic deduction of the latest run
all_runs = sorted([d for d in BASE_RUNS_DIR.iterdir() if d.is_dir()], key=lambda p: p.stat().st_mtime)
if not all_runs:
    log.error("No execution folders found."); exit(1)

RUN_DIR = all_runs[-1]
log.info("🎯 Detected run: %s", RUN_DIR.name)

# 2. Load X_train as float32
x_train_path = RUN_DIR / PhaseDir.PHASE3.value / "3.5.data_formatting.a_train_prepared.parquet"
X_train_matrix = pd.read_parquet(x_train_path).to_numpy().astype(np.float32)
log.info("📊 X_train loaded. Shape: %s dtype: %s", X_train_matrix.shape, X_train_matrix.dtype)

# CAMBIO 1 — cargar y_true para stratified subsampling
y_path = RUN_DIR / PhaseDir.PHASE3.value / "3.5.dataset_formatting.save_auxiliary_labels.c_incident_grade_labels_train.parquet"
if y_path.exists():
    y_df   = pd.read_parquet(y_path)
    col    = "label" if "label" in y_df.columns else y_df.columns[0]
    y_true = y_df[col].values.astype(np.int32)
    log.info("✅ y_true loaded: %d rows, col='%s'", len(y_true), col)
else:
    y_true = None
    log.warning("⚠️ y_true not found — stratified subsampling will fall back to random")

# 3. Configuration
real_config = YmlRepository.load_pipeline_config("clustering")
try:
    step_cfg = real_config.phases.phase4_data_modeling.steps[STEP_4_3_KEY]
    log.info("✅ Config block found: %s", STEP_4_3_KEY)
except KeyError:
    log.error("❌ Key '%s' not found in YAML.", STEP_4_3_KEY); exit(1)

# 4. Context — CAMBIO 2: incluir y_true en artifacts
ctx = SimpleNamespace(
    run_id=RUN_DIR.name,
    run_dir=RUN_DIR,
    config=real_config,
    artifacts={
        "X_train": X_train_matrix,
        "y_true":  y_true,          # para stratified subsampling en el trainer
    }
)

# 5. Execution
try:
    run_step_4_3(ctx)
    log.info("✅ Step 4.3 completed successfully!")
except Exception as err:
    log.error("❌ Error: %s", err, exc_info=True)