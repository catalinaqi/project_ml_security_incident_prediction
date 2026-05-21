"""
Robust execution script for Step 4.3.
- Uses a constant for the Step key (avoids IndexError).
- Deduces paths dynamically (eliminates hardcoding of dates).
"""
from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd
from types import SimpleNamespace

from crispdm.configuration.enum_registry_config import PhaseDir
from crispdm.common.path_service_common import find_project_root
from src.crispdm.configuration.yml_repository_config import YmlRepository
from src.crispdm.phase.phase4_modeling_runner_phase import run_step_4_3

# =============================================================================
# CONFIGURATION CONSTANTS (Only "hardcoded" to avoid key errors)
# =============================================================================
STEP_4_3_KEY = "step_4_3_model_training"

# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("run_4_3_final")

PROJECT_ROOT = find_project_root()
BASE_RUNS_DIR = PROJECT_ROOT / "outputs" / "runs" / "clustering" / "ms_sec_inc_pre"

# 1. Dynamic deduction of the latest run (No date hardcoding)
all_runs = sorted([d for d in BASE_RUNS_DIR.iterdir() if d.is_dir()], key=lambda p: p.stat().st_mtime)
if not all_runs:
    log.error("No execution folders found."); exit(1)

RUN_DIR = all_runs[-1]
log.info("🎯 Detected run: %s", RUN_DIR.name)

# 2. Dynamic data loading
x_train_path = RUN_DIR / PhaseDir.PHASE3.value / "3.5.data_formatting.a_train_prepared.parquet"
X_train_matrix = pd.read_parquet(x_train_path).to_numpy()
log.info("📊 Matrix loaded. Shape: %s", X_train_matrix.shape)

# 3. Configuration
real_config = YmlRepository.load_pipeline_config("clustering")
# Safe and direct access using our constant
try:
    step_cfg = real_config.phases.phase4_data_modeling.steps[STEP_4_3_KEY]
    log.info("✅ Successful access to block: %s", STEP_4_3_KEY)
except KeyError:
    log.error("❌ Key '%s' does not exist in YAML. Check your configuration file.", STEP_4_3_KEY)
    exit(1)

# 4. Data injection
ctx = SimpleNamespace(
    run_id=RUN_DIR.name,
    run_dir=RUN_DIR,
    config=real_config,
    artifacts={"X_train": X_train_matrix}
)

# 5. Execution
try:
    run_step_4_3(ctx)
    log.info("✅ Execution successful!")
except Exception as err:
    log.error("❌ Error during execution: %s", err, exc_info=True)