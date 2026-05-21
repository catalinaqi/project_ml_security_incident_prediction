"""
Isolated execution and validation script for Step 3.5.
Flows in memory from Phase 2 using the real YmlRepository of the project
and resolving paths from PROJECT_ROOT to avoid misalignments.

Run from the project root:
  poetry run python run_3_5.py
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path

# Official services and Enums of your infrastructure
from crispdm.configuration.enum_registry_config import PhaseDir
from crispdm.common.path_service_common import find_project_root
from crispdm.common.context_facade_common import RunContext

# Official runners of Phase 3
from crispdm.phase.phase3_preparation_runner_phase import (
    run_step_3_1,
    run_step_3_2,
    run_step_3_3,
    run_step_3_5,
)

# LOAD REAL CONFIGURATION
from crispdm.configuration.yml_repository_config import YmlRepository

# =============================================================================
# DYNAMIC PATH CONFIGURATION USING PROJECT_ROOT
# =============================================================================
PROJECT_ROOT = find_project_root()
print(f"📁 Project root detected: {PROJECT_ROOT}")

# Base directory where you want to save the NEW results of Phase 3
RUN_DIR = PROJECT_ROOT / "outputs" / "runs" / "clustering" / "ms_sec_inc_pre" / "20260521_095921"

# REAL INPUT PATH: Search in the May 20 run where the real production file resides
F2_INPUT_PATH = (
        PROJECT_ROOT
        / "outputs"
        / "runs"
        / "clustering"
        / "ms_sec_inc_pre"
        / "20260520_204354"  # <-- Source folder with real production parquets
        / PhaseDir.PHASE2.value
        / "2.1.data_acquisition.prod_200000_stratified_train.parquet"
).resolve()

# Expected output file that your modified Step 3.5 will generate (In the May 21 folder)
GT_PATH = (
        RUN_DIR
        / PhaseDir.PHASE3.value
        / "3.5.dataset_formatting.save_auxiliary_labels.c_incident_grade_labels_train.parquet"
)

print("=" * 80)
print("🧪 SEQUENTIAL EXECUTION LOGS IN MEMORY (STEP 3.1 TO 3.5)")
print("=" * 80)

# -----------------------------------------------------------------------------
# 1. Validation of Critical Inputs
# -----------------------------------------------------------------------------
if not F2_INPUT_PATH.exists():
    raise FileNotFoundError(f"❌ Production Parquet file not found on disk: {F2_INPUT_PATH}")

print("📥 Loading merged configuration via YmlRepository...")
cfg = YmlRepository.load_pipeline_config("clustering")

print(f"📥 Reading source dataset from Phase 2 from:\n    {F2_INPUT_PATH}")
df_raw = pd.read_parquet(F2_INPUT_PATH)
print(f"   📊 Initial dataset loaded successfully. Shape: {df_raw.shape}")

# -----------------------------------------------------------------------------
# 2. Initialization of Official RunContext
# -----------------------------------------------------------------------------
ctx = RunContext(
    config=cfg,
    run_dir=RUN_DIR,
    df_train=df_raw,
    df_test=None,
    run_id=RUN_DIR.name,
    dataset_key="ms_sec_inc_pre"  # <-- ADD THIS LINE HERE!
)

# -----------------------------------------------------------------------------
# 3. Chain Processing (Upstream to feed 3.5)
# -----------------------------------------------------------------------------
print("\n⚡ [Executing Step 3.1] Data Selection...")
ctx = run_step_3_1(ctx)

print("⚡ [Executing Step 3.2] Data Cleaning...")
ctx = run_step_3_2(ctx)

print("⚡ [Executing Step 3.3] Feature Transformation...")
ctx = run_step_3_3(ctx)

# -----------------------------------------------------------------------------
# 4. The Moment of Truth: Isolated Step 3.5
# -----------------------------------------------------------------------------
print("\n🎯 [Executing Step 3.5] Final Formatting with Modified Code...")
try:
    ctx = run_step_3_5(ctx)
    print("  ✅ Step 3.5 code executed without exceptions.")
except Exception as e:
    print(f"  ❌ Step 3.5 internal logic failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# -----------------------------------------------------------------------------
# 5. Post-Execution Diagnosis of Output Artifact
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("🔍 VERIFICATION OF MODIFIED ARTIFACT ON DISK")
print("=" * 80)

print(f"[C] Analyzing Ground Truth Parquet generated in:\n    {GT_PATH}")
if GT_PATH.exists():
    df_gt = pd.read_parquet(GT_PATH)
    print(f"  ✅ File rewritten on disk.")
    print(f"  📋 Detected columns: {df_gt.columns.tolist()}")
    print(f"  📊 Final shape of Auxiliary DataFrame: {df_gt.shape}")

    if "IncidentGrade" in df_gt.columns and "label" in df_gt.columns:
        print("\n🎉 ✨ TOTAL SUCCESS! The file now contains the ideal hybrid structure.")
        print("\n👀 Sample of the first 2 records:")
        print(df_gt.head(2).to_string(index=False))
    else:
        print("\n⚠️ Warning: The file was generated but does not contain both columns at the same time.")
else:
    print(f"  ❌ Error: The processing finished but the physical file did not materialize on disk.")

print("\n" + "=" * 80)
print("🏁 End of recovery test.")
print("=" * 80)