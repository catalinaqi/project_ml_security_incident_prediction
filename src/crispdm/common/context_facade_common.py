from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import pandas as pd
from omegaconf import DictConfig

from src.crispdm.common.logging_adapter_common import get_logger
from src.crispdm.common.path_service_common import find_project_root, resolve_path
from src.crispdm.configuration.enum_registry_config import PhaseDir

log = get_logger(__name__)

RUN_ID_FORMAT = "%Y%m%d_%H%M%S"
RUNS_SUBDIR = "runs"


def make_run_id(ts: Optional[datetime] = None) -> str:
    if ts is None:
        ts = datetime.now()
    run_id = ts.strftime(RUN_ID_FORMAT)
    log.debug(f"Generated run_id: {run_id}")
    return run_id


def make_run_dir(
        output_root: str | Path,
        task: str,
        dataset_key: str,
        run_id: str
) -> Path:

    log.info(f"Creating run directory: task={task}, dataset={dataset_key}, run_id={run_id}")

    # Resolve output_root to absolute path
    if isinstance(output_root, str):
        output_root = resolve_path(output_root)

    # Create directory structure: output_root/runs/task/dataset_key/run_id
    run_dir = output_root / RUNS_SUBDIR / task / dataset_key / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Run directory created: {run_dir}")
    return run_dir


@dataclass
class RunContext:
    config: DictConfig
    run_dir: Path
    run_id: str
    dataset_key: str

    # DataFrames
    df_train: Optional[pd.DataFrame] = field(default=None, repr=False)
    df_test: Optional[pd.DataFrame] = field(default=None, repr=False)

    # Tracking
    artifacts: dict[str, Path] = field(default_factory=dict)
    phase_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def task(self) -> str:
        task_value = self.config.metadata.get("pipeline_key", {}).get("task")
        if not task_value:
            log.warning("Task not found in config metadata")
            return "unknown"
        return task_value

    def get_phase_dir(self, phase_name: str) -> Path:
        phase_dir = self.run_dir / phase_name
        phase_dir.mkdir(parents=True, exist_ok=True)
        log.debug(f"Phase directory: {phase_dir}")
        return phase_dir

    @property
    def phase2_dir(self) -> Path:
        return self.get_phase_dir(PhaseDir.PHASE2.value)

    @property
    def phase3_dir(self) -> Path:
        return self.get_phase_dir(PhaseDir.PHASE3.value)

    @property
    def phase4_dir(self) -> Path:
        return self.get_phase_dir(PhaseDir.PHASE4.value)

    @property
    def phase5_dir(self) -> Path:
        return self.get_phase_dir(PhaseDir.PHASE5.value)

    def set_train_data(self, df: pd.DataFrame) -> None:
        self.df_train = df
        log.info(f"Train data set: shape={df.shape}")

    def set_test_data(self, df: pd.DataFrame) -> None:
        self.df_test = df
        log.info(f"Test data set: shape={df.shape}")

    def register_artifact(self, label: str, path: Path) -> None:
        self.artifacts[label] = path
        log.debug(f"Artifact registered: {label} -> {path}")

    def register_phase_result(self, phase_name: str, result: dict[str, Any]) -> None:
        if phase_name in self.phase_results:
            self.phase_results[phase_name].update(result)
            log.debug(f"Phase result updated: {phase_name}")
        else:
            self.phase_results[phase_name] = dict(result)
            log.debug(f"Phase result registered: {phase_name}")

    def collect_error(self, error: str) -> None:
        self.errors.append(error)
        log.warning(f"Error collected: {error}")

    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task": self.task,
            "dataset_key": self.dataset_key,
            "run_dir": str(self.run_dir),
            "train_shape": list(self.df_train.shape) if self.df_train is not None else None,
            "test_shape": list(self.df_test.shape) if self.df_test is not None else None,
            "n_artifacts": len(self.artifacts),
            "phases_completed": list(self.phase_results.keys()),
            "n_errors": len(self.errors),
        }

    def log_summary(self) -> None:
        summary = self.summary()
        log.info(f"Run summary: {summary}")


def create_run_context(
        config: DictConfig,
        dataset_key: str,
        run_id: Optional[str] = None
) -> RunContext:

    log.info(f"Creating run context for dataset: {dataset_key}")

    # Generate run_id if not provided
    if run_id is None:
        run_id = make_run_id()

    # Get task from config
    task = config.metadata.get("pipeline_key", {}).get("task", "unknown")

    # Get output_root from config
    output_root = config.common_base_config.runtime.output_root

    # Create run directory
    run_dir = make_run_dir(output_root, task, dataset_key, run_id)

    # Create context
    context = RunContext(
        config=config,
        run_dir=run_dir,
        run_id=run_id,
        dataset_key=dataset_key
    )

    log.info(f"Run context created: run_id={run_id}, run_dir={run_dir}")
    return context