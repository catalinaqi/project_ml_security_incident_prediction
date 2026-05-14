from __future__ import annotations
from pathlib import Path
from typing import Optional
from omegaconf import OmegaConf, DictConfig

from src.crispdm.common.logging_adapter_common import get_logger
from src.crispdm.common.path_service_common import resolve_path, find_project_root

log = get_logger(__name__)


class YmlRepository:
    _base_cache: Optional[DictConfig] = None
    _project_root: Optional[Path] = None

    @classmethod
    def _get_config_root(cls) -> Path:
        if cls._project_root is None:
            cls._project_root = find_project_root()
        return cls._project_root / "config"

    @classmethod
    def load_dataset_config(cls) -> DictConfig:
        path = cls._get_config_root() / "datasets" / "dataset_config.yml"
        log.info(f"Loading dataset config from {path}")

        if not path.exists():
            log.error(f"Dataset config not found: {path}")
            raise FileNotFoundError(f"Dataset config not found: {path}")

        return OmegaConf.load(path)

    @classmethod
    def get_dataset_by_key(cls, key: str) -> DictConfig:
        cfg = cls.load_dataset_config()

        if key not in cfg.datasets:
            available = list(cfg.datasets.keys())
            log.error(f"Dataset key '{key}' not found. Available: {available}")
            raise KeyError(f"Dataset '{key}' not found in dataset_config.yml")

        log.info(f"Retrieved dataset config for key='{key}'")
        return cfg.datasets[key]

    @classmethod
    def load_base_pipeline_config(cls) -> DictConfig:
        if cls._base_cache is None:
            path = cls._get_config_root() / "pipelines" / "base_pipeline_config.yml"
            log.info(f"Loading base pipeline config from {path}")

            if not path.exists():
                log.error(f"Base pipeline config not found: {path}")
                raise FileNotFoundError(f"Base pipeline config not found: {path}")

            cls._base_cache = OmegaConf.load(path)
            log.info("Base pipeline config cached successfully")

        return cls._base_cache

    @classmethod
    def load_pipeline_config(cls, pipeline_name: str) -> DictConfig:
        log.info(f"Loading pipeline config for '{pipeline_name}'")

        base_cfg = cls.load_base_pipeline_config()
        pipeline_path = cls._get_config_root() / "pipelines" / f"{pipeline_name}_pipeline_config.yml"

        if not pipeline_path.exists():
            log.error(f"Pipeline config not found: {pipeline_path}")
            raise FileNotFoundError(f"Pipeline config not found: {pipeline_path}")

        specific_cfg = OmegaConf.load(pipeline_path)
        log.info(f"Loaded specific pipeline config from {pipeline_path}")

        pipeline_key = f"{pipeline_name}_pipeline_config"

        if pipeline_key not in specific_cfg:
            log.error(f"Expected key '{pipeline_key}' not found in {pipeline_path}")
            raise KeyError(f"Key '{pipeline_key}' not found in pipeline config")

        pipeline_data = specific_cfg[pipeline_key]

        merged = OmegaConf.create({
            "metadata": pipeline_data.get("metadata", {}),
            "common_base_config": base_cfg.pipeline_base.common_base_config,
            "profiles": base_cfg.pipeline_base.profiles,
            "phases": {}
        })

        if "common_phases_config" in base_cfg.pipeline_base:
            merged.phases.update(base_cfg.pipeline_base.common_phases_config.phases)

        if "phases" in pipeline_data:
            merged.phases.update(pipeline_data.phases)

        if "runtime" in pipeline_data:
            merged.common_base_config.runtime.update(pipeline_data.runtime)
            log.info(f"Runtime overrides applied from {pipeline_name}")

        log.info(f"Pipeline config merged successfully for '{pipeline_name}'")
        return merged

    @classmethod
    def get_active_profile(cls) -> str:
        path = cls._get_config_root() / "pipelines" / "active_profile.yml"
        log.info(f"Reading active profile from {path}")

        if not path.exists():
            log.warning(f"Active profile not found: {path}, defaulting to 'dev'")
            return "dev"

        profile_cfg = OmegaConf.load(path)
        active = profile_cfg.get("active_profile", "dev")
        log.info(f"Active profile: {active}")
        return active