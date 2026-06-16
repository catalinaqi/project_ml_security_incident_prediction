from __future__ import annotations
from typing import Optional, Dict, List, Any, Union
from pydantic import BaseModel, Field, field_validator

from crispdm.common.logging_adapter_common import get_logger
from crispdm.configuration.enum_registry_config import PhaseDir, StepsPhase

log = get_logger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# DATASET CONFIG SCHEMAS
# ═════════════════════════════════════════════════════════════════════════════

class CSVParams(BaseModel):
    sep: str = ","
    encoding: str = "utf-8"
    decimal: str = "."
    low_memory: bool = False


class SourceConfig(BaseModel):
    url: str
    download_executor: str


class PathsConfig(BaseModel):
    train: str
    test: str


class DatasetConfig(BaseModel):
    name: str
    version: str
    description: str
    source: SourceConfig
    paths: PathsConfig
    csv_params: CSVParams


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE CONFIG SCHEMAS
# ═════════════════════════════════════════════════════════════════════════════

class RuntimeConfig(BaseModel):
    random_seed: int = 7
    output_root: str = "outputs"
    overwrite_artifacts: bool = True
    log_level: str = "INFO"


class OutputPolicyConfig(BaseModel):
    save_all_as_png: bool = True
    save_all_tables_as_png: bool = True
    dpi: int = 150
    format: str = "parquet"
    compression: str = "snappy"


class CommonBaseConfig(BaseModel):
    methodology: str
    version: str
    runtime: RuntimeConfig
    output_policy: OutputPolicyConfig


class ProfileConfig(BaseModel):
    enabled: bool
    description: str
    mode: str
    sample_rows: int
    stratify_column: Optional[str] = None


class TechniqueConfig(BaseModel):
    enabled: bool
    description: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)

    #output: Optional[str] = None

    # ──── MODIFICA ESTA LÍNEA ────
    output: Optional[Union[Dict[str, str], str]] = None


    @field_validator("params", mode="before")
    @classmethod
    def validate_params(cls, v):
        if v is None:
            return {}
        return v


class MethodConfig(BaseModel):
    enabled: bool = True
    description: Optional[str] = None
    techniques: Dict[str, TechniqueConfig] = Field(default_factory=dict)


class OutputArtifactsConfig(BaseModel):
    class Config:
        extra = "allow"


class StepConfig(BaseModel):
    enabled: bool
    description: Optional[str] = None
    methods: Dict[str, MethodConfig] = Field(default_factory=dict)
    output_artifacts: Optional[Union[Dict[str, str], OutputArtifactsConfig]] = None


class ReadStrategyConfig(BaseModel):
    source_type: Optional[str] = None
    train_path: Optional[str] = None
    test_path: Optional[str] = None
    sample_method: Optional[str] = None
    stratify_column: Optional[str] = None
    random_state: Optional[int] = None
    chunksize: Optional[int] = None
    add_source_column: Optional[bool] = None
    combine_before_sampling: Optional[bool] = None
    dtype_optimization: Optional[bool] = None
    mode: Optional[str] = None
    input_source: Optional[str] = None


class DatasetInputConfig(BaseModel):
    source_type: Optional[str] = None
    train_path: Optional[str] = None
    test_path: Optional[str] = None


class PhaseConfig(BaseModel):
    enabled: bool
    objective: Optional[str] = None
    design_principle: Optional[str] = None
    dataset_input: Optional[DatasetInputConfig] = None
    read_strategy: Optional[ReadStrategyConfig] = None
    steps: Dict[str, StepConfig] = Field(default_factory=dict)
    critical_warnings: Optional[Dict[str, Any]] = None
    output_artifacts: Optional[Dict[str, str]] = None


class MetadataConfig(BaseModel):
    project: Optional[str] = None
    purpose: Optional[str] = None
    methodology: Optional[str] = None
    version: Optional[str] = None
    date: Optional[str] = None
    notes: Optional[str] = None
    pipeline_key: Optional[Dict[str, Any]] = None


class PipelineConfig(BaseModel):
    metadata: MetadataConfig
    common_base_config: CommonBaseConfig
    profiles: Dict[str, ProfileConfig]
    phases: Dict[str, PhaseConfig]

    @field_validator("phases", mode="before")
    @classmethod
    def validate_phases(cls, v):
        if v is None:
            log.warning("No phases defined in pipeline config")
            return {}

        valid_phases = {phase.value for phase in PhaseDir}
        invalid_phases = set(v.keys()) - valid_phases

        if invalid_phases:
            log.error(f"Invalid phase names detected: {invalid_phases}")
            log.debug(f"Valid phases: {valid_phases}")
            raise ValueError(f"Invalid phases: {invalid_phases}")

        log.debug(f"Phase validation passed for: {list(v.keys())}")
        return v

    def get_phase(self, phase_name: str) -> Optional[PhaseConfig]:
        if phase_name not in self.phases:
            log.warning(f"Phase '{phase_name}' not found in config")
            return None

        log.debug(f"Retrieved phase config for: {phase_name}")
        return self.phases.get(phase_name)

    def get_enabled_phases(self) -> List[str]:
        enabled = [name for name, phase in self.phases.items() if phase.enabled]
        log.info(f"Enabled phases: {enabled}")
        return enabled

    def get_active_profile(self, profile_name: str) -> Optional[ProfileConfig]:
        if profile_name not in self.profiles:
            log.error(f"Profile '{profile_name}' not found in config")
            return None

        profile = self.profiles.get(profile_name)
        if not profile.enabled:
            log.warning(f"Profile '{profile_name}' is disabled")

        log.info(f"Active profile: {profile_name}")
        return profile


# ═════════════════════════════════════════════════════════════════════════════
# FACTORY METHODS
# ═════════════════════════════════════════════════════════════════════════════

class DTOFactory:

    @staticmethod
    def create_dataset_config(data: Dict[str, Any]) -> DatasetConfig:
        log.debug("Starting dataset config DTO creation")
        try:
            config = DatasetConfig(**data)
            log.info(f"Dataset config created: {config.name} v{config.version}")
            return config
        except Exception as e:
            log.error(f"Failed to create dataset config: {e}")
            raise

    @staticmethod
    def create_pipeline_config(data: Dict[str, Any]) -> PipelineConfig:
        log.debug("Starting pipeline config DTO creation")
        try:
            config = PipelineConfig(**data)
            log.info(f"Pipeline config created with {len(config.phases)} phases")
            return config
        except Exception as e:
            log.error(f"Failed to create pipeline config: {e}")
            raise

    @staticmethod
    def validate_dataset(data: Dict[str, Any]) -> bool:
        log.debug("Validating dataset config structure")
        try:
            DatasetConfig(**data)
            log.info("Dataset config validation passed")
            return True
        except Exception as e:
            log.error(f"Dataset config validation failed: {e}")
            return False

    @staticmethod
    def validate_pipeline(data: Dict[str, Any]) -> bool:
        log.debug("Validating pipeline config structure")
        try:
            PipelineConfig(**data)
            log.info("Pipeline config validation passed")
            return True
        except Exception as e:
            log.error(f"Pipeline config validation failed: {e}")
            return False