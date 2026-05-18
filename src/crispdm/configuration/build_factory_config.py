from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
from omegaconf import DictConfig, OmegaConf

from src.crispdm.common.logging_adapter_common import get_logger
from src.crispdm.configuration.yml_repository_config import YmlRepository
from src.crispdm.configuration.pipeline_task_dto_config import PipelineConfig

log = get_logger(__name__)

DEFAULT_OUTPUT_ROOT = "outputs"


@dataclass(frozen=True)
class BuiltConfig:
    config: DictConfig
    dataset_key: Optional[str] = None


class ConfigBuilder:

    @staticmethod
    def build_pipeline_config(
            pipeline_name: str,
            dataset_key: str,
            notebook_vars: Optional[Dict[str, Any]] = None
    ) -> BuiltConfig:

        log.info(f"Building pipeline config: pipeline={pipeline_name}, dataset={dataset_key}")

        notebook_vars = notebook_vars or {}

        # Step 1: Load pipeline config (already merged with base)
        pipeline_cfg = YmlRepository.load_pipeline_config(pipeline_name)
        log.debug(f"Loaded pipeline config with {len(pipeline_cfg.phases)} phases")

        # Step 2: Load dataset config
        dataset_cfg = YmlRepository.get_dataset_by_key(dataset_key)
        log.debug(f"Loaded dataset config: {dataset_cfg.name}")

        # Step 3: Inject runtime variables
        ConfigBuilder._inject_runtime_vars(pipeline_cfg, dataset_cfg, notebook_vars)

        # Step 4: Apply dataset defaults to phase2
        ConfigBuilder._apply_dataset_defaults(pipeline_cfg, dataset_cfg)

        # Step 5: Inject mode and sample_rows from active profile into Phase 2 read_strategy
        ConfigBuilder._apply_profile_to_read_strategy(pipeline_cfg)

        # Step 6: Validate config structure with Pydantic DTO
        ConfigBuilder._validate_config_structure(pipeline_cfg)

        log.info("Pipeline config built successfully")
        return BuiltConfig(config=pipeline_cfg, dataset_key=dataset_key)

    @staticmethod
    def _inject_runtime_vars(
            pipeline_cfg: DictConfig,
            dataset_cfg: DictConfig,
            notebook_vars: Dict[str, Any]
    ) -> None:

        log.debug("Injecting runtime variables")

        # Extract dataset paths
        paths = dataset_cfg.get("paths", {})
        train_path = paths.get("train")
        test_path = paths.get("test")

        if not train_path and not test_path:
            log.error("No dataset paths found in dataset config")
            raise ValueError("Dataset config must have at least one path (train or test)")

        # Inject into phase2 if it exists
        if "phase2_data_understanding" in pipeline_cfg.phases:
            phase2 = pipeline_cfg.phases.phase2_data_understanding

            if "dataset_input" not in phase2:
                phase2.dataset_input = {}

            phase2.dataset_input.train_path = train_path
            phase2.dataset_input.test_path = test_path

            log.debug(f"Injected paths: train={train_path}, test={test_path}")

        # Inject notebook variables
        if notebook_vars.get("target_col"):
            OmegaConf.update(pipeline_cfg, "target_col", notebook_vars["target_col"])
            log.debug(f"Injected target_col: {notebook_vars['target_col']}")

        if notebook_vars.get("time_col"):
            OmegaConf.update(pipeline_cfg, "time_col", notebook_vars["time_col"])
            log.debug(f"Injected time_col: {notebook_vars['time_col']}")

        if notebook_vars.get("id_cols"):
            OmegaConf.update(pipeline_cfg, "id_cols", notebook_vars["id_cols"])
            log.debug(f"Injected id_cols: {notebook_vars['id_cols']}")

        # Override output_root if specified
        output_root = notebook_vars.get("output_root", DEFAULT_OUTPUT_ROOT)
        pipeline_cfg.common_base_config.runtime.output_root = output_root
        log.debug(f"Set output_root: {output_root}")

    @staticmethod
    def _apply_dataset_defaults(
            pipeline_cfg: DictConfig,
            dataset_cfg: DictConfig
    ) -> None:

        log.debug("Applying dataset defaults to phase2")

        if "phase2_data_understanding" not in pipeline_cfg.phases:
            log.warning("phase2_data_understanding not found, skipping dataset defaults")
            return

        phase2 = pipeline_cfg.phases.phase2_data_understanding

        if "dataset_input" not in phase2:
            phase2.dataset_input = {}

        dataset_input = phase2.dataset_input

        # Apply CSV params if not already set
        if "csv_params" not in dataset_input or not dataset_input.csv_params:
            csv_params = dataset_cfg.get("csv_params", {})
            dataset_input.csv_params = csv_params
            log.debug(f"Applied csv_params: {dict(csv_params)}")

        # Apply paths as multi_source if not set
        if "multi_source" not in dataset_input or not dataset_input.multi_source:
            paths = dataset_cfg.get("paths", {})
            dataset_input.multi_source = paths
            log.debug(f"Applied multi_source paths: {list(paths.keys())}")

        log.debug("Dataset defaults applied successfully")

    @staticmethod
    def _apply_profile_to_read_strategy(pipeline_cfg: DictConfig) -> None:
        """
        Inject ``mode`` and ``sample_rows`` from the active profile into Phase 2's
        ``read_strategy`` block.

        The base pipeline YAML defines ``read_strategy`` WITHOUT ``mode`` and
        ``sample_rows`` -- those values live only in the profile definitions
        (e.g. ``profiles.dev.mode="sample"``, ``profiles.dev.sample_rows=50000``).

        ``ReadStrategyContract.from_dict()`` requires both fields (raises
        ``KeyError`` otherwise). This method bridges the gap by copying them
        from the active profile into ``phase2_data_understanding.read_strategy``
        **only if they are missing or empty**, so the profile serves as the
        default but inline overrides are respected.

        Called from ``build_pipeline_config()`` after ``_apply_dataset_defaults()``
        and before ``_validate_config_structure()``.

        Parameters
        ----------
        pipeline_cfg : DictConfig
            Merged pipeline configuration (base + specific) with profiles
            already loaded.
        """
        # Step 1: CALL YmlRepository.get_active_profile -- read active profile name
        active_profile: str = YmlRepository.get_active_profile()

        # Step 2: Guard -- profile must exist in pipeline configuration.
        if active_profile not in pipeline_cfg.profiles:
            log.warning(
                "[_apply_profile_to_read_strategy] active profile '%s' not found "
                "in pipeline config -- skipping read_strategy injection",
                active_profile,
            )
            return

        # Step 3: Guard -- Phase 2 must be present.
        if "phase2_data_understanding" not in pipeline_cfg.phases:
            log.debug(
                "[_apply_profile_to_read_strategy] phase2_data_understanding not "
                "present -- skipping profile injection",
            )
            return

        profile = pipeline_cfg.profiles[active_profile]

        # Step 4: Guard -- Phase 2 must have a read_strategy block.
        phase2 = pipeline_cfg.phases.phase2_data_understanding
        if "read_strategy" not in phase2:
            log.debug(
                "[_apply_profile_to_read_strategy] phase2 has no read_strategy "
                "block -- creating empty one",
            )
            phase2.read_strategy = {}

        read_strategy = phase2.read_strategy

        # Step 5: Inject mode -- only if missing or empty.
        if not read_strategy.get("mode"):
            read_strategy.mode = profile.mode
            log.debug(
                "[_apply_profile_to_read_strategy] injected mode='%s' "
                "from profile '%s'",
                profile.mode,
                active_profile,
            )
        else:
            log.debug(
                "[_apply_profile_to_read_strategy] mode already set to '%s' "
                "-- keeping inline value",
                read_strategy.mode,
            )

        # Step 6: Inject sample_rows -- only if missing or empty.
        if not read_strategy.get("sample_rows"):
            read_strategy.sample_rows = profile.sample_rows
            log.debug(
                "[_apply_profile_to_read_strategy] injected sample_rows=%d "
                "from profile '%s'",
                profile.sample_rows,
                active_profile,
            )
        else:
            log.debug(
                "[_apply_profile_to_read_strategy] sample_rows already set to %d "
                "-- keeping inline value",
                read_strategy.sample_rows,
            )
        # ──────────────────────────────────────────────────────
        # Step 7: Build deterministic artifact names  ← New
        # ──────────────────────────────────────────────────────
        ConfigBuilder._inject_artifact_names(
            pipeline_cfg,
            profile,
            active_profile,
            read_strategy.get("sample_method", "random"),
        )

        # ──────────────────────────────────────────────────────
        log.info(
            "[_apply_profile_to_read_strategy] done -- mode=%s sample_rows=%d "
            "from profile='%s'",
            read_strategy.mode,
            read_strategy.sample_rows,
            active_profile,
        )

    @staticmethod
    def _validate_config_structure(pipeline_cfg: DictConfig) -> None:

        log.debug("Validating config structure with Pydantic DTO")

        try:
            config_dict = OmegaConf.to_container(pipeline_cfg, resolve=True)
            PipelineConfig(**config_dict)
            log.info("Config structure validation passed")
        except Exception as e:
            log.error(f"Config structure validation failed: {e}")
            raise ValueError(f"Invalid config structure: {e}")

    @staticmethod
    def get_active_profile_config(config: DictConfig) -> DictConfig:

        active_profile = YmlRepository.get_active_profile()
        log.info(f"Getting active profile config: {active_profile}")

        if active_profile not in config.profiles:
            log.warning(f"Active profile '{active_profile}' not found, using first available")
            available = list(config.profiles.keys())
            if not available:
                raise ValueError("No profiles defined in config")
            active_profile = available[0]

        profile_cfg = config.profiles[active_profile]
        log.debug(f"Active profile: {active_profile}, sample_rows={profile_cfg.sample_rows}")

        return profile_cfg

    @staticmethod
    def _inject_artifact_names(
            pipeline_cfg: DictConfig,
            profile: DictConfig,
            active_profile: str,
            sample_method: str,
    ) -> None:
        """
        Build deterministic artifact filenames encoding profile parameters
        into phase2 output_artifacts so Phase 3 can resolve them via
        OmegaConf interpolation.

        Pattern: 2.1.data_acquisition.{profile}_{sample_rows}_{method}_{split}.parquet
        Example: 2.1.data_acquisition.dev_7000_stratified_train.parquet
        """
        if "phase2_data_understanding" not in pipeline_cfg.phases:
            return

        phase2 = pipeline_cfg.phases.phase2_data_understanding

        if "steps" not in phase2 or "step_2_1_data_acquisition" not in phase2.steps:
            log.warning(
                "[_inject_artifact_names] step_2_1_data_acquisition not found "
                "after merge — skipping artifact name injection"
            )
            return

        suffix = f"{active_profile}_{profile.sample_rows}_{sample_method}"
        artifacts = phase2.steps.step_2_1_data_acquisition.output_artifacts

        artifacts.sample_train_parquet = f"2.1.data_acquisition.{suffix}_train.parquet"
        artifacts.sample_test_parquet  = f"2.1.data_acquisition.{suffix}_test.parquet"

        log.info(
            "[_inject_artifact_names] train=%s test=%s",
            artifacts.sample_train_parquet,
            artifacts.sample_test_parquet,
        )

def build_config(
        pipeline_name: str,
        dataset_key: str,
        notebook_vars: Optional[Dict[str, Any]] = None
) -> BuiltConfig:

    log.info(f"build_config: pipeline={pipeline_name}, dataset={dataset_key}")
    return ConfigBuilder.build_pipeline_config(pipeline_name, dataset_key, notebook_vars)

