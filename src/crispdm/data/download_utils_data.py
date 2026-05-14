# src/crispdm/common/download_utils_data.py

import shutil
from pathlib import Path

import kagglehub

from crispdm.common.logging_adapter_common import get_logger

# Initialize logger
logger = get_logger(__name__)


# =============================================================================
# Why this module exists
# -----------------------------------------------------------------------------
# Implements automated data acquisition and local mirroring for the
# Microsoft Security Incident Prediction dataset.
#
# Ensures the pipeline is self-contained: it eliminates manual downloads by
# fetching raw data directly from Kaggle Hub and organizing it into the
# project's internal directory structure (data/raw/).
#
# Key Features:
# - Environment-agnostic path resolution (relative to project root).
# - Idempotent execution: checks for existing files to skip redundant network
#   calls unless a re-download is explicitly forced.
# - Integrity-aware mirroring: maps remote Kaggle filenames to the specific
#   directory structure required by Stage 2 of the CRISP-DM pipeline.
#
# Program flow:
# -----------------------------------------------------------------------------
# - download_microsoft_dataset(force)
#   -> Resolve absolute project root via __file__ and pathlib.
#   -> Define expected paths for GUIDE_Train.csv and GUIDE_Test.csv.
#   -> IF files exist AND force=False: Log skip and exit.
#   -> ELSE: Authenticate and fetch dataset to global Kaggle cache.
#   -> Ensure local data/raw/{train,test} folders exist.
#   -> Mirror (copy2) files from cache to project raw storage.
#   -> Log completion with absolute paths for traceability.
#
# Design patterns
# -----------------------------------------------------------------------------
# - GoF -> Gang of Four:
#   - Singleton (Logging): Uses a unified logging instance to ensure
#               consistent traceability across the acquisition process.
#   - Proxy: Acts as an infrastructure proxy between the external Kaggle
#               storage and the internal pipeline's Data Access Layer.
# - Enterprise/Architectural:
#   - Repository / DAO (Thin):
#               Encapsulates the logic of "where and how" data is retrieved.
#               Higher-level phases only care that data exists in data/raw/,
#               not that it came from a specific Kaggle API call.
#   - Idempotent Receiver:
#               Ensures that multiple calls to the acquisition service
#               result in the same system state without side effects or
#               unnecessary resource consumption.
# =============================================================================
def download_microsoft_dataset(
        force: bool = False,
) -> None:
    """
    Download the Microsoft Security Incident dataset and mirror it to data/raw/.

    The local filesystem is the primary check — if the target files exist, the
    download is skipped unless explicitly forced. This ensures environment-agnostic
    reproducibility and prevents redundant network overhead during pipeline reruns.

    Covers: Automated acquisition, path resolution, and local persistence.

    Parameters
    ----------
    force : bool, default=False
            If True, ignores existing local files and re-downloads the dataset.
            Useful for data recovery or refreshing corrupted raw files.

    Returns
    -------
    None
            Files are persisted directly to the project's ``data/raw/`` directory.

    Raises
    ------
    FileNotFoundError
        If the expected files are missing from the Kaggle download payload.
    Exception
        For critical failures during network acquisition or filesystem I/O.
    """
    try:
        # 1. Resolve paths using pathlib (Rule PTH)
        # Assuming: src/crispdm/common/download_utils_data.py -> project_root is 3 levels up
        project_root = Path(__file__).resolve().parents[3]

        raw_dir = project_root / "data" / "raw"
        train_dest = raw_dir / "train" / "GUIDE_Train.csv"
        test_dest = raw_dir / "test" / "GUIDE_Test.csv"

        # 2. Idempotency check
        if not force and train_dest.exists() and test_dest.exists():
            logger.info("Dataset already exists at %s. Skipping download.", raw_dir)
            return

        logger.info("Starting download from Kaggle Hub...")

        # 3. Download via kagglehub (Cross-platform global cache)
        # This returns a string path, so we cast to Path immediately
        download_path = Path(
            kagglehub.dataset_download(
                "microsoft/microsoft-security-incident-prediction"
            )
        )

        # 4. Ensure directory structure (Exist_ok prevents race conditions)
        train_dest.parent.mkdir(parents=True, exist_ok=True)
        test_dest.parent.mkdir(parents=True, exist_ok=True)

        # 5. Mirror files to project structure
        # Mapping for clarity and easier maintenance if filenames change
        files_to_copy = {
            "GUIDE_Train.csv": train_dest,
            "GUIDE_Test.csv": test_dest,
        }

        for filename, destination in files_to_copy.items():
            source = download_path / filename
            if not source.exists():
                logger.warning(
                    "Expected file %s not found in downloaded dataset.", filename
                )
                continue

            shutil.copy2(source, destination)  # copy2 preserves metadata
            logger.debug("Copied %s to %s", filename, destination)

        logger.info("Download and mirroring completed successfully.")
        logger.info("Data located in: %s", raw_dir)

    except Exception:
        # Rule TRY400: Use logging.exception to capture the stack trace automatically
        logger.exception("Critical error during data acquisition from Kaggle")
        raise
