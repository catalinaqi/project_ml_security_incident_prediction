# src/crispdm/main.py
import argparse
import logging
import sys
from pathlib import Path

from crispdm.api.execution_facade_api import (
    init_run_phase2,
    run_phase2_1,
    run_phase2_2,
    run_phase2_3,
    run_phase2_4,
    run_phase3_1,
    run_phase3_2,
    run_phase3_3,
    run_phase3_5,
    run_phase4_1,
    run_phase4_2,
    run_phase4_3,
    run_phase4_4,
    run_phase4_5,
)
from crispdm.common.path_service_common import find_project_root


def main():
    parser = argparse.ArgumentParser(
        prog="crisp-dm-pipeline",
        description="CRISP-DM Pipeline - Phase 2 (Data Understanding) + Phase 3.1 (Data Selection)",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Root directory for outputs (default: project_root/outputs)",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )

    parser.add_argument(
        "--pipeline",
        default="clustering",
        help="Pipeline name (e.g. 'clustering', 'classification') (default: clustering)",
    )

    parser.add_argument(
        "--dataset-key",
        default="ms_sec_inc_pre",
        help="Dataset key in dataset_config.yml (default: ms_sec_inc_pre)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configs without executing phases",
    )

    args = parser.parse_args()

    try:
        project_root = find_project_root()
    except RuntimeError as e:
        print(f"ERROR: Cannot find project root: {e}", file=sys.stderr)
        sys.exit(1)

    output_root = args.output_root or (project_root / "outputs")
    output_root.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("crisp-dm.main")
    logger.info("=" * 70)
    logger.info("CRISP-DM PIPELINE")
    logger.info("Phase 2 (Data Understanding) + Phase 3.1 (Data Selection)")
    logger.info("=" * 70)
    logger.info(f"Project root: {project_root}")
    logger.info(f"Output root: {output_root}")
    logger.info(f"Pipeline: {args.pipeline}")
    logger.info(f"Dataset key: {args.dataset_key}")

    notebook_vars = {
        "output_root": str(output_root),
        "id_cols": "IncidentId,AlertId,DetectorId,DeviceId",
    }

    logger.debug(f"Runtime variables: {notebook_vars}")

    if args.dry_run:
        logger.info("DRY RUN MODE - configs validated, exiting")
        sys.exit(0)

    try:
        # ── Phase 2 ──────────────────────────────────────────────────────
        logger.info("\n" + "-" * 70)
        logger.info("Step 2.0: Initializing run context")
        logger.info("-" * 70)

        ctx = init_run_phase2(
            pipeline_name=args.pipeline,
            dataset_key=args.dataset_key,
            notebook_vars=notebook_vars,
        )

        logger.info(f"Run ID: {ctx.run_id}")
        logger.info(f"Run directory: {ctx.run_dir}")

        logger.info("\n" + "-" * 70)
        logger.info("Step 2.1: Initial Data Collection")
        logger.info("-" * 70)
        ctx = run_phase2_1(ctx)
        if ctx.df_train is not None:
            logger.info(f"Train data loaded: {len(ctx.df_train):,} rows x {len(ctx.df_train.columns)} cols")
        if ctx.df_test is not None:
            logger.info(f"Test data loaded: {len(ctx.df_test):,} rows x {len(ctx.df_test.columns)} cols")

        logger.info("\n" + "-" * 70)
        logger.info("Step 2.2: Data Description")
        logger.info("-" * 70)
        ctx = run_phase2_2(ctx)
        logger.info(f"Artifacts directory: {ctx.phase2_dir}")

        logger.info("\n" + "-" * 70)
        logger.info("Step 2.3: Data Quality Verification")
        logger.info("-" * 70)
        ctx = run_phase2_3(ctx)

        logger.info("\n" + "-" * 70)
        logger.info("Step 2.4: Exploratory Data Analysis")
        logger.info("-" * 70)
        ctx = run_phase2_4(ctx)

        logger.info("\n" + "=" * 70)
        logger.info("PHASE 2 COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)

        # ── Phase 3.1 ────────────────────────────────────────────────────
        logger.info("\n" + "-" * 70)
        logger.info("Step 3.1: Data Selection (Sentinel Removal)")
        logger.info("-" * 70)
        ctx = run_phase3_1(ctx)
        if ctx.df_train is not None:
            logger.info(f"Train data after Phase 3.1: {len(ctx.df_train):,} rows x {len(ctx.df_train.columns)} cols")

        # ── Phase 3.2 ────────────────────────────────────────────────────
        logger.info("\n" + "-" * 70)
        logger.info("Step 3.2: data_cleaning")
        logger.info("-" * 70)
        ctx = run_phase3_2(ctx)
        if ctx.df_train is not None:
            logger.info(f"Train data after Phase 3.2: {len(ctx.df_train):,} rows x {len(ctx.df_train.columns)} cols")

        # ── Phase 3.3 ────────────────────────────────────────────────────
        logger.info("\n" + "-" * 70)
        logger.info("Step 3.3: data_transformation")
        logger.info("-" * 70)
        ctx = run_phase3_3(ctx)
        if ctx.df_train is not None:
            logger.info(f"Train data after Phase 3.3: {len(ctx.df_train):,} rows x"
                        f" {len(ctx.df_train.columns)} cols")

        # ── Phase 3.5 ────────────────────────────────────────────────────
        logger.info("\n" + "-" * 70)
        logger.info("Step 3.5: data_transformation")
        logger.info("-" * 70)
        ctx = run_phase3_5(ctx)
        if ctx.df_train is not None:
            logger.info(f"Train data after Phase 3.5: {len(ctx.df_train):,} rows x"
                        f" {len(ctx.df_train.columns)} cols")

        # ── Phase 4.1 ────────────────────────────────────────────────────
        logger.info("\n" + "-" * 70)
        logger.info("Step 4.1: data_transformation")
        logger.info("-" * 70)
        ctx = run_phase4_1(ctx)
        if ctx.df_train is not None:
            logger.info(f"Train data after Phase 4.1: {len(ctx.df_train):,} rows x"
                        f" {len(ctx.df_train.columns)} cols")
        # ── Phase 4.2 ────────────────────────────────────────────────────
        logger.info("\n" + "-" * 70)
        logger.info("Step 4.2: data_transformation")
        logger.info("-" * 70)
        ctx = run_phase4_2(ctx)
        if ctx.df_train is not None:
            logger.info(f"Train data after Phase 4.2: {len(ctx.df_train):,} rows x"
                        f" {len(ctx.df_train.columns)} cols")


        # ── Phase 4.3 ────────────────────────────────────────────────────
        logger.info("\n" + "-" * 70)
        logger.info("Step 4.3: data_transformation")
        logger.info("-" * 70)
        ctx = run_phase4_3(ctx)
        if ctx.df_train is not None:
            logger.info(f"Train data after Phase 4.3: {len(ctx.df_train):,} rows x"
                        f" {len(ctx.df_train.columns)} cols")

        # ────────────────────────────────────────────────────
        # ── Phase 4.4 ────────────────────────────────────────────────────
        logger.info("\n" + "-" * 70)
        logger.info("Step 4.3: data_transformation")
        logger.info("-" * 70)
        ctx = run_phase4_4(ctx)
        if ctx.df_train is not None:
            logger.info(f"Train data after Phase 4.3: {len(ctx.df_train):,} rows x"
                        f" {len(ctx.df_train.columns)} cols")

        # ────────────────────────────────────────────────────
        # ── Phase 4.5 ────────────────────────────────────────────────────
        logger.info("\n" + "-" * 70)
        logger.info("Step 4.3: data_transformation")
        logger.info("-" * 70)
        ctx = run_phase4_5(ctx)
        if ctx.df_train is not None:
            logger.info(f"Train data after Phase 4.3: {len(ctx.df_train):,} rows x"
                        f" {len(ctx.df_train.columns)} cols")

        # ────────────────────────────────────────────────────
        logger.info("\n" + "=" * 70)
        logger.info("PIPELINE EXECUTED SUCCESSFULLY (Phase 2 + Phase 3.1 + Phase 3.2 "
                    "+ Phase 3.3 + Phase 3.5 + 4.1 + 4.2 + 4.3)")
        logger.info("=" * 70)
        logger.info(f"Run directory: {ctx.run_dir}")

        sys.exit(0)

    except Exception as e:
        logger.critical(f"Pipeline failed: {type(e).__name__}: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()