#!/usr/bin/env python3
"""
CRISP-DM - Fase 4: Modelling Runner
Script isolato che individua dinamicamente l'ultimo run sul disco, carica i dati
preparati dalla Fase 3, esegue l'addestramento dei modelli (4.3) e la loro
valutazione prestazionale (4.4 e 4.5).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from omegaconf import OmegaConf  # <-- AGGIUNTO per iniettare il nodo runtime

# Servizi ufficiali ed Enum dell'infrastruttura CRISP-DM
from crispdm.configuration.enum_registry_config import PhaseDir
from crispdm.common.path_service_common import find_project_root
from crispdm.common.context_facade_common import RunContext
from crispdm.configuration.yml_repository_config import YmlRepository

# Runner ufficiali di Step della Fase 4 aggiornati
from crispdm.phase.phase4_modeling_runner_phase import (
    run_step_4_1,
    run_step_4_2,
    run_step_4_3,
    run_step_4_4,
    run_step_4_5,
)

# Integrazione del modulo di logging personalizzato del progetto
from crispdm.common.logging_adapter_common import get_logger

# Inizializzazione del logger per la Fase 4
log = get_logger("run_phase4")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CRISP-DM Isolated Runner - Phase 4 (Modelling)")
    parser.add_argument("--pipeline", default="clustering")
    parser.add_argument("--dataset-key", default="ms_sec_inc_pre")
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def find_target_run_dir(base_dir: Path, explicit_id: str | None) -> Path:
    if not base_dir.exists():
        raise FileNotFoundError(f"❌ La directory di base dei run non esiste: {base_dir}")
    if explicit_id:
        target_dir = base_dir / explicit_id
        if not target_dir.exists():
            raise FileNotFoundError(f"❌ Il run ID specificato '{explicit_id}' non esiste.")
        return target_dir
    all_runs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
    if not all_runs:
        raise RuntimeError(f"❌ Nessun run trovato nella directory: {base_dir}")
    return all_runs[-1]


def main():
    args = parse_arguments()

    log.info("=" * 80)
    log.info("🚀 AVVIO ESECUZIONE ISOLATA: CRISP-DM FASE 4 (MODELLING)")
    log.info("=" * 80)

    try:
        project_root = find_project_root()
        base_runs_dir = project_root / "outputs" / "runs" / args.pipeline / args.dataset_key
        run_dir = find_target_run_dir(base_runs_dir, args.run_id)
        log.info(f"📁 Directory del Run: {run_dir}")
    except Exception as e:
        log.error(f"❌ Errore durante la risoluzione dei percorsi: {e}")
        sys.exit(1)

    phase3_folder = "phase3_data_preparation" if (run_dir / "phase3_data_preparation").exists() else PhaseDir.PHASE3.value
    phase3_path = run_dir / phase3_folder

    p3_files = list(phase3_path.glob("*.parquet"))
    if not p3_files:
        log.error(f"❌ Nessun file Parquet trovato in: {phase3_path}")
        sys.exit(1)

    main_train_files = [f for f in p3_files if "auxiliary_labels" not in f.name]
    f3_input_path = main_train_files[0] if main_train_files else p3_files[0]

    # 3. Caricamento Dati e Configurazione con PATCH
    try:
        cfg = YmlRepository.load_pipeline_config(args.pipeline)

        # --- PATCH: Iniezione del blocco runtime mancante ---
        if "runtime" not in cfg:
            cfg = OmegaConf.merge(cfg, OmegaConf.create({"runtime": {"random_seed": 7}}))
            log.info("⚙️ Nodo 'runtime' iniettato artificialmente per evitare crash.")

        # --- PATCH: Limite righe per la RAM ---
        limit = cfg.phases.phase4_data_modeling.steps \
            .step_4_3_model_training.methods.model_training \
            .techniques.fit.params.get("max_training_rows", 30000)
        log.info(f"⚙️ max_training_rows impostato a: {limit}")

        df_prepared = pd.read_parquet(f3_input_path)
    except Exception as e:
        log.error(f"❌ Errore caricamento: {e}", exc_info=True)
        sys.exit(1)

    ctx = RunContext(
        config=cfg,
        run_dir=run_dir,
        df_train=df_prepared,
        df_test=None,
        run_id=run_dir.name,
        dataset_key=args.dataset_key
    )

    # Estrazione difensiva e troncamento di X_train
    feature_cols = [c for c in df_prepared.columns if c not in ["Id", "label"]]
    X_full = df_prepared[feature_cols].to_numpy().astype(np.float32)
    ctx.artifacts["X_train"] = X_full[:limit] if limit and limit < len(X_full) else X_full
    log.info(f"   ⚙️ Matrice X_train troncata e popolata. Shape finale: {ctx.artifacts['X_train'].shape}")

    # 5. Esecuzione
    try:
        log.info("\n⚡ [Esecuzione Step 4.1] Algorithm Selection...")
        ctx = run_step_4_1(ctx)

        log.info("\n⚡ [Esecuzione Step 4.2] Pretrain Analysis (k-NN Distance)...")
        ctx = run_step_4_2(ctx)

        log.info("\n⚡ [Esecuzione Step 4.3] Model Training & Hyperparameter Tuning...")
        ctx = run_step_4_3(ctx)

        log.info("\n⚡ [Esecuzione Step 4.4] Test Design Generation...")
        ctx = run_step_4_4(ctx)

        log.info("\n⚡ [Esecuzione Step 4.5] Model Evaluation...")
        ctx = run_step_4_5(ctx)

    except Exception as e:
        log.error(f"❌ Pipeline fallita in Fase 4: {e}", exc_info=True)
        sys.exit(1)

    log.info("\n" + "=" * 80)
    log.info("🏁 FASE 4 COMPLETATA CON SUCCESSO E MODELLI PERSISTITI")
    log.info(f"📍 Risultati in: {run_dir / PhaseDir.PHASE4.value}")
    log.info("=" * 80)

if __name__ == "__main__":
    main()