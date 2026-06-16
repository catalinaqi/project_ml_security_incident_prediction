#!/usr/bin/env python3
"""
CRISP-DM - Fase 3: Data Preparation Runner
Script isolato che individua dinamicamente l'ultimo run della Fase 2 sul disco,
carica i dati persistiti ed esegue sequenzialmente i passi da 3.1 a 3.5 in memoria.

Esecuzione dalla radice del progetto:
    poetry run python src/crispdm/scripts/run_phase3.py --pipeline clustering --dataset-key ms_sec_inc_pre
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd

# Servizi ufficiali ed Enum dell'infrastruttura CRISP-DM
from crispdm.configuration.enum_registry_config import PhaseDir
from crispdm.common.path_service_common import find_project_root
from crispdm.common.context_facade_common import RunContext
from crispdm.configuration.yml_repository_config import YmlRepository

# Runner ufficiali della Fase 3
from crispdm.phase.phase3_preparation_runner_phase import (
    run_step_3_1,
    run_step_3_2,
    run_step_3_3,
    run_step_3_5,
)

# Integrazione del modulo di logging personalizzato del progetto
from crispdm.common.logging_adapter_common import get_logger

# Inizializzazione del logger per la Fase 3
log = get_logger("run_phase3")


def parse_arguments() -> argparse.Namespace:
    """Analizza gli argomenti della riga di comando per la fase 3."""
    parser = argparse.ArgumentParser(
        description="CRISP-DM Isolated Runner - Phase 3 (Data Preparation)"
    )
    parser.add_argument(
        "--pipeline",
        default="clustering",
        help="Nome della pipeline (es. 'clustering') (default: clustering)",
    )
    parser.add_argument(
        "--dataset-key",
        default="ms_sec_inc_pre",
        help="Chiave del dataset definita in dataset_config.yml (default: ms_sec_inc_pre)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Opzionale: ID specifico di un run (es. '20260616_152420'). Se omesso, userà l'ultimo disponibile.",
    )
    return parser.parse_args()


def find_target_run_dir(base_dir: Path, explicit_id: str | None) -> Path:
    """Individua la directory del run target (l'ultima creata o quella passata esplicitamente)."""
    if not base_dir.exists():
        raise FileNotFoundError(f"❌ La directory di base dei run non esiste: {base_dir}")

    if explicit_id:
        target_dir = base_dir / explicit_id
        if not target_dir.exists():
            raise FileNotFoundError(f"❌ Il run ID specificato '{explicit_id}' non esiste in: {base_dir}")
        return target_dir

    # Ricerca dinamica dell'ultimo run (ordinamento alfabetico basato sul timestamp YYYYMMDD_HHMMSS)
    all_runs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
    if not all_runs:
        raise RuntimeError(f"❌ Nessun run trovato nella directory: {base_dir}")

    latest_run = all_runs[-1]
    log.info(f"🔎 Nessun Run ID specificato. Individuato automaticamente l'ultimo run: {latest_run.name}")
    return latest_run


def main():
    args = parse_arguments()

    log.info("=" * 80)
    log.info("🚀 AVVIO ESECUZIONE ISOLATA: CRISP-DM FASE 3 (DATA PREPARATION)")
    log.info("=" * 80)

    # 1. Risoluzione della Radice del Progetto e dei Percorsi dei Run
    try:
        project_root = find_project_root()
        base_runs_dir = project_root / "outputs" / "runs" / args.pipeline / args.dataset_key

        # Individuazione dinamica della cartella di esecuzione
        run_dir = find_target_run_dir(base_runs_dir, args.run_id)
        log.info(f"📁 Directory del Run selezionata: {run_dir}")
    except Exception as e:
        log.error(f"❌ Errore durante la risoluzione dei percorsi: {e}")
        sys.exit(1)

    # 2. Individuazione dei File di Input della Fase 2 (Risoluzione Intelligente)
    # Supporta sia 'phase2' che 'phase2_data_understanding' come mostrato nelle immagini
    phase2_folder = "phase2_data_understanding" if (run_dir / "phase2_data_understanding").exists() else PhaseDir.PHASE2.value
    phase2_path = run_dir / phase2_folder

    log.info(f"📂 Ricerca degli artefatti della Fase 2 in: {phase2_path}")

    # Cerchiamo qualsiasi file che termini con '_train.parquet' (es. dev_1000_stratified_train.parquet)
    train_candidates = list(phase2_path.glob("*_train.parquet"))
    if not train_candidates:
        log.error(f"❌ Impossibile trovare il file Parquet di Train (*_train.parquet) in: {phase2_path}")
        sys.exit(1)

    f2_input_train_path = train_candidates[0]
    log.info(f"✅ File di Train individuato: {f2_input_train_path.name}")

    # Proviamo a cercare anche il file di test (opzionale)
    test_candidates = list(phase2_path.glob("*_test.parquet"))
    f2_input_test_path = test_candidates[0] if test_candidates else None
    if f2_input_test_path:
        log.info(f"✅ File di Test individuato: {f2_input_test_path.name}")

    # 3. Caricamento dei Dati e della Configurazione
    try:
        log.info("📥 Caricamento della configurazione tramite YmlRepository...")
        cfg = YmlRepository.load_pipeline_config(args.pipeline)

        log.info(f"📥 Lettura del dataset di Train dal disco...")
        df_raw_train = pd.read_parquet(f2_input_train_path)
        log.info(f"   📊 Dataset di Train caricato con successo. Shape: {df_raw_train.shape}")

        df_raw_test = None
        if f2_input_test_path:
            log.debug("📥 Lettura del dataset di Test dal disco...")
            df_raw_test = pd.read_parquet(f2_input_test_path)
            log.debug(f"   📊 Dataset di Test caricato. Shape: {df_raw_test.shape}")

    except Exception as e:
        log.error(f"❌ Errore durante il caricamento dei file sorgente: {e}", exc_info=True)
        sys.exit(1)

    # 4. Inizializzazione del RunContext Ufficiale per la Fase 3
    ctx = RunContext(
        config=cfg,
        run_dir=run_dir,
        df_train=df_raw_train,
        df_test=df_raw_test,
        run_id=run_dir.name,
        dataset_key=args.dataset_key
    )

    # 5. Esecuzione Sequenziale della Pipeline in Memoria (Passi da 3.1 a 3.5)
    try:
        log.info("\n⚡ [Esecuzione Step 3.1] Data Selection...")
        ctx = run_step_3_1(ctx)

        log.info("⚡ [Esecuzione Step 3.2] Data Cleaning...")
        ctx = run_step_3_2(ctx)

        log.info("⚡ [Esecuzione Step 3.3] Feature Transformation...")
        ctx = run_step_3_3(ctx)

        log.info("\n🎯 [Esecuzione Step 3.5] Final Formatting (Salvataggio etichette ausiliarie)...")
        ctx = run_step_3_5(ctx)
        log.info("  ✅ Step 3.5 completato con successo senza eccezioni.")

    except Exception as e:
        log.error(f"❌ La pipeline è fallita durante l'elaborazione della Fase 3: {e}", exc_info=True)
        sys.exit(1)

    # 6. Diagnostica Post-Esecuzione degli Artefatti Generati su Disco
    log.info("\n" + "=" * 80)
    log.info("🔍 VERIFICA DEGLI ARTEFATTI GENERATI DELLA FASE 3 SU DISCO")
    log.info("=" * 80)

    # Verifica dinamica della cartella di output della Fase 3
    phase3_folder_name = "phase3_data_preparation" if (run_dir / "phase3_data_preparation").exists() else PhaseDir.PHASE3.value
    phase3_dir = run_dir / phase3_folder_name

    # Cerchiamo il file di Ground Truth o i file Parquet generati nel passo 3.5
    gt_candidates = list(phase3_dir.glob("*save_auxiliary_labels*"))

    if gt_candidates:
        gt_path = gt_candidates[0]
        log.info(f"📄 Analisi del file Parquet generato dallo Step 3.5:\n    {gt_path}")
        df_gt = pd.read_parquet(gt_path)
        log.info(f"  ✅ File letto correttamente dal disco.")
        log.info(f"  📋 Colonne rilevate: {df_gt.columns.tolist()}")
        log.info(f"  📊 Dimensione finale dell'Auxiliary DataFrame: {df_gt.shape}")

        if "IncidentGrade" in df_gt.columns and "label" in df_gt.columns:
            log.info("🎉 ✨ SUCCESSO TOTALE! Il file contiene la struttura ibrida attesa.")
            log.info(f"\n👀 Campione dei primi 2 record:\n{df_gt.head(2).to_string(index=False)}")
        else:
            log.warning("⚠️ Attenzione: Il file esiste ma non contiene contemporaneamente le colonne 'IncidentGrade' e 'label'.")
    else:
        # Controllo generico se la cartella contiene comunque dei file processati
        all_p3_files = list(phase3_dir.glob("*.parquet"))
        if all_p3_files:
            log.info(f"✅ La Fase 3 ha scritto correttamente {len(all_p3_files)} file Parquet in: {phase3_dir}")
        else:
            log.error(f"❌ Errore: L'elaborazione è terminata ma nessun file fisico è stato scritto in: {phase3_dir}")

    log.info("\n" + "=" * 80)
    log.info("🏁 FINE DEL TEST DI COPERTURA FASE 3.")
    log.info("=" * 80)


if __name__ == "__main__":
    main()