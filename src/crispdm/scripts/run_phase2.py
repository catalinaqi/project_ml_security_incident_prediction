#!/usr/bin/env python3
"""
CRISP-DM - Fase 2: Data Understanding Runner
Script isolato per inizializzare il contesto di esecuzione ed eseguire i passi da 2.1 a 2.4.
I risultati vengono persisti su disco per essere consumati dalle fasi successive.

Esecuzione dalla radice del progetto:
    poetry run python scripts/run_phase2.py --pipeline clustering --dataset-key ms_sec_inc_pre
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Infrastruttura ufficiale CRISP-DM
from crispdm.api.execution_facade_api import (
    init_run_phase2,
    run_phase2_1,
    run_phase2_2,
    run_phase2_3,
    run_phase2_4,
)
from crispdm.common.path_service_common import find_project_root

# Integrazione del modulo di logging personalizzato del progetto
from crispdm.common.logging_adapter_common import get_logger

# Inizializzazione del logger specifico per questo modulo
log = get_logger("run_phase2")


def parse_arguments() -> argparse.Namespace:
    """Analizza gli argomenti della riga di comando per la fase 2."""
    parser = argparse.ArgumentParser(
        description="CRISP-DM Isolated Runner - Phase 2 (Data Understanding)"
    )
    parser.add_argument(
        "--pipeline",
        default="clustering",
        help="Nome della pipeline (es. 'clustering', 'classification') (default: clustering)",
    )
    parser.add_argument(
        "--dataset-key",
        default="ms_sec_inc_pre",
        help="Chiave del dataset definita in dataset_config.yml (default: ms_sec_inc_pre)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Livello di verbosità del log (default: INFO)",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Nota: Il livello effettivo del logger radice verrà configurato internamente
    # dalla Facade tramite il file YAML. Usiamo INFO per i messaggi di avvio del runner.
    log.info("=" * 80)
    log.info("🚀 AVVIO ESECUZIONE ISOLATA: CRISP-DM FASE 2")
    log.info("=" * 80)

    # 1. Risoluzione dei percorsi di base del progetto
    try:
        project_root = find_project_root()
        log.info(f"✅ Radice del progetto rilevata: {project_root}")
    except RuntimeError as e:
        log.error(f"❌ Impossibile determinare la radice del progetto: {e}")
        sys.exit(1)

    output_root = project_root / "outputs"
    notebook_vars = {
        "output_root": str(output_root),
        "id_cols": "IncidentId,AlertId,DetectorId,DeviceId",
    }
    log.debug(f"Variabili di runtime configurate: {notebook_vars}")

    # 2. Inizializzazione del RunContext (Passo 2.0)
    # Questa chiamata attiverà internamente il modulo logging_adapter_common
    try:
        log.info("⏳ [Step 2.0] Inizializzazione del contesto di esecuzione (RunContext)...")
        ctx = init_run_phase2(
            pipeline_name=args.pipeline,
            dataset_key=args.dataset_key,
            notebook_vars=notebook_vars,
        )
        log.info(f"✨ Run ID Generato con successo: {ctx.run_id}")
        log.info(f"📁 Directory della corsa corrente: {ctx.run_dir}")
    except Exception as e:
        log.error(f"❌ Errore critico durante l'inizializzazione del contesto: {e}", exc_info=True)
        sys.exit(1)

    # 3. Esecuzione sequenziale dei passi della Fase 2
    try:
        # --- PASO 2.1: Raccolta Iniziale dei Dati ---
        log.info("⏳ [Step 2.1] Esecuzione: Initial Data Collection...")
        ctx = run_phase2_1(ctx)

        if ctx.df_train is not None:
            log.info(f"📊 Dati di Train caricati: {len(ctx.df_train):,} righe x {len(ctx.df_train.columns)} colonne")
        else:
            log.warning("⚠️ df_train è None o non è stato caricato correttamente.")

        if ctx.df_test is not None:
            log.info(f"📊 Dati di Test caricati: {len(ctx.df_test):,} righe x {len(ctx.df_test.columns)} colonne")
        else:
            log.debug("Nessun dataset di test indipendente rilevato (df_test è None).")

        # --- PASO 2.2: Descrizione dei Dati ---
        log.info("⏳ [Step 2.2] Esecuzione: Data Description...")
        ctx = run_phase2_2(ctx)
        log.info(f"💾 Artefatti di descrizione salvati in: {ctx.phase2_dir}")

        # --- PASO 2.3: Verifica della Qualità dei Dati ---
        log.info("⏳ [Step 2.3] Esecuzione: Data Quality Verification...")
        ctx = run_phase2_3(ctx)

        # --- PASO 2.4: Analisi Esplorativa dei Dati (EDA) ---
        log.info("⏳ [Step 2.4] Esecuzione: Exploratory Data Analysis (EDA)...")
        ctx = run_phase2_4(ctx)

        # 4. Chiusura della Fase con successo
        log.info("=" * 80)
        log.info("🏁 FASE 2 COMPLETATA CON SUCCESSO E PERSISTITA SU DISCO")
        log.info(f"📍 ID della Corsa da usare per la Fase 3: {ctx.run_id}")
        log.info(f"📍 Posizione fisica dei file Parquet: {ctx.run_dir}")
        log.info("=" * 80)

    except Exception as e:
        log.error(f"❌ Il pipeline è fallito alla Fase 2 a causa di un'eccezione: {type(e).__name__}: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()