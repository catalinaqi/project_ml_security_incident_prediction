#!/usr/bin/env python3
"""
CRISP-DM - Master Pipeline Orchestrator (Integrazione Definitiva)
Punto di ingresso centrale del progetto che coordina l'esecuzione sequenziale
o isolata delle fases del ciclo di vita CRISP-DM (Dalla Fase 2 alla Fase 5).

================================================================================
SCENARI DI ESECUZIONE, COMANDI E COMPORTAMENTO DEI RUN-ID
================================================================================

SCENARIO 1: FLUSSO COMPLETO END-TO-END (Nuovo esperimento globale)
--------------------------------------------------------------------------------
* Comando:
    poetry run python src/crispdm/main.py --phase all
* Comportamento Run-ID:
    SÌ, genera un NUOVO run_id univoco (es. run_20260616_153022) tramite la Fase 2.
* Descrizione:
    Inizializza un nuovo run, esegue i controlli di qualità (F2), prepara i dati (F3),
    addestra i modelli gestendo la RAM (F4) e produce i report per il business (F5).
    Il passaggio dati avviene in memoria tramite l'oggetto RunContext.

SCENARIO 2: SOLO DATA UNDERSTANDING (Controllo qualità e auditing fonti)
--------------------------------------------------------------------------------
* Comando:
    poetry run python src/crispdm/main.py --phase 2
* Comportamento Run-ID:
    SÌ, genera un NUOVO run_id univoco su disco.
* Descrizione:
    Esegue esclusivamente la validazione statistica, il controllo dei tipi,
    dei valori nulli e l'Analisi Esplorativa dei Dati (EDA), salvando i report base.

SCENARIO 3: SOLO DATA PREPARATION (Modifiche a encoding, scaling o feature engineering)
--------------------------------------------------------------------------------
* Comando:
    poetry run python src/crispdm/main.py --phase 3
* Comportamento Run-ID:
    NO, non crea un nuovo ID. Si aggancia all'ULTIMO run disponibile sul disco.
* Descrizione:
    Recupera i dati grezzi validati dall'ultimo run della Fase 2, applica le
    trasformazioni logiche e scrive i file Parquet pronti nella cartella del run esistente.

SCENARIO 4: SOLO MODELLING (Tuning di iperparametri o cambio algoritmi nel YAML)
--------------------------------------------------------------------------------
* Comando:
    poetry run python src/crispdm/main.py --phase 4
* Comportamento Run-ID:
    NO, non crea un nuovo ID. Si aggancia all'ULTIMO run disponibile sul disco.
* Descrizione:
    Carica i Parquet della Fase 3 dall'ultimo run, applica la protezione difensiva
    contro le stringhe (es. 'TruePositive', 'IncidentGrade') per evitare crash matematici,
    addestra i clusterer e salva i modelli serializzati (.pkl) nello stesso run.

SCENARIO 5: SOLO VALUTAZIONE DI BUSINESS (Iterazioni rapide su metriche e report)
--------------------------------------------------------------------------------
* Comando:
    poetry run python src/crispdm/main.py --phase 5
* Comportamento Run-ID:
    NO, non crea un nuovo ID. Si aggancia all'ULTIMO run disponibile sul disco.
* Descrizione:
    Ripristina lo stato della pipeline leggendo i modelli e le label della Fase 4
    dal disco. Genera matrici di confusione e distribuzioni di business senza
    riaddestrare i modelli, risparmiando tempo e risorse computazionali.
================================================================================
"""

from __future__ import annotations

import argparse
import sys
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

# Infrastruttura e Servizi Comuni del Framework CRISP-DM
from crispdm.configuration.enum_registry_config import PhaseDir
from crispdm.common.path_service_common import find_project_root
from crispdm.common.context_facade_common import RunContext
from crispdm.configuration.yml_repository_config import YmlRepository
from crispdm.common.logging_adapter_common import get_logger

# Import dei Runner ufficiali estratti dagli script delle singole fases
from crispdm.api.execution_facade_api import (
    init_run_phase2, run_phase2_1, run_phase2_2, run_phase2_3, run_phase2_4
)
from crispdm.phase.phase3_preparation_runner_phase import (
    run_step_3_1, run_step_3_2, run_step_3_3, run_step_3_5
)
from crispdm.phase.phase4_modeling_runner_phase import (
    run_step_4_1, run_step_4_2, run_step_4_3, run_step_4_4, run_step_4_5
)
from crispdm.phase.phase5_evaluation_and_interpretation_phase import (
    run_step_5_1, run_step_5_2, run_step_5_3, run_step_5_4
)

# Inizializzazione del logger centrale per l'orchestrazione
log = get_logger("main_orchestrator")


def parse_arguments() -> argparse.Namespace:
    """Configura e analizza gli argomenti passati tramite l'interfaccia a riga di comando."""
    parser = argparse.ArgumentParser(description="CRISP-DM Master Orchestrator Pipeline")
    parser.add_argument("--pipeline", default="clustering")
    parser.add_argument("--dataset-key", default="ms_sec_inc_pre")
    parser.add_argument(
        "--phase",
        default="all",
        choices=["all", "2", "3", "4", "5"],
        help="Esegui il flusso completo ('all') o una fase specifica in modo isolato"
    )
    parser.add_argument("--run-id", default=None, help="Forza l'aggancio a un run-id specifico per debug")
    return parser.parse_args()


def find_latest_run_dir(base_dir: Path, explicit_id: str | None) -> Path:
    """Individua sul disco la cartella del run corretto basandosi sull'ordinamento cronologico di modifica."""
    if not base_dir.exists():
        base_dir.mkdir(parents=True, exist_ok=True)
    if explicit_id:
        target = base_dir / explicit_id
        if not target.exists():
            raise FileNotFoundError(f"❌ Il run-id esplicito '{explicit_id}' specificato non esiste.")
        return target

    # Ordinamento cronologico reale tramite il tempo di ultima modifica (st_mtime)
    all_runs = sorted([d for d in base_dir.iterdir() if d.is_dir()], key=lambda p: p.stat().st_mtime)
    if not all_runs:
        raise RuntimeError(f"❌ Nessun run precedente trovato in: {base_dir}. Eseguire prima il flusso '--phase all' o '--phase 2'.")
    return all_runs[-1]


def _ensure_runtime_config(ctx: RunContext) -> RunContext:
    """Garantisce la presenza del nodo 'runtime' nell'oggetto OmegaConf per evitare crash strutturali."""
    if ctx.config is not None:
        # Disattiva la modalità rigida (struct mode) di OmegaConf per consentire modifiche dinamiche
        OmegaConf.set_struct(ctx.config, False)
        if "runtime" not in ctx.config:
            log.info("🛡️ Iniezione difensiva del nodo 'runtime' mancante nella configurazione.")
            runtime_patch = OmegaConf.create({"runtime": {"random_seed": 7}})
            ctx.config = OmegaConf.merge(ctx.config, runtime_patch)
    return ctx


# =============================================================================
# ESECUTORI LOGICI DELLE FASI CRISP-DM
# =============================================================================

def execute_phase2(pipeline: str, dataset_key: str) -> RunContext:
    """Esegue la Fase 2 (Data Understanding) generando un nuovo identificativo di esecuzione (Run-ID)."""
    log.info("\n" + "="*50 + "\n⚙️ [FASE 2] AVVIO: DATA UNDERSTANDING (CREAZIONE NUOVO RUN)\n" + "="*50)

    # Crea fisicamente la nuova cartella del run temporizzato sul disco
    ctx = init_run_phase2(pipeline, dataset_key)
    ctx = _ensure_runtime_config(ctx)

    ctx = run_phase2_1(ctx)
    ctx = run_phase2_2(ctx)
    ctx = run_phase2_3(ctx)
    ctx = run_phase2_4(ctx)
    log.info(f"✅ FASE 2 COMPLETATA: Dati controllati e registrati nel run: {ctx.run_id}")
    return ctx


def execute_phase3(ctx: RunContext) -> RunContext:
    """Esegue la Fase 3 (Data Preparation) in memoria o ripristinando lo stato dall'ultimo run."""
    log.info("\n" + "="*50 + "\n⚙️ [FASE 3] AVVIO: DATA PREPARATION\n" + "="*50)
    ctx = _ensure_runtime_config(ctx)

    # Ripristino dello stato per l'esecuzione isolata dal disco
    if ctx.df_train is None:
        phase2_folder = PhaseDir.PHASE2.value
        p2_files = list((ctx.run_dir / phase2_folder).glob("*.parquet"))
        if not p2_files:
            raise FileNotFoundError(f"❌ Dati base della Fase 2 non trovati nella directory: {ctx.run_dir / phase2_folder}")
        ctx.df_train = pd.read_parquet(p2_files[0])
        log.info(f"📥 [Modalità Isolata] Caricati dati grezzi strutturati da: {p2_files[0].name}")

    ctx = run_step_3_1(ctx)
    ctx = run_step_3_2(ctx)
    ctx = run_step_3_3(ctx)
    ctx = run_step_3_5(ctx)
    log.info("✅ FASE 3 COMPLETATA: Ingegneria delle feature conclusa con successo.")
    return ctx


def execute_phase4(ctx: RunContext, limit: int) -> RunContext:
    """Esegue la Fase 4 (Modelling) applicando paracadute difensivi contro anomalie di memoria e tipi stringa."""
    log.info("\n" + "="*50 + "\n⚙️ [FASE 4] AVVIO: MODELLING (TRAINING & TUNING)\n" + "="*50)
    ctx = _ensure_runtime_config(ctx)

    phase3_folder = "phase3_data_preparation" if (ctx.run_dir / "phase3_data_preparation").exists() else PhaseDir.PHASE3.value
    p3_dir = ctx.run_dir / phase3_folder

    # Ripristino del dataframe se la fase viene lanciata in maniera indipendente
    if ctx.df_train is None or len(ctx.artifacts) == 0:
        path_x = next(p3_dir.glob("*_train_prepared.parquet"), None) or next(p3_dir.glob("*.parquet"), None)
        if not path_x:
            raise FileNotFoundError(f"❌ Impossibile trovare il dataset preparato della Fase 3 in: {p3_dir}")
        log.info(f"📥 [Modalità Isolata] Caricamento del dataset di addestramento: {path_x.name}")
        ctx.df_train = pd.read_parquet(path_x)

    # CAPA DIFENSIVA ANTIMUTAZIONE: Esclude colonne non numeriche (es. TruePositive, IncidentGrade) dall'input matematico
    df_prepared = ctx.df_train
    numeric_cols = df_prepared.select_dtypes(include=[np.number, bool]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in ["Id", "label", "IncidentGrade"]]

    X_full = df_prepared[feature_cols].to_numpy().astype(np.float32)
    ctx.artifacts["X_train"] = X_full[:limit] if limit and limit < len(X_full) else X_full
    log.info(f"   📊 Matrice X_train estratta correttamente. Dimensione: {ctx.artifacts['X_train'].shape}")

    # Caricamento di y_true (opzionale) per consentire il campionamento stratificato
    path_y = next(p3_dir.glob("*save_auxiliary_labels*"), None)
    if path_y and path_y.exists():
        y_df = pd.read_parquet(path_y)
        col = "label" if "label" in y_df.columns else y_df.columns[0]
        ctx.artifacts["y_true"] = y_df[col].values.astype(np.int32)[:limit]

    ctx = run_step_4_1(ctx)
    ctx = run_step_4_2(ctx)
    ctx = run_step_4_3(ctx)
    ctx = run_step_4_4(ctx)
    ctx = run_step_4_5(ctx)
    log.info("✅ FASE 4 COMPLETATA: Modelli addestrati, validati e serializzati.")
    return ctx


def execute_phase5(ctx: RunContext, limit: int) -> RunContext:
    """Esegue la Fase 5 (Evaluation) ricostruendo l'albero degli artefatti generati sul disco dalle fasi precedenti."""
    log.info("\n" + "="*50 + "\n⚙️ [FASE 5] AVVIO: EVALUATION & BUSINESS ALIGNMENT\n" + "="*50)
    ctx = _ensure_runtime_config(ctx)

    # Ricostruzione profonda dello stato della memoria per l'esecuzione completamente disaccoppiata
    if "best_models" not in ctx.artifacts or "cluster_labels" not in ctx.artifacts:
        log.info("📥 [Modalità Isolata] Ricostruzione dello stato logico dagli artefatti su disco...")
        phase3_folder = "phase3_data_preparation" if (ctx.run_dir / "phase3_data_preparation").exists() else PhaseDir.PHASE3.value
        p3_dir = ctx.run_dir / phase3_folder
        p4_dir = ctx.run_dir / PhaseDir.PHASE4.value

        path_x = next(p3_dir.glob("*_train_prepared.parquet"), None) or next(p3_dir.glob("*.parquet"), None)
        path_y = next(p3_dir.glob("*save_auxiliary_labels*"), None)
        path_lbl = next(p4_dir.glob("*cluster_assignments*.parquet"), None)
        path_mdl = next(p4_dir.glob("*.pkl"), None)

        if not all([path_x, path_lbl, path_mdl]):
            raise FileNotFoundError("❌ Artefatti su disco insufficienti. Eseguire la Fase 4 prima di lanciare la Fase 5 isolata.")

        # Rigenerazione controllata di X_train
        df_x = pd.read_parquet(path_x)
        numeric_cols = df_x.select_dtypes(include=[np.number, bool]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in ["Id", "label", "IncidentGrade"]]
        X_full = df_x[feature_cols].to_numpy().astype(np.float32)
        ctx.artifacts["X_train"] = X_full[:limit] if limit and limit < len(X_full) else X_full
        ctx.df_train = df_x

        if path_y and path_y.exists():
            y_df = pd.read_parquet(path_y)
            col = "label" if "label" in y_df.columns else y_df.columns[0]
            ctx.artifacts["y_true"] = y_df[col].values.astype(np.int32)[:limit]

        # Ripristino delle assegnazioni dei cluster e dei modelli pre-addestrati
        lbl_df = pd.read_parquet(path_lbl).iloc[:limit]
        ctx.artifacts["cluster_labels"] = {col: lbl_df[col].values for col in lbl_df.columns}

        models_dict = joblib.load(path_mdl)
        ctx.artifacts["best_models"] = {k: v for k, v in models_dict.items() if k in ctx.artifacts["cluster_labels"]}

    ctx = run_step_5_1(ctx)
    ctx = run_step_5_2(ctx)
    ctx = run_step_5_3(ctx)
    ctx = run_step_5_4(ctx)
    log.info("✅ FASE 5 COMPLETATA: Allineamento con gli obiettivi di business concluso.")
    return ctx


# =============================================================================
# COORDINATORE PRINCIPALE (MAIN ENTRYPOINT)
# =============================================================================

def main():
    args = parse_arguments()
    project_root = find_project_root()
    base_runs_dir = project_root / "outputs" / "runs" / args.pipeline / args.dataset_key

    log.info("=" * 80)
    log.info(f"🏁 CRISP-DM MASTER PIPELINE ORCHESTRATOR — MODALITÀ SELEZIONATA: {args.phase.upper()}")
    log.info("=" * 80)

    # 1. Caricamento iniziale della configurazione YAML
    try:
        cfg = YmlRepository.load_pipeline_config(args.pipeline)

        # Estrazione del limite massimo di righe per l'addestramento per prevenire l'esaurimento della RAM
        limit = cfg.phases.phase4_data_modeling.steps \
            .step_4_3_model_training.methods.model_training \
            .techniques.fit.params.get("max_training_rows", 30000)
    except Exception as e:
        log.error(f"❌ Errore critico durante il setup iniziale della configurazione: {e}")
        sys.exit(1)

    # 2. Instradamento dei flussi di esecuzione in base al parametro --phase passato
    try:
        if args.phase in ["all", "2"]:
            # La Fase 2 avvia l'orchestrazione creando la cartella fisica con il nuovo ID univoco
            ctx = execute_phase2(args.pipeline, args.dataset_key)
            if args.phase == "all":
                ctx = execute_phase3(ctx)
                ctx = execute_phase4(ctx, limit)
                ctx = execute_phase5(ctx, limit)
        else:
            # Le fasi isolate (3, 4, 5) ereditano dinamicamente l'ultimo run modificato sul disco
            run_dir = find_latest_run_dir(base_runs_dir, args.run_id)
            log.info(f"📁 Collegamento stabilito alla directory del Run esistente: {run_dir.name}")

            ctx = RunContext(
                config=cfg, run_dir=run_dir, df_train=None, df_test=None,
                run_id=run_dir.name, dataset_key=args.dataset_key
            )

            if args.phase == "3":
                ctx = execute_phase3(ctx)
            elif args.phase == "4":
                ctx = execute_phase4(ctx, limit)
            elif args.phase == "5":
                ctx = execute_phase5(ctx, limit)

    except Exception as e:
        log.error(f"\n❌ CRASH CRITICO RILEVATO DURANTE L'ORCHESTRAZIONE: {e}", exc_info=True)
        sys.exit(1)

    log.info("\n" + "=" * 80)
    log.info("🎉 PIPELINE ELABORATA CON SUCCESSO RISPETTANDO LO STANDARD CRISP-DM!")
    log.info("=" * 80)


if __name__ == "__main__":
    main()