#!/usr/bin/env python3
"""
CRISP-DM - Fase 5: Evaluation and Interpretation Runner
Script isolato che individua dinamicamente l'ultimo run sul disco, carica
sia i dati della Fase 3 che i modelli addestrati della Fase 4, ed esegue
l'interpretazione di business e l'allineamento degli obiettivi (passi da 5.1 a 5.4).

Esecuzione dalla radice del progetto:
    poetry run python src/crispdm/scripts/run_phase5.py --pipeline clustering --dataset-key ms_sec_inc_pre

    \Project_ML_Security_Incident_Prediction
    poetry run python src/crispdm/scripts/run_phase5.py
"""

from __future__ import annotations

import argparse
import sys
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

# Servizi ufficiali ed Enum dell'infrastruttura CRISP-DM
from crispdm.configuration.enum_registry_config import PhaseDir
from crispdm.common.path_service_common import find_project_root
from crispdm.common.context_facade_common import RunContext
from crispdm.configuration.yml_repository_config import YmlRepository

# Runner ufficiali della Fase 5
from crispdm.phase.phase5_evaluation_and_interpretation_phase import (
    run_step_5_1,
    run_step_5_2,
    run_step_5_3,
    run_step_5_4,
)

# Integrazione del modulo di logging personalizzato del progetto
from crispdm.common.logging_adapter_common import get_logger

# Inizializzazione del logger per la Fase 5
log = get_logger("run_phase5")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CRISP-DM Isolated Runner - Phase 5 (Evaluation)")
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
    log.info("🚀 AVVIO ESECUZIONE ISOLATA: CRISP-DM FASE 5 (EVALUATION)")
    log.info("=" * 80)

    try:
        project_root = find_project_root()
        base_runs_dir = project_root / "outputs" / "runs" / args.pipeline / args.dataset_key
        run_dir = find_target_run_dir(base_runs_dir, args.run_id)
        log.info(f"📁 Directory del Run: {run_dir}")
    except Exception as e:
        log.error(f"❌ Errore durante la risoluzione dei percorsi: {e}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 2. Configurazione e Patch di Sicurezza
    # -------------------------------------------------------------------------
    try:
        cfg = YmlRepository.load_pipeline_config(args.pipeline)
        if "runtime" not in cfg:
            cfg = OmegaConf.merge(cfg, OmegaConf.create({"runtime": {"random_seed": 7}}))
            log.debug("Nodo 'runtime' iniettato artificialmente per evitare crash.")

        limit = cfg.phases.phase4_data_modeling.steps \
            .step_4_3_model_training.methods.model_training \
            .techniques.fit.params.get("max_training_rows", 30000)
    except Exception as e:
        log.error(f"❌ Errore caricamento configurazione YAML: {e}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 3. Risoluzione dei Percorsi degli Artefatti Precedenti
    # -------------------------------------------------------------------------
    phase3_folder = "phase3_data_preparation" if (run_dir / "phase3_data_preparation").exists() else PhaseDir.PHASE3.value
    p3_dir = run_dir / phase3_folder
    p4_dir = run_dir / PhaseDir.PHASE4.value

    path_x = next(p3_dir.glob("*_train.parquet"), None)
    path_y = next(p3_dir.glob("*save_auxiliary_labels*"), None)

    # Cerchiamo dinamicamente i file della fase 4, gestendo nomi flessibili
    path_lbl = next(p4_dir.glob("*cluster_assignments*.parquet"), None)
    path_mdl = next(p4_dir.glob("*.pkl"), None)

    if not all([path_x, path_lbl, path_mdl]):
        log.error("❌ Impossibile trovare gli artefatti necessari della Fase 3 o Fase 4.")
        log.error(f"  - X_train trovato: {path_x is not None}")
        log.error(f"  - Cluster Labels trovate: {path_lbl is not None}")
        log.error(f"  - Modelli (.pkl) trovati: {path_mdl is not None}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 4. Caricamento in Memoria e Troncamento (Gestione RAM)
    # -------------------------------------------------------------------------
    # X_train
    log.info("📥 Caricamento X_train...")
    df_x = pd.read_parquet(path_x)

    #feature_cols = [c for c in df_x.columns if c not in ["Id", "label"]]

    # Selezioniamo SOLO le colonne numeriche ed escludiamo ID e Target
    numeric_cols = df_x.select_dtypes(include=[np.number, bool]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in ["Id", "label", "IncidentGrade"]]

    X_full = df_x[feature_cols].to_numpy().astype(np.float32)
    X_train = X_full[:limit] if limit and limit < len(X_full) else X_full
    #log.info(f"   📊 X_train shape finale: {X_train.shape}")
    log.info(f"   📊 X_train shape finale: {X_train.shape} (usando {len(feature_cols)} feature numeriche)")

    # y_true (Opzionale)
    y_true = None
    if path_y and path_y.exists():
        y_df = pd.read_parquet(path_y)
        col = "label" if "label" in y_df.columns else y_df.columns[0]
        y_true = y_df[col].values.astype(np.int32)[:limit]
        log.info(f"   ✅ y_true caricato: {len(y_true)} righe, colonna='{col}'")
    else:
        log.warning("   ⚠️ y_true non trovato — l'analisi avanzata potrebbe essere limitata.")

    # cluster_labels
    lbl_df = pd.read_parquet(path_lbl).iloc[:limit]
    log.info(f"   📋 Varianti di cluster rilevate: {list(lbl_df.columns)}")
    cluster_labels = {col: lbl_df[col].values for col in lbl_df.columns}

    # best_models
    models_dict = joblib.load(path_mdl)
    log.info(f"   📦 Modelli caricati dal file .pkl: {list(models_dict.keys())}")

    # Filtriamo i modelli mantenendo solo quelli che hanno etichette valide
    best_models = {k: v for k, v in models_dict.items() if k in cluster_labels}

    # -------------------------------------------------------------------------
    # 5. Esecuzione della Fase 5
    # -------------------------------------------------------------------------
    ctx = RunContext(
        config=cfg,
        run_dir=run_dir,
        df_train=df_x,  # Passiamo il dataframe originale per eventuali riferimenti di feature
        df_test=None,
        run_id=run_dir.name,
        dataset_key=args.dataset_key
    )

    ctx.artifacts = {
        "X_train": X_train,
        "y_true": y_true,
        "cluster_labels": cluster_labels,
        "best_models": best_models,
    }

    try:
        log.info("\n⚡ [Esecuzione Step 5.1] Business Metric Alignment...")
        ctx = run_step_5_1(ctx)

        log.info("\n⚡ [Esecuzione Step 5.2] Model Review...")
        ctx = run_step_5_2(ctx)

        log.info("\n⚡ [Esecuzione Step 5.3] Process Assessment...")
        ctx = run_step_5_3(ctx)

        log.info("\n⚡ [Esecuzione Step 5.4] Next Steps & Deployment Decision...")
        ctx = run_step_5_4(ctx)

    except Exception as e:
        log.error(f"❌ La pipeline è fallita durante l'elaborazione della Fase 5: {e}", exc_info=True)
        sys.exit(1)

    log.info("\n" + "=" * 80)
    log.info("🏁 FASE 5 COMPLETATA CON SUCCESSO!")
    log.info(f"📍 I report di interpretazione business si trovano in: {run_dir / PhaseDir.PHASE5.value}")
    log.info("=" * 80)


if __name__ == "__main__":
    main()