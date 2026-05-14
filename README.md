# Microsoft Security Incident Prediction

---

## 1. Overview

End-to-end Machine Learning framework for predicting Microsoft security incidents based
on the **GUIDE dataset** (~13M rows train / ~6M rows test). The codebase is structured
around **CRISP-DM** phases (phases 2–5), exposed through a **Facade API** and driven
entirely by **YAML configuration files** — no source code changes are required to switch
datasets, problem types, or hyperparameters.

The architecture enforces strict separation of concerns via established software design
and architectural patterns (Facade, Builder, Factory, Template Method, Registry,
Strategy, Artifact Repository, DAO, ...),others, making each module independently 
testable and replaceable.

---

## 2. Project Structure

```
Project_DS_Microsoft_Security_Incident_Prediction/
│
├── config/                                        # YAML-driven configuration — no hardcoded params
│   ├── datasets/
│   │   └── dataset_config.yml                     # Source: path, separator, encoding, chunking strategy
│   └── pipelines/
│       ├── base_pipeline_config.yml                # Phase 2 config
│       ├── clustering_pipeline_config.yml          # Phases 2→5 config 
│       ├── active_profile.yml          # 
│
├── data/                                          # Raw data only — never modified by the pipeline
│   └── raw/
│       ├── train/                                 # GUIDE_Train.csv (~13M rows, ~2 GB)
│       └── test/                                  # GUIDE_Test.csv  (~6M rows,  ~1 GB)
│
├── notebooks/                                     # Entry points — delegate exclusively to api/
│
├── output/                                        # All pipeline outputs(auto-generated, git-ignored)
│   └── runs/<task>/<dataset_key>/<timestamp>/
│
├── src/
│   └── crispdm/                                   # Core framework package
│       ├── api/                                   # ← Facade layer (only public interface)
│       ├── config/                                # ← Config subsystem: Load → DTO → Validate → Build
│       ├── core/                                  # ← Cross-cutting utilities
│       ├── data/                                  # ← Data ingestion & quality
│       ├── feature/                               # ← Feature engineering & splitting
│       ├── model/                                 # ← Training, evaluation & registry
│       ├── interpretation/                        # ← Explainability & error analysis
│       ├── pipeline/                              # ← Task-level orchestrators
│       ├── reporting/                             # ← Artifact persistence & plots
│       ├── phase/                                 # ← CRISP-DM phase runners
│       └── __init__.py
│
├── pyproject.toml                                 # Poetry dependency manifest
├── poetry.lock                                    # Locked dependency tree
├── README.md                                      # This file — technical architecture
```

## 5. Source Code — `src/crispdm/`

### `api/` — Facade Layer

Single public interface for all notebooks. Hides all internal complexity.

---

### `config/` — Configuration Subsystem

Implements a deterministic 4-step pipeline: **Load → Resolve → Validate → Build**.

---

### `core/` — Cross-Cutting Utilities

Stateless helpers with no business logic, usable from any layer.

---

### `data/` — Data Ingestion & Quality

Handles raw data acquisition and structural characterisation. **No transformation occurs
in this layer.**

---

### `feature/` — Feature Engineering & Splitting

Converts raw data into model-ready features. Applied sequentially in phase 3.

---

### `model/` — Training, Evaluation & Registry

Encapsulates all algorithm-level logic.

---

### `interpretation/` — Explainability & Error Analysis

Post-training analysis for model transparency and failure diagnosis.

---

### `pipeline/` — Task-Level Orchestrators

One runner per problem type. Each runner calls phases 2→5 in sequence. **No algorithm or
transformation logic lives here** — runners only coordinate phase execution.

**Algorithm rationale per classification model:**

---

### `reporting/` — Artifact Persistence

Implements the **artifact policy**: every run produces a reproducible, self-contained,
navigable folder. All outputs are PNG or JSON.

---

### `phase/` — CRISP-DM phase Runners

Each phase runner is driven by its section in the pipeline YAML. **phase 2 is fully
task-agnostic** — the same runner executes for all four problem types. phases 3–5 are
task-aware.

---

## 6. Configuration Layer

All pipeline behaviour is controlled through YAML. No source code changes are needed to
switch algorithms, hyperparameters, feature strategies, or output settings.

### Files

```
config/
├── datasets/
│   └── dataset_config.yml               # Dataset source, paths, csv_params, read_strategy per phase
└── pipelines/
    ├── clustering_pipeline_config.yml
    ├── classification_pipeline_config.yml
```

### Dataset config key fields (`dataset_config.yml`)

| Field                   | Value                               | Note                                                           |
|-------------------------|-------------------------------------|----------------------------------------------------------------|
| `paths.train`           | `data/raw/train/GUIDE_Train.csv`    | ~13M rows, ~2 GB                                               |
| `paths.test`            | `data/raw/test/GUIDE_Test.csv`      | ~6M rows, ~1 GB                                                |
| `csv_params.sep`        | `,`                                 | —                                                              |
| `csv_params.low_memory` | `false`                             | **Critical** — prevents silent dtype corruption on large files |
| `download_executor`     | `src/crispdm/data/download_data.py` | Acquisition module reference                                   |

---

## 7. Expected Outputs per phase & Model

All outputs follow the **artifact policy**: every item is a PNG or JSON file, persisted
under `out/runs/<task>/<dataset_key>/<timestamp>/` .


## 9. Output Structure

Every run produces a self-contained, reproducible snapshot under `out/`:

```
out/runs/<task>/<dataset_key>/<timestamp>/
├── logs/                           # Full execution log
├── phase2_data_understanding/
├── phase3_data_preparation/
├── phase4_data_modeling/
└── phase5_evaluation_and_interpretation/
```
