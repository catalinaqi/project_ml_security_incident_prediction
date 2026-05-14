# Microsoft Security Incident Prediction

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Poetry](https://img.shields.io/badge/Dependency_Manager-Poetry-orange.svg)](https://python-poetry.org/)
[![Methodology](https://img.shields.io/badge/Methodology-CRISP--DM-green.svg)](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)]()

---

## Table of Contents

1. [Overview](#1-overview)
2. [Project Structure](#2-project-structure)
3. [Notebooks](#3-notebooks)
4. [Execution Scenarios](#4-execution-scenarios)
    - [Scenario 1 — Preview Mode](#scenario-1--preview-mode-)
    - [Scenario 2 — Full Pipeline Run](#scenario-2--full-pipeline-run-)
5. [Source Code — src/crispdm/](#5-source-code--srccrispdm)
    - [api/](#api--facade-layer)
    - [config/](#config--configuration-subsystem)
    - [core/](#core--cross-cutting-utilities)
    - [data/](#data--data-ingestion--quality)
    - [feature/](#feature--feature-engineering--splitting)
    - [model/](#model--training-evaluation--registry)
    - [interpretation/](#interpretation--explainability--error-analysis)
    - [pipeline/](#pipeline--task-level-orchestrators)
    - [reporting/](#reporting--artifact-persistence)
    - [stage/](#stage--crisp-dm-stage-runners)
6. [Configuration Layer](#6-configuration-layer)
7. [Expected Outputs per Stage & Model](#7-expected-outputs-per-stage--model)
8. [Design & Architectural Patterns](#8-design--architectural-patterns)
9. [Output Structure](#9-output-structure)
10. [Reproducibility](#10-reproducibility)
11. [Getting Started](#11-getting-started)
12. [References](#12-references)

---

## 1. Overview

End-to-end Machine Learning framework for predicting Microsoft security incidents based
on the **GUIDE dataset** (~13M rows train / ~6M rows test). The codebase is structured
around **CRISP-DM** phases (Stages 2–5), exposed through a **Facade API** and driven
entirely by **YAML configuration files** — no source code changes are required to switch
datasets, problem types, or hyperparameters.

The framework implements four independent problem types over the same dataset, each with
its own notebook, pipeline config, and set of algorithms: **Clustering**, *
*Classification**, **Regression**, and **Time Series**.

The architecture enforces strict separation of concerns via established software design
and architectural patterns (Facade, Builder, Factory, Template Method, Registry,
Strategy, Artifact Repository, DAO, ...), making each module independently testable and
replaceable.

> For full dataset description, fields, and business context see [
`README_DATASET.md`](README_DATASET.md).

---

## 2. Project Structure

```
Project_DS_Microsoft_Security_Incident_Prediction/
│
├── config/                                        # YAML-driven configuration — no hardcoded params
│   ├── datasets/
│   │   └── dataset_config.yml                     # Source: path, separator, encoding, chunking strategy
│   └── pipelines/
│       ├── classification_pipeline_config.yml      # Stages 2→5 config for classification
│       ├── clustering_pipeline_config.yml          # Stages 2→5 config for clustering
│       ├── regression_pipeline_config.yml          # Stages 2→5 config for regression
│       └── timeseries_pipeline_config.yml          # Stages 2→5 config for time series
│
├── data/                                          # Raw data only — never modified by the pipeline
│   └── raw/
│       ├── train/                                 # GUIDE_Train.csv (~13M rows, ~2 GB)
│       └── test/                                  # GUIDE_Test.csv  (~6M rows,  ~1 GB)
│
├── notebooks/                                     # Entry points — delegate exclusively to api/
│
├── out/                                           # All pipeline outputs (auto-generated, git-ignored)
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
│       ├── stage/                                 # ← CRISP-DM stage runners
│       └── __init__.py
│
├── pyproject.toml                                 # Poetry dependency manifest
├── poetry.lock                                    # Locked dependency tree
├── README.md                                      # This file — technical architecture
├── README_POETRY.md                               # Guide to Poetry setup, dependency management, and troubleshooting
└── README_DATASET.md                              # Dataset description, fields, business context
```

---

## 3. Notebooks

The `notebooks/` folder contains **7 files** in three categories. The **only allowed
imports from `src/`** are from `src/crispdm/`.

### Pre-framework notebooks (`00_*`)

| File                           | Role                                                                                                                                                                                                                                                     | Key dependency          |
|--------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------|
| `00_data_pre_analyst.ipynb`    | **DuckDB pre-analysis** — SQL-based exploration of the raw CSV before engaging the framework. Inspects scale, field distributions, and cardinality without loading the full dataset into memory. Fully independent of `crispdm`                          | `duckdb`                |
| `00_preview_notebook.ipynb`    | **Framework preview** — calls `dryrun_facade_api.run_preview()` to execute Stage 2 in read-only mode. Returns column suggestions (`target_col`, `time_col`, `id_cols`). Dataset download is triggered **internally by the Facade**, not by this notebook | `api/dryrun_facade_api` |
| `00_test_loader_project.ipynb` | **Sanity test** — verifies that all environment variables, YAML paths, Poetry packages, and config keys load correctly before running any pipeline. Catches misconfiguration early                                                                       | `api/`, `config/`       |

### Pipeline runner notebooks (`01–04`)

Each notebook is dedicated to one `ProblemType`. All four share the same call contract:
they pass `pipeline_config_path`, `dataset_config_path`, `dataset_key`, and optional
`notebook_vars`. Dataset download is triggered **internally**.

| File                                   | Problem Type   | Business Question                                                                                    | Algorithms                                       |
|----------------------------------------|----------------|------------------------------------------------------------------------------------------------------|--------------------------------------------------|
| `01_run_clustering_notebook.ipynb`     | Clustering     | Do natural groups of incidents exist without using `IncidentGrade`? Do they align with TP / BP / FP? | K-Means, DBSCAN                                  |
| `02_run_classification_notebook.ipynb` | Classification | Is this incident TP, BP, or FP? → assist the SOC analyst in triage                                   | Random Forest *(primary)*, SVM, Naive Bayes, KNN |
| `03_run_regression_notebook.ipynb`     | Regression     | What will the `SuspicionLevel` of an incident be before the analyst reviews it?                      | Linear, Ridge/Lasso, SVR, KNN Regression         |
| `04_run_timeseries_notebook.ipynb`     | Time Series    | How many incidents will occur tomorrow / this week? Are there hourly or daily patterns?              | ARIMA, SARIMA                                    |

**Methodological execution order:** clustering (01) runs before classification (02) to
validate whether the data has natural structure. If K-Means or DBSCAN find 3 natural
groups that align with TP/BP/FP, it confirms that the supervised labels are reliable. If
they do not align, it signals significant label noise in `IncidentGrade`.

---

## 4. Execution Scenarios

The framework exposes **five entry points** (`src/crispdm/`). Notebooks interact *
*exclusively** with this layer.

---

### Scenario 1 — Preview Mode ✅ *(implemented)*

A lightweight, read-only Stage 2 run. No model is trained, no data is transformed.
Returns column suggestions to guide pipeline configuration.

**Entry point:** `00_preview_notebook.ipynb` → `api/dryrun_facade_api.run_preview()`

**Signature:**
`run_preview(pipeline_config_path, dataset_config_path, dataset_key, notebook_vars={})`

**Returns:** `PreviewResult(config, suggestions, audit_config_path, log_file)`

---

### Scenario 2 — Run Clustering

Executes the complete CRISP-DM lifecycle (Stages 2→5). Clustering = `ProblemType`.
**Entry point:** `01_run_clustering_notebook.ipynb`

### Scenario 2 — Run Classification

Executes the complete CRISP-DM lifecycle (Stages 2→5). Clustering = `ProblemType`.
**Entry point:** `01_run_classification_notebook.ipynb`

### Scenario 2 — Run Regression

Executes the complete CRISP-DM lifecycle (Stages 2→5). Clustering = `ProblemType`.
**Entry point:** `01_run_regression_notebook.ipynb`

### Scenario 2 — Run Time Series

Executes the complete CRISP-DM lifecycle (Stages 2→5). Clustering = `ProblemType`.
**Entry point:** `01_run_time_series_notebook.ipynb`

### Scenario 3 — Step to Step Execution

**Per-task configuration differences:**

| Aspect            | Clustering                      | Classification                          | Regression                         | Time Series                  |
|-------------------|---------------------------------|-----------------------------------------|------------------------------------|------------------------------|
| `target_col`      | ❌ absent                        | ✅ `IncidentGrade`                       | ✅ `SuspicionLevel`                 | ✅ aggregated series          |
| `time_col`        | ❌                               | ❌                                       | ❌                                  | ✅ `Timestamp`                |
| Split strategy    | No split — full fit             | Stratified holdout                      | Random holdout                     | Temporal / Walk-forward      |
| Temporal features | ❌                               | ❌                                       | ❌                                  | ✅ `temporal_service_feature` |
| Algorithms        | K-Means, DBSCAN                 | RandomForest, SVM, NaiveBayes, KNN      | Linear, Ridge/Lasso, SVR, KNN Reg. | ARIMA, SARIMA                |
| Primary metrics   | Silhouette, Davies-Bouldin, BIC | F1-weighted, Precision, Recall, ROC-AUC | RMSE, MAE, R²                      | RMSE, MAE, MAPE              |
| Error analysis    | Cluster profiling               | Confusion matrix slicing                | Residual analysis                  | Forecast error by horizon    |

---

## 5. Source Code — `src/crispdm/`

### `api/` — Facade Layer

Single public interface for all notebooks. Hides all internal complexity.

| File                         | Role                                                                                                                                                                                         | Pattern                      |
|------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------|
| `dryrun_facade_api.py`       | Orchestrates the preview flow: calls `build_preview_config()`, triggers `download_microsoft_dataset()` internally, initialises logging, runs `run_stage2_preview()`, returns `PreviewResult` | Facade + Application Service |
| `execution_facade_api.py.py` | Orchestrates the full pipeline: calls `build_run_config()`, triggers download internally, creates run directory, seeds RNG, routes to the correct pipeline runner by `ProblemType`           | Facade + Application Service |

**`PreviewResult` fields:** `config` (resolved `ProjectConfig`), `suggestions` (
target/time/id column candidates), `audit_config_path` (YAML snapshot path),
`log_file` (execution log path).

---

### `config/` — Configuration Subsystem

Implements a deterministic 4-step pipeline: **Load → Resolve → Validate → Build**.

| File                           | Role                                                                                                                                                                                                                                                                                           | Pattern                      |
|--------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------|
| `load_loader_config.py`        | Reads pipeline and dataset YAML files; resolves `${var}` placeholders using `runtime_vars` dict assembled by `build_factory_config`                                                                                                                                                            | Loader + Interpreter         |
| `schema_dto_config.py`         | Immutable typed dataclasses: `ProjectConfig`, `PipelineConfig`, `RuntimeConfig`, `StagesConfig`, per-stage DTOs. Single source of truth after `from_dict()`                                                                                                                                    | DTO / Value Object           |
| `validate_validator_config.py` | Two validation modes — `preview` (relaxed: target optional) and `run` (strict: required fields enforced by `ProblemType`, enum membership checked). Returns structured `ValidationResult`                                                                                                      | Specification + Validator    |
| `build_factory_config.py`      | `build_preview_config()` and `build_run_config()`. Assembles `ProjectConfig` from pipeline YAML + dataset defaults (`apply_dataset_defaults()`) + notebook overrides. Saves audit snapshot                                                                                                     | Builder + Factory            |
| `enums_utils_config.py`        | Canonical enums: `ProblemType` (clustering/classification/regression/timeseries), `ReadMode` (full/sample/chunked), `LogLevel` (DEBUG/INFO/WARNING/ERROR), `FeatureSelectionMode` (AUTO/INCLUDE/EXCLUDE), `CsvSourceType`. Includes `normalize_*` helpers for fail-fast string→enum conversion | Typed Configuration Boundary |

**Validation rules by mode:**

| Rule                                                    | Preview | Run   |
|---------------------------------------------------------|---------|-------|
| `clustering` → `target_col` must be absent              | warning | error |
| `classification` / `regression` → `target_col` required | skip    | error |
| `timeseries` → `target_col` + `time_col` required       | skip    | error |
| Method and model names must exist in enums              | warning | error |

---

### `core/` — Cross-Cutting Utilities

Stateless helpers with no business logic, usable from any layer.

| File                    | Role                                                                                                                                                                             |
|-------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `seeds_utils_core.py`   | Sets global seeds for `numpy`, `random`, and `sklearn` at pipeline start — guarantees run-to-run reproducibility                                                                 |
| `logging_utils_core.py` | `get_logger()` — module-level logger factory. `build_log_file()` — unique log path per execution. `init_logging()` — initialises root logger once per run (console + `logs.txt`) |
| `helpers_utils_core.py` | Utilities for timestamps, run-directory naming, path resolution, and string formatting                                                                                           |

---

### `data/` — Data Ingestion & Quality

Handles raw data acquisition and structural characterisation. **No transformation occurs
in this layer.**

| File                          | Role                                                                                                                                                                                                                                                                | Pattern          |
|-------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------|
| `download_data.py`            | Downloads `GUIDE_Train.csv` and `GUIDE_Test.csv` from Kaggle Hub via `kagglehub`. Skips if files already exist (`force=False`). Resolves project root via `Path(__file__).parents[3]`. **Called internally by both Facade functions — never directly by notebooks** | DAO / Repository |
| `load_utils_data.py`          | Loads CSV as full DataFrame, chunk iterator, or random sample. Controlled by `read_strategy` in YAML. Applies `csv_params` including `low_memory=False` — critical to prevent silent dtype corruption on large GUIDE files                                          | —                |
| `profiling_service_data.py`   | Computes schema profile (dtypes, null %, cardinality), identifies time-column candidates, flags high-null columns. Runs Stage 2 preview and returns suggestions dict                                                                                                | —                |
| `quality_rules_utils_data.py` | Loads and applies external YAML rulesets: range validations, logic constraints, business KPI rules. Returns structured violation report used by Stage 2                                                                                                             | —                |

---

### `feature/` — Feature Engineering & Splitting

Converts raw data into model-ready features. Applied sequentially in Stage 3.

| File                            | Role                                                                                                                                                                                                         |
|---------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `cleaning_service_feature.py`   | Missing value treatment (imputation / drop), outlier handling (IQR / Z-score), duplicate removal, categorical noise correction                                                                               |
| `selection_service_feature.py`  | Feature selection by `FeatureSelectionMode`: AUTO (all except id/target/time), INCLUDE (explicit list), EXCLUDE (all except listed). Removes constant and near-constant features. Applies correlation filter |
| `transforms_service_feature.py` | Encoding (OneHot, Ordinal, Target), scaling (Standard, MinMax, Robust), feature generation                                                                                                                   |
| `temporal_service_feature.py`   | Time-based feature extraction: lags, rolling statistics, calendar decomposition (hour, weekday, day of month). **Used only by the time series pipeline**                                                     |
| `split_service_feature.py`      | Train/Validation/Test splitting: stratified holdout (classification), random holdout (regression), temporal holdout / walk-forward cross-validation (time series), full fit with no split (clustering)       |

---

### `model/` — Training, Evaluation & Registry

Encapsulates all algorithm-level logic.

| File                         | Role                                                                                                                                                                                                                                                                 | Pattern  |
|------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|
| `registry_registry_model.py` | Maps string identifiers to model classes by `ProblemType`. Clustering: `kmeans`, `dbscan`. Classification: `random_forest`, `svm`, `naive_bayes`, `knn_classification`. Regression: `linear`, `ridge_lasso`, `svr`, `knn_regression`. Time Series: `arima`, `sarima` | Registry |
| `train_service_model.py`     | Trains models with cross-validation and hyperparameter tuning (GridSearch / RandomSearch). Tuning strategy and parameter grids are defined in the pipeline YAML                                                                                                      | Strategy |
| `evaluate_service_model.py`  | Computes task-specific metrics. Classification: F1-weighted, Precision, Recall, ROC-AUC. Regression: RMSE, MAE, R². Clustering: Silhouette, Davies-Bouldin, BIC. Time Series: RMSE, MAE, MAPE                                                                        | Strategy |

---

### `interpretation/` — Explainability & Error Analysis

Post-training analysis for model transparency and failure diagnosis.

| File                                       | Role                                                                                                                                                                                     |
|--------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `explain_service_interpretation.py`        | Global and local explainability: feature importances, model coefficients, permutation importance, SHAP values, Partial Dependence Plots (PDP)                                            |
| `error_analysis_service_interpretation.py` | Model failure diagnostics: confusion matrix slicing (classification), residual analysis by segment (regression), cluster profiling (clustering), forecast error by horizon (time series) |

---

### `pipeline/` — Task-Level Orchestrators

One runner per problem type. Each runner calls Stages 2→5 in sequence. **No algorithm or
transformation logic lives here** — runners only coordinate stage execution.

| File                                | Problem Type              | Target                                            | Algorithms                                       |
|-------------------------------------|---------------------------|---------------------------------------------------|--------------------------------------------------|
| `clustering_runner_pipeline.py`     | Unsupervised clustering   | None                                              | K-Means, DBSCAN                                  |
| `classification_runner_pipeline.py` | Multiclass classification | `IncidentGrade` (TP / BP / FP)                    | Random Forest *(primary)*, SVM, Naive Bayes, KNN |
| `regression_runner_pipeline.py`     | Continuous regression     | `SuspicionLevel`                                  | Linear, Ridge/Lasso, SVR, KNN Regression         |
| `timeseries_runner_pipeline.py`     | Time-ordered forecasting  | Incident count / hour aggregated from `Timestamp` | ARIMA, SARIMA                                    |

**Algorithm rationale per classification model:**

| Algorithm          | Specific value                                                     |
|--------------------|--------------------------------------------------------------------|
| Random Forest      | Handles high-dimensional categorical features well — primary model |
| SVM                | Robust decision boundary in high-dimensional space                 |
| Naive Bayes        | Fast baseline, well-suited to categorical features                 |
| KNN Classification | Similarity-based: do similar incidents share the same grade?       |

---

### `reporting/` — Artifact Persistence

Implements the **artifact policy**: every run produces a reproducible, self-contained,
navigable folder. All outputs are PNG or JSON.

| File                             | Role                                                                                                                                                                                                                   | Pattern                                             |
|----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------|
| `artifacts_service_reporting.py` | `create_run_dir(task, dataset_key, timestamp)` — creates the full run directory tree. `stage_dir()`, `save_table_png()`, `save_figure()`, `save_json()`, `save_metrics()`, `save_model_pickle()` — used by every stage | Artifact Repository + Convention over Configuration |
| `audit_service_reporting.py`     | `save_config_used()` — writes `config_used.yml` at the start of every run (called by `build_factory_config`). Also generates lineage summary and model card metadata at Stage 5                                        | —                                                   |
| `plots_utils_reporting.py`       | Shared plotting helpers used by all stages: consistent DPI, axis formatting, `plot_missingness_top()`, `plot_numeric_hist()`                                                                                           | —                                                   |

---

### `stage/` — CRISP-DM Stage Runners

Each stage runner is driven by its section in the pipeline YAML. **Stage 2 is fully
task-agnostic** — the same runner executes for all four problem types. Stages 3–5 are
task-aware.

| File                                   | CRISP-DM Phase     | Task-agnostic | Key services                                                                                                                                                  |
|----------------------------------------|--------------------|---------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `stage2_understanding_runner_stage.py` | Data Understanding | ✅ Yes         | `load_utils_data`, steps 2.1→2.4, `artifacts_service_reporting`                                                                                               |
| `stage3_preparation_runner_stage.py`   | Data Preparation   | Partially     | `cleaning_service_feature`, `selection_service_feature`, `transforms_service_feature`, `temporal_service_feature` *(TS only)*, `split_service_feature`        |
| `stage4_modeling_runner_stage.py`      | Modeling           | ❌ No          | `registry_registry_model`, `train_service_model`                                                                                                              |
| `stage5_evaluation_runner_stage.py`    | Evaluation         | ❌ No          | `evaluate_service_model`, `explain_service_interpretation`, `error_analysis_service_interpretation`, `audit_service_reporting`, `artifacts_service_reporting` |

**Stage 2 sub-steps (implemented):**

| Sub-step                    | Techniques / methods                                                                                         |
|-----------------------------|--------------------------------------------------------------------------------------------------------------|
| 2.1 Data Acquisition        | `load_csv_by_strategy` — full / sample / chunked mode per `read_strategy` in YAML                            |
| 2.2 Describe Data           | `describe()`, `min_max_mean_std`, `schema_inspection` (dtype/null/unique), `cardinality_count`, `null_count` |
| 2.3 Data Quality Assessment | `missing_analysis`, `duplicate_detection`, `range_validation`, `inconsistency_checks`, `business_kpi_rules`  |
| 2.4 EDA                     | `histograms` per numeric column (configurable `max_columns`, `bins`)                                         |

---

## 6. Configuration Layer

All pipeline behaviour is controlled through YAML. No source code changes are needed to
switch algorithms, hyperparameters, feature strategies, or output settings.

### Files

```
config/
├── datasets/
│   └── dataset_config.yml               # Dataset source, paths, csv_params, read_strategy per stage
└── pipelines/
    ├── clustering_pipeline_config.yml
    ├── classification_pipeline_config.yml
    ├── regression_pipeline_config.yml
    └── timeseries_pipeline_config.yml
```

### Pipeline YAML structure

All four pipeline configs follow the same three-section schema:

**`pipeline`** — identity and runtime-injected variables:

```
pipeline:
  name:      <pipeline_name>
  task:      clustering | classification | regression | timeseries
  objective: <free-text description>
  variables:
    dataset_path:  "${dataset_path}"    # resolved from dataset_config or notebook_vars
    id_cols:       "${id_cols}"         # columns to ignore during training
```

**`runtime`** — global execution settings:

```
runtime:
  random_seed:          42
  output_root:          "${output_root}"   # default: "out"
  overwrite_artifacts:  true
  log_level:            "DEBUG"            # DEBUG | INFO | WARNING | ERROR | CRITICAL
```

**`stages`** — per-stage CRISP-DM configuration:

```
stages:
  phase2_data_understanding:   <5 keys>   # steps 2.1→2.4, output_policy, dataset_input
  phase3_data_preparation:     <4 keys>   # cleaning, selection, transforms, split
  phase4_data_modeling:        <4 keys>   # algorithms, tuning strategy, CV
  stage5_evaluation:      <keys>     # metrics, explainability, error analysis config
```

### Runtime variable resolution

Variables in `${...}` are resolved by `load_loader_config.load_and_resolve()` using the
`runtime_vars` dict built by `build_factory_config`:

| Variable          | Source                                                             | Required in run mode                       |
|-------------------|--------------------------------------------------------------------|--------------------------------------------|
| `${dataset_path}` | `dataset_config.yml` `paths.train` or `notebook_vars.dataset_path` | ✅ always                                   |
| `${target_col}`   | `notebook_vars.target_col`                                         | ✅ classification / regression / timeseries |
| `${time_col}`     | `notebook_vars.time_col`                                           | ✅ timeseries                               |
| `${id_cols}`      | `notebook_vars.id_cols`                                            | optional                                   |
| `${output_root}`  | `notebook_vars.output_root` or default `"out"`                     | optional                                   |

### Dataset config key fields (`dataset_config.yml`)

| Field                   | Value                               | Note                                                           |
|-------------------------|-------------------------------------|----------------------------------------------------------------|
| `paths.train`           | `data/raw/train/GUIDE_Train.csv`    | ~13M rows, ~2 GB                                               |
| `paths.test`            | `data/raw/test/GUIDE_Test.csv`      | ~6M rows, ~1 GB                                                |
| `csv_params.sep`        | `,`                                 | —                                                              |
| `csv_params.low_memory` | `false`                             | **Critical** — prevents silent dtype corruption on large files |
| `download_executor`     | `src/crispdm/data/download_data.py` | Acquisition module reference                                   |

---

## 7. Expected Outputs per Stage & Model

All outputs follow the **artifact policy**: every item is a PNG or JSON file, persisted
under `out/runs/<task>/<dataset_key>/<timestamp>/` by `artifacts_service_reporting.py`.

### Stage 2 — Data Understanding *(task-agnostic — identical for all 4 pipelines)*

| Type       | Artifact                       | Content                                                                     |
|------------|--------------------------------|-----------------------------------------------------------------------------|
| Table PNG  | `describe.png`                 | Full `df.describe(include='all')` transposed                                |
| Table PNG  | `min_max_mean_std.png`         | Min, max, mean, std per numeric column                                      |
| Table PNG  | `schema_dtype_null_unique.png` | Dtype, null count, null %, unique count per column                          |
| Table PNG  | `cardinality_top.png`          | Top-N columns by unique value count                                         |
| Table PNG  | `null_count.png`               | Null count and null % sorted descending                                     |
| Table PNG  | `duplicates.png`               | Duplicate row count, percentage, subset used                                |
| Table PNG  | `quality_rules_violations.png` | Rule name, column, condition, violation row count                           |
| Figure PNG | `missingness_top.png`          | Bar chart — top-N columns by null %                                         |
| Figure PNG | `hist_<col>.png`               | Histogram per numeric column (up to `max_columns`)                          |
| JSON       | `stage_report.json`            | rows, cols, dataset_path, read_mode, total_violations, total_rules_violated |

### Stage 3 — Data Preparation *(task-aware)*

| Type       | Artifact                 | Content                                                      |
|------------|--------------------------|--------------------------------------------------------------|
| Table PNG  | `cleaning_summary.png`   | Imputed / dropped counts per column and strategy             |
| Table PNG  | `selected_features.png`  | Final feature list after selection, with selection mode      |
| Table PNG  | `split_sizes.png`        | Train / validation / test row counts and percentages         |
| Figure PNG | `correlation_matrix.png` | Heatmap of numeric feature correlations                      |
| JSON       | `stage_report.json`      | Feature count before/after, split strategy used, split sizes |

### Stage 4 — Modeling *(task-specific per ProblemType)*

| Type       | Artifact                  | Content                                                        |
|------------|---------------------------|----------------------------------------------------------------|
| Table PNG  | `cv_results.png`          | Cross-validation scores per fold per algorithm                 |
| Table PNG  | `best_params.png`         | Best hyperparameters from GridSearch / RandomSearch            |
| Table PNG  | `feature_importances.png` | Top features ranked by importance (tree-based / linear models) |
| Figure PNG | `learning_curve.png`      | Train vs. validation score by training size                    |
| Model file | `models/model.pkl`        | Serialized best model                                          |
| JSON       | `stage_report.json`       | Best algorithm name, best params, CV score, fit time           |

### Stage 5 — Evaluation *(task-specific outputs per ProblemType)*

**Clustering (K-Means, DBSCAN):**

| Type       | Artifact                 | Content                                                          |
|------------|--------------------------|------------------------------------------------------------------|
| Table PNG  | `metrics_clustering.png` | Silhouette score, Davies-Bouldin index, BIC (DBSCAN), n_clusters |
| Table PNG  | `cluster_profiles.png`   | Mean feature values per cluster — alignment with TP/BP/FP        |
| Figure PNG | `pca_clusters.png`       | 2D PCA projection coloured by cluster assignment                 |
| Figure PNG | `shap_summary.png`       | Feature contribution to cluster separation                       |
| JSON       | `metrics.json`           | Silhouette, Davies-Bouldin, n_clusters, noise_ratio (DBSCAN)     |

**Classification (Random Forest, SVM, Naive Bayes, KNN):**

| Type       | Artifact                     | Content                                                |
|------------|------------------------------|--------------------------------------------------------|
| Table PNG  | `metrics_classification.png` | F1-weighted, Precision, Recall, ROC-AUC per class      |
| Table PNG  | `confusion_matrix.png`       | TP / BP / FP confusion matrix                          |
| Table PNG  | `error_by_segment.png`       | Misclassification rate sliced by key feature segments  |
| Figure PNG | `roc_curve.png`              | ROC curve per class (one-vs-rest)                      |
| Figure PNG | `shap_summary.png`           | SHAP beeswarm — global feature importance              |
| Figure PNG | `shap_waterfall_<n>.png`     | SHAP waterfall — local explanation per selected sample |
| JSON       | `metrics.json`               | F1-weighted, Precision, Recall, ROC-AUC                |

**Regression (Linear, Ridge/Lasso, SVR, KNN Regression):**

| Type       | Artifact                   | Content                                          |
|------------|----------------------------|--------------------------------------------------|
| Table PNG  | `metrics_regression.png`   | RMSE, MAE, R² on test set                        |
| Table PNG  | `residuals_by_segment.png` | Mean residual sliced by `Category`, `EntityType` |
| Figure PNG | `residuals_plot.png`       | Residuals vs. predicted `SuspicionLevel`         |
| Figure PNG | `actual_vs_predicted.png`  | Scatter: actual vs. predicted `SuspicionLevel`   |
| Figure PNG | `shap_summary.png`         | SHAP global feature importance                   |
| Figure PNG | `pdp_<feature>.png`        | Partial Dependence Plot per key feature          |
| JSON       | `metrics.json`             | RMSE, MAE, R²                                    |

**Time Series (ARIMA, SARIMA):**

| Type       | Artifact                        | Content                                 |
|------------|---------------------------------|-----------------------------------------|
| Table PNG  | `metrics_timeseries.png`        | RMSE, MAE, MAPE per forecast horizon    |
| Table PNG  | `forecast_error_by_horizon.png` | Error breakdown by 1h / 6h / 24h ahead  |
| Figure PNG | `forecast_vs_actual.png`        | Forecast line vs. actual incident count |
| Figure PNG | `hourly_patterns.png`           | Average incident count by hour of day   |
| Figure PNG | `daily_patterns.png`            | Average incident count by day of week   |
| JSON       | `metrics.json`                  | RMSE, MAE, MAPE, forecast_horizon       |

> **Time series limitation:** the GUIDE dataset covers approximately 2 weeks of data.
> ARIMA/SARIMA require longer history for reliable seasonal decomposition. The analysis
> is
> valid for detecting intra-day and intra-week patterns (known attack activity peaks on
> Monday mornings; lower activity on Sundays).

---

## 8. Design & Architectural Patterns

### GoF Design Patterns

| Pattern                       | Location                                                                          | Purpose                                                                                             |
|-------------------------------|-----------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| **Facade**                    | `api/dryrun_facade_api.py`, `api/execution_facade_api.py.py`                      | Single entry point for notebooks — encapsulates all pipeline internals                              |
| **DTO / Value Object**        | `config/schema_dto_config.py`                                                     | Immutable typed config objects (`ProjectConfig`, `PipelineConfig`, `RuntimeConfig`, per-stage DTOs) |
| **Builder**                   | `config/build_factory_config.py` — `build_preview_config()`, `build_run_config()` | Assembles `ProjectConfig` from multiple YAML sources + notebook overrides in a defined sequence     |
| **Factory**                   | `config/build_factory_config.py` — routing by `ProblemType`                       | Selects validation mode, default technique families, and pipeline runner per task type              |
| **Specification + Validator** | `config/validate_validator_config.py`                                             | Composable constraint rules with two modes: `preview` (relaxed) and `run` (strict)                  |
| **Registry**                  | `model/registry_registry_model.py`                                                | Decouples algorithm name strings from class instantiation per `ProblemType`                         |
| **Strategy**                  | `model/train_service_model.py`, `model/evaluate_service_model.py`                 | Swappable training (GridSearch / RandomSearch) and metric computation per task type                 |

### Architectural Patterns

| Pattern                                                 | Location                                        | Purpose                                                                                                                                                                                         |
|---------------------------------------------------------|-------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Application Service / Orchestrator (thin)**           | `api/`                                          | Facade functions contain no business logic — they only coordinate calls between the config, data, and stage layers                                                                              |
| **Stage Runner**                                        | `stage/`                                        | Each runner has a fixed execution structure driven by YAML sub-steps. Stage 2 is fully task-agnostic; Stages 3–5 are task-aware                                                                 |
| **Artifact Repository (Convention over Configuration)** | `reporting/artifacts_service_reporting.py`      | Fixed, auto-generated directory structure per run (`create_run_dir`). Every stage writes to its own `{figures/, tables_png/, stage_report.json}` subdirectory — no per-run configuration needed |
| **DAO / Repository**                                    | `data/download_data.py`                         | Abstracts the Kaggle Hub data source. Internal to the Facade — no caller interacts with Kaggle directly                                                                                         |
| **Single Source of Truth (typed config)**               | `config/schema_dto_config.py` — `ProjectConfig` | Once built by `build_factory_config`, the typed DTO is the only config object passed through the entire pipeline. Raw dict access ends after `from_dict()`                                      |
| **Pipeline Pattern**                                    | `pipeline/` runners                             | Each runner defines the ordered composition of stage calls for its `ProblemType`. Stages are reusable; runners define the sequence                                                              |

---

## 9. Output Structure

Every run produces a self-contained, reproducible snapshot under `out/`:

```
out/runs/<task>/<dataset_key>/<timestamp>/
├── logs.log                           # Full execution log
├── metrics.json                       # All computed metrics
├── models/
│   └── model.pkl
├── phase2_data_understanding/
│   ├── figures/
│   │   ├── missingness_top.png
│   │   └── hist_<col>.png
│   ├── tables_png/
│   │   ├── describe.png
│   │   ├── schema_dtype_null_unique.png
│   │   ├── cardinality_top.png
│   │   ├── null_count.png
│   │   ├── min_max_mean_std.png
│   │   ├── duplicates.png
│   │   └── quality_rules_violations.png
│   └── stage_report.json
├── phase3_data_preparation/
│   ├── figures/ ...
│   ├── tables_png/ ...
│   └── stage_report.json
├── phase4_data_modeling/
│   ├── figures/ ...
│   ├── tables_png/ ...
│   └── stage_report.json
└── stage5_evaluation/
    ├── figures/ ...
    ├── tables_png/ ...
    └── stage_report.json
```

---

## 10. References

| Reference                                    | URL / Path                                                                       |
|----------------------------------------------|----------------------------------------------------------------------------------|
| CRISP-DM methodology                         | https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining    |
| GUIDE Dataset — Kaggle                       | https://www.kaggle.com/datasets/microsoft/microsoft-security-incident-prediction |
| kagglehub — download library                 | https://github.com/Kaggle/kagglehub                                              |
| Poetry — dependency manager                  | https://python-poetry.org/                                                       |
| Dataset technical description                | [`README_DATASET.md`](README_DATASET.md)                                         |
| Gang of Four — Design Patterns               | https://en.wikipedia.org/wiki/Design_Patterns                                    |
| Fowler — Enterprise Application Architecture | https://martinfowler.com/books/eaa.html                                          |

---

*Developed as a modular, production-oriented Data Science framework following CRISP-DM
and software engineering best practices.*