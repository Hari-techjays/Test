# Sentiment Analysis MLOps Pipeline — Design Document

**Date:** 2026-03-03
**Status:** Approved
**Purpose:** Portfolio / assessment project demonstrating end-to-end MLOps

## Overview

An end-to-end MLOps pipeline for training, deploying, and monitoring a sentiment analysis model on retail customer reviews. The system ingests Amazon Product Reviews, fine-tunes a DistilBERT classifier, serves predictions via a FastAPI API on GCP Cloud Run, and monitors model performance and data drift.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11 |
| DL Framework | PyTorch + Hugging Face Transformers |
| Model | DistilBERT (fine-tuned) |
| HP Tuning | Optuna |
| Data Pipeline | pandas + custom scripts |
| Data Versioning | DVC + GCS |
| API Framework | FastAPI + Uvicorn |
| Experiment Tracking | MLflow |
| Containerization | Docker (multi-stage) |
| Orchestration | Docker Compose (local), Cloud Run (prod) |
| CI/CD | GitHub Actions |
| IaC | Terraform |
| Cloud | GCP |
| Monitoring | GCP Cloud Monitoring + MLflow |
| Testing | pytest |
| Linting | ruff |

## Architecture

### Approach: Monorepo, Single-Pipeline

Single Git repository with clear directory structure. One unified pipeline orchestrated via Makefile, with separate Docker images for training and inference.

## Component Designs

### 1. Data Pipeline

**Ingestion** (`src/data/ingest.py`):
- Downloads Amazon Product Reviews dataset (Electronics category, ~50K reviews)
- Raw data stored as JSON/CSV in `data/raw/`

**Preprocessing** (`src/data/preprocess.py`):
1. Load raw reviews with pandas
2. Clean text: lowercase, remove HTML tags, handle special characters
3. Map star ratings to sentiment labels (1-2 = negative, 3 = neutral, 4-5 = positive)
4. Train/validation/test split (80/10/10)
5. Tokenization handled at training time by Hugging Face tokenizer
6. Save processed splits as Parquet files in `data/processed/`

**Data versioning:** DVC tracks `data/` directory with GCS remote storage.

### 2. Model Training & Optimization

**Training** (`src/model/train.py`):
- Fine-tunes `distilbert-base-uncased` using PyTorch + Hugging Face Trainer API
- Loads processed Parquet data, tokenizes with DistilBertTokenizer
- Cross-entropy loss, AdamW optimizer

**Hyperparameter tuning** (`src/model/tune.py`):
- Optuna study over: learning rate (1e-5 to 5e-5), batch size (16, 32), warmup steps, epochs (2-5)
- MedianPruner for early stopping of unpromising trials
- Results logged to MLflow

**Experiment tracking:**
- MLflow tracks all runs: metrics (accuracy, F1, loss per epoch), parameters, artifacts
- Best model registered in MLflow Model Registry

**Evaluation** (`src/model/evaluate.py`):
- Runs on held-out test split
- Outputs classification report, confusion matrix
- Saves metrics as JSON for CI comparison

### 3. Inference Service

**FastAPI app** (`src/serving/app.py`):
- `POST /predict` — accepts review text, returns sentiment label + confidence scores
- `POST /predict/batch` — batch prediction for multiple reviews
- `GET /health` — health check for Cloud Run
- `GET /metrics` — Prometheus-format metrics endpoint

**Model loading:** Loads best model from MLflow artifact store on startup (or local path in Docker).

**Validation:** Pydantic request/response schemas (`src/serving/schemas.py`).

### 4. Containerization

- `docker/Dockerfile.train` — Python 3.11, training dependencies, runs train/tune scripts
- `docker/Dockerfile.serve` — Python 3.11-slim, inference dependencies only, Uvicorn + FastAPI
- `docker-compose.yml` — Local dev: training, inference, MLflow server (SQLite backend)
- Multi-stage builds for lean images

### 5. GCP Infrastructure (Terraform)

Resources in `terraform/`:
- **Artifact Registry** — Docker image repository
- **Cloud Run** — Inference service (min 0, max 5 instances, 2GB memory, 2 vCPU)
- **Cloud Storage** — Buckets for raw data, processed data, model artifacts, MLflow artifacts
- **Cloud Monitoring** — Custom dashboards
- **IAM** — Service accounts with least-privilege roles
- **Secret Manager** — Credentials storage

### 6. CI/CD (GitHub Actions)

**`ci.yml`** — On every PR:
- Lint (ruff)
- Type check (mypy)
- Unit tests (pytest)
- Build Docker images (smoke test)

**`cd-staging.yml`** — On merge to main:
- Build + push images to Artifact Registry
- Deploy to Cloud Run (staging)
- Run integration tests against staging
- Report metrics

**`train.yml`** — Manual trigger or on data/model code changes:
- Run training pipeline
- Log to MLflow
- Update model artifact if metrics improve

**Branching:** main (stable) + feature branches + PR-based merging.

### 7. Monitoring

**MLflow:** Experiment tracking, model registry, metric comparison.

**GCP Cloud Monitoring dashboard:**
- Request latency (p50, p95, p99)
- Request throughput (RPM)
- Error rate
- CPU/memory utilization

**Data drift detection** (`src/monitoring/drift.py`):
- Periodic comparison of incoming reviews vs. training data
- Statistical tests (KS test on text length, vocabulary overlap)
- Alerts via Cloud Monitoring on threshold breach

**Confidence tracking:** Log prediction confidence distributions, flag average confidence drops.

## Project Structure

```
hf-sentiment-mlops/
├── .github/workflows/
│   ├── ci.yml
│   ├── cd-staging.yml
│   └── train.yml
├── data/
│   ├── raw/
│   └── processed/
├── docker/
│   ├── Dockerfile.train
│   └── Dockerfile.serve
├── docs/plans/
├── mlflow/
│   └── mlflow.env
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingest.py
│   │   └── preprocess.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── tune.py
│   │   └── evaluate.py
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── app.py
│   │   └── schemas.py
│   └── monitoring/
│       ├── __init__.py
│       └── drift.py
├── terraform/
│   ├── main.tf
│   ├── variables.tf
│   └── outputs.tf
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_serving.py
├── .dvc/
├── .gitignore
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── requirements-train.txt
├── requirements-serve.txt
├── dvc.yaml
└── README.md
```
