# Sentiment Analysis MLOps Pipeline

End-to-end MLOps pipeline for training, deploying, and monitoring a sentiment analysis model on retail customer reviews.

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Data Ingest │───▶│ Preprocessing│───▶│   Training   │───▶│  Evaluation  │
│  (Amazon     │    │  (Clean,     │    │  (DistilBERT │    │  (Metrics,   │
│   Reviews)   │    │   Split)     │    │   Fine-tune) │    │   Report)    │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                              │                     │
                                              ▼                     ▼
                                        ┌──────────┐         ┌──────────┐
                                        │  MLflow   │         │  Model   │
                                        │ Tracking  │         │ Registry │
                                        └──────────┘         └──────────┘
                                                                   │
                                                                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Monitoring  │◀───│  Cloud Run   │◀───│   Docker     │◀───│  CI/CD       │
│  (Drift,     │    │  (Inference  │    │  (Build,     │    │  (GitHub     │
│   Metrics)   │    │   API)       │    │   Push)      │    │   Actions)   │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11 |
| DL Framework | PyTorch + Hugging Face Transformers |
| Model | DistilBERT (fine-tuned, 3-class sentiment) |
| HP Tuning | Optuna |
| API Framework | FastAPI |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| Containerization | Docker |
| Cloud | GCP (Cloud Run, Artifact Registry, GCS) |
| IaC | Terraform |
| CI/CD | GitHub Actions |
| Monitoring | GCP Cloud Monitoring + MLflow |

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- (Optional) GCP account with `gcloud` CLI
- (Optional) Terraform 1.5+

### Local Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/sentiment-mlops.git
cd sentiment-mlops

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
make install-train

# Run the full pipeline
make ingest       # Download Amazon Reviews
make preprocess   # Clean and split data
make train        # Fine-tune DistilBERT
make evaluate     # Evaluate on test set
make serve        # Start inference API at localhost:8080
```

### Docker Setup

```bash
# Start all services (MLflow + training + serving)
make docker-up

# View MLflow dashboard
open http://localhost:5000

# Query the API
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is amazing! Best purchase I ever made."}'
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Single review prediction |
| `/predict/batch` | POST | Batch prediction |
| `/metrics` | GET | Prometheus metrics |

### Sample Request

```json
POST /predict
{
  "text": "This product is amazing! Best purchase I ever made."
}
```

### Sample Response

```json
{
  "label": "positive",
  "confidence": 0.9523,
  "probabilities": {
    "negative": 0.0124,
    "neutral": 0.0353,
    "positive": 0.9523
  }
}
```

## Training

### Run Training

```bash
# Basic training
make train

# Hyperparameter tuning (10 trials)
make tune

# Evaluate best model
make evaluate
```

### MLflow Tracking

Training metrics are logged to MLflow. View the dashboard:

```bash
mlflow ui --port 5000
```

## Cloud Deployment (GCP)

### Setup Infrastructure

```bash
cd terraform
terraform init
terraform plan -var="project_id=YOUR_PROJECT_ID"
terraform apply -var="project_id=YOUR_PROJECT_ID"
```

### CI/CD

The pipeline uses GitHub Actions:

- **CI** (`ci.yml`): Runs on every PR — lint, typecheck, test, Docker build
- **CD** (`cd-staging.yml`): Runs on merge to main — builds, pushes, and deploys to Cloud Run
- **Training** (`train.yml`): Manual trigger or on model/data code changes

Required GitHub Secrets:
- `GCP_PROJECT_ID`
- `GCP_WORKLOAD_IDENTITY_PROVIDER`
- `GCP_SERVICE_ACCOUNT`

## Monitoring

### Metrics Tracked

- **MLflow**: Training loss, accuracy, F1 score per epoch across experiments
- **Prometheus**: Request latency, throughput, error rate, prediction label distribution
- **GCP Cloud Monitoring**: CPU/memory utilization, request count, 5xx error alerts
- **Data Drift**: Text length distribution (KS test), vocabulary overlap

### Run Drift Detection

```bash
python -m src.monitoring.drift
```

## Project Structure

```
├── .github/workflows/     # CI/CD pipelines
├── data/                  # Raw and processed data (DVC tracked)
├── docker/                # Dockerfiles for train and serve
├── docs/plans/            # Design and implementation docs
├── mlflow/                # MLflow configuration
├── src/
│   ├── data/              # Data ingestion and preprocessing
│   ├── model/             # Training, tuning, evaluation
│   ├── serving/           # FastAPI inference API
│   └── monitoring/        # Drift detection
├── terraform/             # GCP infrastructure as code
├── tests/                 # Unit and integration tests
├── docker-compose.yml     # Local development orchestration
├── dvc.yaml               # DVC pipeline definition
├── Makefile               # Task runner
└── pyproject.toml         # Project configuration
```

## Development

```bash
# Run tests
make test

# Run tests with coverage
make test-cov

# Lint
make lint

# Type check
make typecheck
```

## License

MIT
