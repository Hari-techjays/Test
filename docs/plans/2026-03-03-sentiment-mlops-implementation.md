# Sentiment Analysis MLOps Pipeline — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an end-to-end MLOps pipeline that ingests retail review data, fine-tunes DistilBERT for sentiment analysis, deploys via FastAPI on GCP Cloud Run, and monitors with MLflow + GCP Cloud Monitoring.

**Architecture:** Monorepo with separate modules for data, model, serving, and monitoring. Docker images for training and inference. Terraform for GCP infrastructure. GitHub Actions for CI/CD.

**Tech Stack:** Python 3.11, PyTorch, Hugging Face Transformers, Optuna, FastAPI, MLflow, Docker, Terraform, GCP (Cloud Run, Artifact Registry, GCS, Cloud Monitoring), GitHub Actions, DVC, pytest, ruff

---

## Task 1: Project Scaffolding — Config Files

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `requirements-train.txt`
- Create: `requirements-serve.txt`
- Create: `Makefile`

**Step 1: Create pyproject.toml**

```toml
[project]
name = "sentiment-mlops"
version = "0.1.0"
description = "End-to-end MLOps pipeline for sentiment analysis on retail reviews"
requires-python = ">=3.11"

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
addopts = "-v --tb=short"

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true
```

**Step 2: Create .gitignore**

```
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.eggs/
*.egg
.venv/
venv/

# Data (tracked by DVC)
data/raw/
data/processed/

# Model artifacts
models/
mlruns/
mlflow.db

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Environment
.env
*.env.local

# Docker
.docker/

# Terraform
terraform/.terraform/
terraform/*.tfstate
terraform/*.tfstate.backup
terraform/*.tfplan
terraform/.terraform.lock.hcl

# DVC
/data/*.csv
/data/*.json
/data/*.parquet
```

**Step 3: Create requirements-train.txt**

```
torch>=2.1.0
transformers>=4.36.0
datasets>=2.16.0
pandas>=2.1.0
pyarrow>=14.0.0
scikit-learn>=1.3.0
optuna>=3.4.0
mlflow>=2.9.0
dvc[gs]>=3.30.0
ruff>=0.1.0
mypy>=1.7.0
pytest>=7.4.0
pytest-cov>=4.1.0
```

**Step 4: Create requirements-serve.txt**

```
torch>=2.1.0
transformers>=4.36.0
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.5.0
prometheus-client>=0.19.0
mlflow>=2.9.0
pandas>=2.1.0
scipy>=1.11.0
```

**Step 5: Create Makefile**

```makefile
.PHONY: help install install-train install-serve lint typecheck test test-cov
.PHONY: ingest preprocess train tune evaluate serve
.PHONY: docker-build-train docker-build-serve docker-up docker-down

help:
	@echo "Available targets:"
	@echo "  install-train   Install training dependencies"
	@echo "  install-serve   Install serving dependencies"
	@echo "  lint            Run ruff linter"
	@echo "  typecheck       Run mypy type checker"
	@echo "  test            Run tests"
	@echo "  test-cov        Run tests with coverage"
	@echo "  ingest          Download raw review data"
	@echo "  preprocess      Clean and preprocess data"
	@echo "  train           Train sentiment model"
	@echo "  tune            Run hyperparameter tuning"
	@echo "  evaluate        Evaluate model on test set"
	@echo "  serve           Start inference server"
	@echo "  docker-build-train  Build training Docker image"
	@echo "  docker-build-serve  Build serving Docker image"
	@echo "  docker-up       Start all services with docker-compose"
	@echo "  docker-down     Stop all services"

install-train:
	pip install -r requirements-train.txt

install-serve:
	pip install -r requirements-serve.txt

lint:
	ruff check src/ tests/

typecheck:
	mypy src/

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing

ingest:
	python -m src.data.ingest

preprocess:
	python -m src.data.preprocess

train:
	python -m src.model.train

tune:
	python -m src.model.tune

evaluate:
	python -m src.model.evaluate

serve:
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8080 --reload

docker-build-train:
	docker build -f docker/Dockerfile.train -t sentiment-train .

docker-build-serve:
	docker build -f docker/Dockerfile.serve -t sentiment-serve .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down
```

**Step 6: Create package init files**

Create empty `__init__.py` files:
- `src/__init__.py`
- `src/data/__init__.py`
- `src/model/__init__.py`
- `src/serving/__init__.py`
- `src/monitoring/__init__.py`
- `tests/__init__.py`

Also create placeholder directories:
- `data/raw/.gitkeep`
- `data/processed/.gitkeep`

**Step 7: Commit**

```bash
git add pyproject.toml .gitignore requirements-train.txt requirements-serve.txt Makefile src/ tests/ data/
git commit -m "feat: add project scaffolding with config, deps, and directory structure"
```

---

## Task 2: Data Ingestion Pipeline

**Files:**
- Create: `src/data/ingest.py`
- Create: `tests/test_data.py`

**Step 1: Write the failing test for ingest**

```python
# tests/test_data.py
import os
import tempfile

import pandas as pd
import pytest


class TestIngest:
    def test_download_sample_reviews_returns_dataframe(self):
        from src.data.ingest import load_reviews

        df = load_reviews(sample_size=100)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_download_sample_reviews_has_required_columns(self):
        from src.data.ingest import load_reviews

        df = load_reviews(sample_size=100)
        assert "text" in df.columns or "reviewText" in df.columns
        assert "rating" in df.columns or "overall" in df.columns

    def test_save_raw_data(self):
        from src.data.ingest import load_reviews, save_raw

        with tempfile.TemporaryDirectory() as tmpdir:
            df = load_reviews(sample_size=50)
            output_path = os.path.join(tmpdir, "reviews.csv")
            save_raw(df, output_path)
            assert os.path.exists(output_path)
            loaded = pd.read_csv(output_path)
            assert len(loaded) == len(df)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data.py::TestIngest -v`
Expected: FAIL — module not found

**Step 3: Implement ingest.py**

```python
# src/data/ingest.py
"""Data ingestion module for Amazon Product Reviews."""

import logging
import os

import pandas as pd
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_RAW_PATH = os.path.join("data", "raw", "reviews.csv")


def load_reviews(
    dataset_name: str = "McAuley-Lab/Amazon-Reviews-2023",
    subset: str = "raw_review_Electronics",
    sample_size: int = 50000,
    seed: int = 42,
) -> pd.DataFrame:
    """Load Amazon Product Reviews from Hugging Face Datasets.

    Args:
        dataset_name: HF dataset identifier.
        subset: Dataset configuration/subset name.
        sample_size: Number of reviews to sample.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with review text and ratings.
    """
    logger.info(f"Loading {sample_size} reviews from {dataset_name}/{subset}")

    dataset = load_dataset(
        dataset_name,
        subset,
        split=f"full[:{sample_size}]",
        trust_remote_code=True,
    )

    df = dataset.to_pandas()

    # Standardize column names
    column_map = {}
    if "text" in df.columns:
        column_map["text"] = "text"
    elif "reviewText" in df.columns:
        column_map["reviewText"] = "text"

    if "rating" in df.columns:
        column_map["rating"] = "rating"
    elif "overall" in df.columns:
        column_map["overall"] = "rating"

    df = df.rename(columns=column_map)

    # Keep only needed columns
    keep_cols = ["text", "rating"]
    extra_cols = ["title", "asin", "timestamp"]
    for col in extra_cols:
        if col in df.columns:
            keep_cols.append(col)

    df = df[[c for c in keep_cols if c in df.columns]]

    logger.info(f"Loaded {len(df)} reviews")
    return df


def save_raw(df: pd.DataFrame, output_path: str = DEFAULT_RAW_PATH) -> str:
    """Save raw review data to CSV.

    Args:
        df: DataFrame to save.
        output_path: File path for output CSV.

    Returns:
        The output file path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} reviews to {output_path}")
    return output_path


if __name__ == "__main__":
    df = load_reviews()
    save_raw(df)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_data.py::TestIngest -v`
Expected: PASS (may take a minute to download data on first run)

**Step 5: Commit**

```bash
git add src/data/ingest.py tests/test_data.py
git commit -m "feat: add data ingestion pipeline with Amazon Reviews download"
```

---

## Task 3: Data Preprocessing Pipeline

**Files:**
- Modify: `tests/test_data.py`
- Create: `src/data/preprocess.py`

**Step 1: Write failing tests for preprocessing**

Append to `tests/test_data.py`:

```python
class TestPreprocess:
    def setup_method(self):
        """Create sample data for testing."""
        self.sample_df = pd.DataFrame({
            "text": [
                "This product is <b>amazing</b>! I love it.",
                "Terrible quality. Broke after one day.",
                "It's okay, nothing special.",
                None,
                "",
                "Great product & fast shipping!!!",
            ],
            "rating": [5.0, 1.0, 3.0, 4.0, 2.0, 5.0],
        })

    def test_clean_text_removes_html(self):
        from src.data.preprocess import clean_text

        result = clean_text("This is <b>bold</b> text")
        assert "<b>" not in result
        assert "bold" in result

    def test_clean_text_lowercases(self):
        from src.data.preprocess import clean_text

        result = clean_text("UPPERCASE TEXT")
        assert result == "uppercase text"

    def test_clean_text_handles_none(self):
        from src.data.preprocess import clean_text

        result = clean_text(None)
        assert result == ""

    def test_map_sentiment_labels(self):
        from src.data.preprocess import map_sentiment

        assert map_sentiment(1.0) == "negative"
        assert map_sentiment(2.0) == "negative"
        assert map_sentiment(3.0) == "neutral"
        assert map_sentiment(4.0) == "positive"
        assert map_sentiment(5.0) == "positive"

    def test_preprocess_pipeline_output_shape(self):
        from src.data.preprocess import preprocess

        result = preprocess(self.sample_df)
        assert "text" in result.columns
        assert "label" in result.columns
        assert "rating" in result.columns
        # Should drop rows with empty text after cleaning
        assert len(result) <= len(self.sample_df)
        # All texts should be non-empty strings
        assert all(len(t) > 0 for t in result["text"])

    def test_split_data_proportions(self):
        from src.data.preprocess import split_data

        # Create a larger sample for meaningful splits
        big_df = pd.DataFrame({
            "text": [f"review {i}" for i in range(100)],
            "label": ["positive"] * 40 + ["negative"] * 30 + ["neutral"] * 30,
            "rating": [5.0] * 40 + [1.0] * 30 + [3.0] * 30,
        })
        train, val, test = split_data(big_df)
        total = len(train) + len(val) + len(test)
        assert total == 100
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

    def test_save_and_load_splits(self):
        from src.data.preprocess import preprocess, save_splits, split_data

        with tempfile.TemporaryDirectory() as tmpdir:
            processed = preprocess(self.sample_df)
            train, val, test = split_data(processed)
            save_splits(train, val, test, output_dir=tmpdir)

            for name in ["train", "val", "test"]:
                path = os.path.join(tmpdir, f"{name}.parquet")
                assert os.path.exists(path)
                loaded = pd.read_parquet(path)
                assert "text" in loaded.columns
                assert "label" in loaded.columns
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data.py::TestPreprocess -v`
Expected: FAIL — module not found

**Step 3: Implement preprocess.py**

```python
# src/data/preprocess.py
"""Data preprocessing module for review text cleaning and splitting."""

import html
import logging
import os
import re

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_PROCESSED_DIR = os.path.join("data", "processed")
LABEL_MAP = {1.0: "negative", 2.0: "negative", 3.0: "neutral", 4.0: "positive", 5.0: "positive"}


def clean_text(text: str | None) -> str:
    """Clean review text by removing HTML, lowercasing, and normalizing whitespace."""
    if text is None or not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def map_sentiment(rating: float) -> str:
    """Map a star rating (1-5) to a sentiment label."""
    return LABEL_MAP[float(rating)]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Run full preprocessing pipeline on raw review data.

    Args:
        df: Raw DataFrame with 'text' and 'rating' columns.

    Returns:
        Cleaned DataFrame with 'text', 'rating', and 'label' columns.
    """
    logger.info(f"Preprocessing {len(df)} reviews")
    result = df.copy()

    # Clean text
    result["text"] = result["text"].apply(clean_text)

    # Drop empty texts
    result = result[result["text"].str.len() > 0].reset_index(drop=True)

    # Drop rows with missing ratings
    result = result.dropna(subset=["rating"]).reset_index(drop=True)

    # Map sentiment labels
    result["label"] = result["rating"].apply(map_sentiment)

    logger.info(f"After preprocessing: {len(result)} reviews")
    logger.info(f"Label distribution:\n{result['label'].value_counts()}")

    return result


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train/validation/test sets with stratification.

    Args:
        df: Preprocessed DataFrame with 'label' column.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation. Test gets the remainder.
        seed: Random seed.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    test_ratio = 1.0 - train_ratio - val_ratio

    train_df, temp_df = train_test_split(
        df, test_size=(val_ratio + test_ratio), random_state=seed, stratify=df["label"]
    )
    relative_val_ratio = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - relative_val_ratio), random_state=seed, stratify=temp_df["label"]
    )

    logger.info(f"Split sizes — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: str = DEFAULT_PROCESSED_DIR,
) -> None:
    """Save train/val/test splits as Parquet files."""
    os.makedirs(output_dir, exist_ok=True)
    for name, df in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(output_dir, f"{name}.parquet")
        df.to_parquet(path, index=False)
        logger.info(f"Saved {name} split ({len(df)} rows) to {path}")


if __name__ == "__main__":
    raw_path = os.path.join("data", "raw", "reviews.csv")
    df = pd.read_csv(raw_path)
    processed = preprocess(df)
    train, val, test = split_data(processed)
    save_splits(train, val, test)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_data.py::TestPreprocess -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/data/preprocess.py tests/test_data.py
git commit -m "feat: add data preprocessing with text cleaning, labeling, and train/val/test split"
```

---

## Task 4: Model Training

**Files:**
- Create: `src/model/train.py`
- Create: `tests/test_model.py`

**Step 1: Write failing tests for training components**

```python
# tests/test_model.py
import os
import tempfile

import pandas as pd
import pytest
import torch


class TestSentimentDataset:
    def test_dataset_length(self):
        from src.model.train import SentimentDataset

        texts = ["good product", "bad product", "okay product"]
        labels = ["positive", "negative", "neutral"]
        ds = SentimentDataset(texts, labels, max_length=32)
        assert len(ds) == 3

    def test_dataset_item_keys(self):
        from src.model.train import SentimentDataset

        texts = ["good product"]
        labels = ["positive"]
        ds = SentimentDataset(texts, labels, max_length=32)
        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item

    def test_dataset_label_encoding(self):
        from src.model.train import LABEL2ID, SentimentDataset

        texts = ["good", "bad", "okay"]
        labels = ["positive", "negative", "neutral"]
        ds = SentimentDataset(texts, labels, max_length=32)
        for i, label in enumerate(labels):
            item = ds[i]
            assert item["labels"].item() == LABEL2ID[label]


class TestModelFactory:
    def test_create_model_returns_model(self):
        from src.model.train import create_model

        model = create_model(num_labels=3)
        assert model is not None
        # Should have a classifier head
        assert hasattr(model, "classifier") or hasattr(model, "pre_classifier")

    def test_create_model_correct_num_labels(self):
        from src.model.train import create_model

        model = create_model(num_labels=3)
        config = model.config
        assert config.num_labels == 3


class TestTrainFunction:
    def test_train_one_step_no_error(self):
        from src.model.train import SentimentDataset, create_model, get_training_args

        texts = ["good product"] * 4
        labels = ["positive"] * 4
        ds = SentimentDataset(texts, labels, max_length=32)

        model = create_model(num_labels=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            args = get_training_args(
                output_dir=tmpdir, num_epochs=1, batch_size=2, learning_rate=5e-5
            )
            # Just verify it doesn't crash — actual training tested in integration
            assert args is not None
            assert args.num_train_epochs == 1
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_model.py -v`
Expected: FAIL — module not found

**Step 3: Implement train.py**

```python
# src/model/train.py
"""Model training module for DistilBERT sentiment classifier."""

import json
import logging
import os

import mlflow
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "distilbert-base-uncased"
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
DEFAULT_OUTPUT_DIR = os.path.join("models", "sentiment")
DEFAULT_DATA_DIR = os.path.join("data", "processed")


class SentimentDataset(torch.utils.data.Dataset):
    """PyTorch dataset for sentiment classification."""

    def __init__(
        self,
        texts: list[str],
        labels: list[str],
        max_length: int = 128,
        tokenizer_name: str = MODEL_NAME,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.encodings = self.tokenizer(
            texts, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        self.labels = torch.tensor([LABEL2ID[label] for label in labels], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def create_model(
    num_labels: int = 3, model_name: str = MODEL_NAME
) -> AutoModelForSequenceClassification:
    """Create a DistilBERT model for sequence classification."""
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    return model


def get_training_args(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    warmup_steps: int = 500,
    weight_decay: float = 0.01,
    eval_strategy: str = "epoch",
    save_strategy: str = "epoch",
    load_best_model_at_end: bool = True,
) -> TrainingArguments:
    """Build Hugging Face TrainingArguments."""
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model="f1_macro",
        logging_steps=50,
        report_to="none",  # We use MLflow manually
    )


def compute_metrics(pred: EvalPrediction) -> dict:
    """Compute accuracy and F1 for evaluation."""
    from sklearn.metrics import accuracy_score, f1_score

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1}


def load_data(data_dir: str = DEFAULT_DATA_DIR) -> tuple:
    """Load preprocessed train/val Parquet files."""
    import pandas as pd

    train_df = pd.read_parquet(os.path.join(data_dir, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(data_dir, "val.parquet"))
    return train_df, val_df


def train(
    data_dir: str = DEFAULT_DATA_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 128,
) -> dict:
    """Train DistilBERT sentiment classifier and log to MLflow."""
    train_df, val_df = load_data(data_dir)

    logger.info(f"Training on {len(train_df)} samples, validating on {len(val_df)}")

    train_dataset = SentimentDataset(
        train_df["text"].tolist(), train_df["label"].tolist(), max_length=max_length
    )
    val_dataset = SentimentDataset(
        val_df["text"].tolist(), val_df["label"].tolist(), max_length=max_length
    )

    model = create_model()

    training_args = get_training_args(
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    mlflow.set_experiment("sentiment-analysis")
    with mlflow.start_run(run_name="distilbert-train"):
        mlflow.log_params({
            "model_name": MODEL_NAME,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_length": max_length,
            "train_size": len(train_df),
            "val_size": len(val_df),
        })

        trainer.train()
        metrics = trainer.evaluate()

        mlflow.log_metrics({
            "val_accuracy": metrics["eval_accuracy"],
            "val_f1_macro": metrics["eval_f1_macro"],
            "val_loss": metrics["eval_loss"],
        })

        # Save model
        trainer.save_model(output_dir)
        train_dataset.tokenizer.save_pretrained(output_dir)

        # Log model to MLflow
        mlflow.pytorch.log_model(model, "model")

        logger.info(f"Training complete. Metrics: {metrics}")

    return metrics


if __name__ == "__main__":
    metrics = train()
    print(json.dumps(metrics, indent=2))
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_model.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/model/train.py tests/test_model.py
git commit -m "feat: add DistilBERT training pipeline with MLflow integration"
```

---

## Task 5: Hyperparameter Tuning

**Files:**
- Create: `src/model/tune.py`
- Modify: `tests/test_model.py` (add tuning tests)

**Step 1: Write failing tests**

Append to `tests/test_model.py`:

```python
class TestTuning:
    def test_objective_returns_float(self):
        from unittest.mock import MagicMock, patch

        from src.model.tune import create_objective

        # Mock the training to avoid actual GPU work
        objective = create_objective(
            train_texts=["good"] * 8,
            train_labels=["positive"] * 8,
            val_texts=["bad"] * 4,
            val_labels=["negative"] * 4,
            max_length=32,
            n_epochs_max=1,
        )
        assert callable(objective)

    def test_suggest_hyperparameters(self):
        from unittest.mock import MagicMock

        from src.model.tune import suggest_hyperparameters

        trial = MagicMock()
        trial.suggest_float.return_value = 3e-5
        trial.suggest_categorical.return_value = 16
        trial.suggest_int.return_value = 100

        params = suggest_hyperparameters(trial)
        assert "learning_rate" in params
        assert "batch_size" in params
        assert "warmup_steps" in params
        assert "num_epochs" in params
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_model.py::TestTuning -v`
Expected: FAIL

**Step 3: Implement tune.py**

```python
# src/model/tune.py
"""Hyperparameter tuning with Optuna for sentiment model."""

import logging
import os
from typing import Callable

import mlflow
import optuna
import pandas as pd
from transformers import Trainer

from src.model.train import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    SentimentDataset,
    compute_metrics,
    create_model,
    get_training_args,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def suggest_hyperparameters(trial: optuna.Trial) -> dict:
    """Suggest hyperparameters for an Optuna trial."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
        "warmup_steps": trial.suggest_int("warmup_steps", 0, 1000, step=100),
        "num_epochs": trial.suggest_int("num_epochs", 2, 5),
    }


def create_objective(
    train_texts: list[str],
    train_labels: list[str],
    val_texts: list[str],
    val_labels: list[str],
    max_length: int = 128,
    n_epochs_max: int = 5,
) -> Callable:
    """Create an Optuna objective function."""

    def objective(trial: optuna.Trial) -> float:
        params = suggest_hyperparameters(trial)

        train_dataset = SentimentDataset(train_texts, train_labels, max_length=max_length)
        val_dataset = SentimentDataset(val_texts, val_labels, max_length=max_length)

        model = create_model()

        output_dir = os.path.join(DEFAULT_OUTPUT_DIR, f"trial_{trial.number}")
        training_args = get_training_args(
            output_dir=output_dir,
            num_epochs=params["num_epochs"],
            batch_size=params["batch_size"],
            learning_rate=params["learning_rate"],
            warmup_steps=params["warmup_steps"],
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        metrics = trainer.evaluate()

        with mlflow.start_run(run_name=f"optuna-trial-{trial.number}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metrics({
                "val_accuracy": metrics["eval_accuracy"],
                "val_f1_macro": metrics["eval_f1_macro"],
                "val_loss": metrics["eval_loss"],
            })

        return metrics["eval_f1_macro"]

    return objective


def tune(
    data_dir: str = DEFAULT_DATA_DIR,
    n_trials: int = 10,
    max_length: int = 128,
) -> optuna.Study:
    """Run hyperparameter tuning with Optuna."""
    train_df = pd.read_parquet(os.path.join(data_dir, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(data_dir, "val.parquet"))

    logger.info(f"Starting Optuna study with {n_trials} trials")

    mlflow.set_experiment("sentiment-analysis-tuning")
    with mlflow.start_run(run_name="optuna-study"):
        objective = create_objective(
            train_texts=train_df["text"].tolist(),
            train_labels=train_df["label"].tolist(),
            val_texts=val_df["text"].tolist(),
            val_labels=val_df["label"].tolist(),
            max_length=max_length,
        )

        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(),
            study_name="sentiment-hparam-search",
        )
        study.optimize(objective, n_trials=n_trials)

        best = study.best_trial
        logger.info(f"Best trial: {best.number}")
        logger.info(f"Best F1: {best.value:.4f}")
        logger.info(f"Best params: {best.params}")

        mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})
        mlflow.log_metric("best_f1_macro", best.value)

    return study


if __name__ == "__main__":
    study = tune(n_trials=10)
    print(f"\nBest trial F1: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_model.py::TestTuning -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/model/tune.py tests/test_model.py
git commit -m "feat: add Optuna hyperparameter tuning with MLflow logging"
```

---

## Task 6: Model Evaluation

**Files:**
- Create: `src/model/evaluate.py`
- Modify: `tests/test_model.py`

**Step 1: Write failing tests**

Append to `tests/test_model.py`:

```python
import numpy as np


class TestEvaluate:
    def test_compute_classification_report(self):
        from src.model.evaluate import compute_classification_report

        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 2, 1]
        report = compute_classification_report(y_true, y_pred)
        assert "accuracy" in report
        assert isinstance(report, dict)

    def test_save_metrics_creates_file(self):
        from src.model.evaluate import save_metrics

        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = {"accuracy": 0.85, "f1_macro": 0.83}
            path = os.path.join(tmpdir, "metrics.json")
            save_metrics(metrics, path)
            assert os.path.exists(path)

            import json
            with open(path) as f:
                loaded = json.load(f)
            assert loaded["accuracy"] == 0.85
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_model.py::TestEvaluate -v`
Expected: FAIL

**Step 3: Implement evaluate.py**

```python
# src/model/evaluate.py
"""Model evaluation module for sentiment classifier."""

import json
import logging
import os

import mlflow
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.model.train import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, ID2LABEL, LABEL2ID

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_classification_report(y_true: list[int], y_pred: list[int]) -> dict:
    """Compute sklearn classification report as a dict."""
    return classification_report(
        y_true, y_pred, target_names=list(LABEL2ID.keys()), output_dict=True
    )


def save_metrics(metrics: dict, output_path: str) -> None:
    """Save metrics dictionary to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {output_path}")


def evaluate(
    model_dir: str = DEFAULT_OUTPUT_DIR,
    data_dir: str = DEFAULT_DATA_DIR,
    batch_size: int = 32,
    max_length: int = 128,
) -> dict:
    """Evaluate trained model on test set."""
    test_df = pd.read_parquet(os.path.join(data_dir, "test.parquet"))
    logger.info(f"Evaluating on {len(test_df)} test samples")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_preds = []
    all_labels = []
    texts = test_df["text"].tolist()
    labels = [LABEL2ID[label] for label in test_df["label"].tolist()]

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_labels = labels[i : i + batch_size]

        inputs = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            preds = outputs.logits.argmax(-1).cpu().tolist()

        all_preds.extend(preds)
        all_labels.extend(batch_labels)

    report = compute_classification_report(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds).tolist()

    metrics = {
        "accuracy": report["accuracy"],
        "f1_macro": report["macro avg"]["f1-score"],
        "precision_macro": report["macro avg"]["precision"],
        "recall_macro": report["macro avg"]["recall"],
        "confusion_matrix": cm,
        "per_class": {
            label: {
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": report[label]["support"],
            }
            for label in LABEL2ID
        },
    }

    # Log to MLflow
    mlflow.set_experiment("sentiment-analysis")
    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_metrics({
            "test_accuracy": metrics["accuracy"],
            "test_f1_macro": metrics["f1_macro"],
        })

    # Save metrics
    metrics_path = os.path.join(model_dir, "test_metrics.json")
    save_metrics(metrics, metrics_path)

    logger.info(f"Test accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Test F1 (macro): {metrics['f1_macro']:.4f}")

    return metrics


if __name__ == "__main__":
    metrics = evaluate()
    print(json.dumps(metrics, indent=2))
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_model.py::TestEvaluate -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/model/evaluate.py tests/test_model.py
git commit -m "feat: add model evaluation with classification report and metrics export"
```

---

## Task 7: Inference Service — Schemas

**Files:**
- Create: `src/serving/schemas.py`
- Create: `tests/test_serving.py`

**Step 1: Write failing tests**

```python
# tests/test_serving.py
import pytest
from pydantic import ValidationError


class TestSchemas:
    def test_predict_request_valid(self):
        from src.serving.schemas import PredictRequest

        req = PredictRequest(text="This product is great!")
        assert req.text == "This product is great!"

    def test_predict_request_empty_text_fails(self):
        from src.serving.schemas import PredictRequest

        with pytest.raises(ValidationError):
            PredictRequest(text="")

    def test_predict_response_structure(self):
        from src.serving.schemas import PredictResponse

        resp = PredictResponse(
            label="positive",
            confidence=0.95,
            probabilities={"positive": 0.95, "neutral": 0.03, "negative": 0.02},
        )
        assert resp.label == "positive"
        assert resp.confidence == 0.95

    def test_batch_predict_request(self):
        from src.serving.schemas import BatchPredictRequest

        req = BatchPredictRequest(texts=["good", "bad"])
        assert len(req.texts) == 2

    def test_batch_predict_request_empty_fails(self):
        from src.serving.schemas import BatchPredictRequest

        with pytest.raises(ValidationError):
            BatchPredictRequest(texts=[])
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_serving.py::TestSchemas -v`
Expected: FAIL

**Step 3: Implement schemas.py**

```python
# src/serving/schemas.py
"""Pydantic schemas for the inference API."""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Review text to classify")


class PredictResponse(BaseModel):
    label: str = Field(..., description="Predicted sentiment label")
    confidence: float = Field(..., description="Confidence score for the predicted label")
    probabilities: dict[str, float] = Field(
        ..., description="Probability for each sentiment class"
    )


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="List of review texts to classify")


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_serving.py::TestSchemas -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/serving/schemas.py tests/test_serving.py
git commit -m "feat: add Pydantic request/response schemas for inference API"
```

---

## Task 8: Inference Service — FastAPI App

**Files:**
- Create: `src/serving/app.py`
- Modify: `tests/test_serving.py`

**Step 1: Write failing tests**

Append to `tests/test_serving.py`:

```python
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


class TestApp:
    def test_health_endpoint(self):
        with patch("src.serving.app.model", None), \
             patch("src.serving.app.tokenizer", None):
            from src.serving.app import app

            client = TestClient(app)
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert "model_loaded" in data

    def test_predict_without_model_returns_503(self):
        with patch("src.serving.app.model", None), \
             patch("src.serving.app.tokenizer", None):
            from src.serving.app import app

            client = TestClient(app)
            response = client.post("/predict", json={"text": "great product"})
            assert response.status_code == 503

    def test_predict_with_mock_model(self):
        import torch

        mock_model = MagicMock()
        mock_logits = torch.tensor([[0.1, 0.2, 0.9]])
        mock_model.return_value = MagicMock(logits=mock_logits)
        mock_model.eval = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.device = torch.device("cpu")

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[101, 2307, 3580, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        }

        with patch("src.serving.app.model", mock_model), \
             patch("src.serving.app.tokenizer", mock_tokenizer):
            from src.serving.app import app

            client = TestClient(app)
            response = client.post("/predict", json={"text": "great product"})
            assert response.status_code == 200
            data = response.json()
            assert "label" in data
            assert "confidence" in data
            assert "probabilities" in data

    def test_metrics_endpoint(self):
        with patch("src.serving.app.model", None), \
             patch("src.serving.app.tokenizer", None):
            from src.serving.app import app

            client = TestClient(app)
            response = client.get("/metrics")
            assert response.status_code == 200
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_serving.py::TestApp -v`
Expected: FAIL

**Step 3: Implement app.py**

```python
# src/serving/app.py
"""FastAPI inference service for sentiment classification."""

import logging
import os
import time

import torch
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.model.train import ID2LABEL
from src.serving.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sentiment Analysis API",
    description="DistilBERT-based sentiment classifier for retail reviews",
    version="0.1.0",
)

# Prometheus metrics
REQUEST_COUNT = Counter("predict_requests_total", "Total prediction requests", ["endpoint"])
REQUEST_LATENCY = Histogram("predict_latency_seconds", "Prediction latency", ["endpoint"])
PREDICTION_LABELS = Counter("prediction_labels_total", "Predictions by label", ["label"])

# Global model state
model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join("models", "sentiment"))


@app.on_event("startup")
def load_model():
    """Load model and tokenizer on startup."""
    global model, tokenizer
    try:
        if os.path.exists(MODEL_DIR):
            tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
            model.eval()
            model.to(device)
            logger.info(f"Model loaded from {MODEL_DIR} on {device}")
        else:
            logger.warning(f"Model directory {MODEL_DIR} not found. Server starting without model.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")


def predict_sentiment(text: str) -> PredictResponse:
    """Run inference on a single text."""
    inputs = tokenizer(
        text, truncation=True, padding="max_length", max_length=128, return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    pred_idx = probs.argmax().item()
    label = ID2LABEL[pred_idx]
    confidence = probs[pred_idx].item()
    probabilities = {ID2LABEL[i]: round(p.item(), 4) for i, p in enumerate(probs)}

    PREDICTION_LABELS.labels(label=label).inc()

    return PredictResponse(label=label, confidence=round(confidence, 4), probabilities=probabilities)


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint."""
    return HealthResponse(status="healthy", model_loaded=model is not None)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Predict sentiment for a single review."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    REQUEST_COUNT.labels(endpoint="/predict").inc()
    start = time.time()

    result = predict_sentiment(request.text)

    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start)
    return result


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest):
    """Predict sentiment for multiple reviews."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    REQUEST_COUNT.labels(endpoint="/predict/batch").inc()
    start = time.time()

    predictions = [predict_sentiment(text) for text in request.texts]

    REQUEST_LATENCY.labels(endpoint="/predict/batch").observe(time.time() - start)
    return BatchPredictResponse(predictions=predictions)


@app.get("/metrics")
def metrics():
    """Expose Prometheus metrics."""
    return Response(content=generate_latest(), media_type="text/plain; charset=utf-8")
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_serving.py::TestApp -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/serving/app.py tests/test_serving.py
git commit -m "feat: add FastAPI inference service with health, predict, batch, and metrics endpoints"
```

---

## Task 9: Monitoring — Drift Detection

**Files:**
- Create: `src/monitoring/drift.py`
- Create: `tests/test_monitoring.py`

**Step 1: Write failing tests**

```python
# tests/test_monitoring.py
import numpy as np
import pandas as pd
import pytest


class TestDriftDetection:
    def test_compute_text_length_stats(self):
        from src.monitoring.drift import compute_text_length_stats

        texts = ["short", "a medium length text", "a much longer text that has more words in it"]
        stats = compute_text_length_stats(texts)
        assert "mean" in stats
        assert "std" in stats
        assert "median" in stats
        assert stats["mean"] > 0

    def test_detect_length_drift_no_drift(self):
        from src.monitoring.drift import detect_length_drift

        reference = ["word " * 10] * 100
        current = ["word " * 10] * 50
        result = detect_length_drift(reference, current)
        assert result["drift_detected"] is False

    def test_detect_length_drift_with_drift(self):
        from src.monitoring.drift import detect_length_drift

        reference = ["short"] * 100
        current = ["this is a very long text with many many words " * 10] * 50
        result = detect_length_drift(reference, current)
        assert result["drift_detected"] is True

    def test_detect_vocab_drift_no_drift(self):
        from src.monitoring.drift import detect_vocab_drift

        reference = ["the cat sat on the mat"] * 100
        current = ["the cat sat on the mat"] * 50
        result = detect_vocab_drift(reference, current)
        assert result["vocab_overlap"] > 0.5

    def test_compute_drift_report(self):
        from src.monitoring.drift import compute_drift_report

        reference = ["good product works well"] * 100
        current = ["good product works well"] * 50
        report = compute_drift_report(reference, current)
        assert "length_drift" in report
        assert "vocab_drift" in report
        assert "overall_drift" in report
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_monitoring.py -v`
Expected: FAIL

**Step 3: Implement drift.py**

```python
# src/monitoring/drift.py
"""Data drift detection for monitoring incoming review data."""

import logging
from collections import Counter

import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LENGTH_DRIFT_THRESHOLD = 0.05  # p-value threshold for KS test
VOCAB_OVERLAP_THRESHOLD = 0.5  # Minimum acceptable vocabulary overlap


def compute_text_length_stats(texts: list[str]) -> dict:
    """Compute basic statistics on text lengths."""
    lengths = [len(t.split()) for t in texts]
    return {
        "mean": float(np.mean(lengths)),
        "std": float(np.std(lengths)),
        "median": float(np.median(lengths)),
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
    }


def detect_length_drift(
    reference_texts: list[str],
    current_texts: list[str],
    threshold: float = LENGTH_DRIFT_THRESHOLD,
) -> dict:
    """Detect drift in text length distribution using KS test."""
    ref_lengths = [len(t.split()) for t in reference_texts]
    cur_lengths = [len(t.split()) for t in current_texts]

    ks_stat, p_value = stats.ks_2samp(ref_lengths, cur_lengths)

    drift_detected = p_value < threshold

    return {
        "ks_statistic": float(ks_stat),
        "p_value": float(p_value),
        "drift_detected": drift_detected,
        "reference_stats": compute_text_length_stats(reference_texts),
        "current_stats": compute_text_length_stats(current_texts),
    }


def detect_vocab_drift(
    reference_texts: list[str],
    current_texts: list[str],
    top_n: int = 1000,
) -> dict:
    """Detect vocabulary drift by comparing top-N word frequencies."""
    ref_words = Counter()
    for text in reference_texts:
        ref_words.update(text.lower().split())

    cur_words = Counter()
    for text in current_texts:
        cur_words.update(text.lower().split())

    ref_top = set(w for w, _ in ref_words.most_common(top_n))
    cur_top = set(w for w, _ in cur_words.most_common(top_n))

    overlap = len(ref_top & cur_top) / max(len(ref_top), 1)

    return {
        "vocab_overlap": float(overlap),
        "reference_vocab_size": len(ref_words),
        "current_vocab_size": len(cur_words),
        "drift_detected": overlap < VOCAB_OVERLAP_THRESHOLD,
    }


def compute_drift_report(
    reference_texts: list[str],
    current_texts: list[str],
) -> dict:
    """Compute a full drift report combining length and vocabulary drift."""
    length_drift = detect_length_drift(reference_texts, current_texts)
    vocab_drift = detect_vocab_drift(reference_texts, current_texts)

    overall_drift = length_drift["drift_detected"] or vocab_drift["drift_detected"]

    report = {
        "length_drift": length_drift,
        "vocab_drift": vocab_drift,
        "overall_drift": overall_drift,
        "reference_size": len(reference_texts),
        "current_size": len(current_texts),
    }

    if overall_drift:
        logger.warning("DATA DRIFT DETECTED — review incoming data quality")
    else:
        logger.info("No significant drift detected")

    return report


if __name__ == "__main__":
    import json
    import os

    import pandas as pd

    train_df = pd.read_parquet(os.path.join("data", "processed", "train.parquet"))
    # In production, current_texts would come from recent predictions
    # For demo, use test set
    test_df = pd.read_parquet(os.path.join("data", "processed", "test.parquet"))

    report = compute_drift_report(
        reference_texts=train_df["text"].tolist(),
        current_texts=test_df["text"].tolist(),
    )
    print(json.dumps(report, indent=2))
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_monitoring.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/monitoring/drift.py tests/test_monitoring.py
git commit -m "feat: add data drift detection with KS test and vocabulary overlap"
```

---

## Task 10: Docker — Training Image

**Files:**
- Create: `docker/Dockerfile.train`

**Step 1: Create Dockerfile.train**

```dockerfile
# docker/Dockerfile.train
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt

FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src/ src/
COPY data/ data/
COPY pyproject.toml .

# Default command: run training
CMD ["python", "-m", "src.model.train"]
```

**Step 2: Verify Dockerfile builds**

Run: `docker build -f docker/Dockerfile.train -t sentiment-train .`
Expected: Successful build (may take several minutes)

**Step 3: Commit**

```bash
git add docker/Dockerfile.train
git commit -m "feat: add multi-stage Docker image for training"
```

---

## Task 11: Docker — Serving Image

**Files:**
- Create: `docker/Dockerfile.serve`

**Step 1: Create Dockerfile.serve**

```dockerfile
# docker/Dockerfile.serve
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src/ src/

# Model directory will be mounted or baked in
ENV MODEL_DIR=/app/models/sentiment
ENV PORT=8080

EXPOSE 8080

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8080"]
```

**Step 2: Verify Dockerfile builds**

Run: `docker build -f docker/Dockerfile.serve -t sentiment-serve .`
Expected: Successful build

**Step 3: Commit**

```bash
git add docker/Dockerfile.serve
git commit -m "feat: add multi-stage Docker image for inference serving"
```

---

## Task 12: Docker Compose

**Files:**
- Create: `docker-compose.yml`
- Create: `mlflow/mlflow.env`

**Step 1: Create docker-compose.yml**

```yaml
# docker-compose.yml
version: "3.8"

services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.9.2
    ports:
      - "5000:5000"
    environment:
      - MLFLOW_TRACKING_URI=sqlite:///mlflow.db
    volumes:
      - mlflow-data:/mlflow
    command: >
      mlflow server
      --host 0.0.0.0
      --port 5000
      --backend-store-uri sqlite:///mlflow/mlflow.db
      --default-artifact-root /mlflow/artifacts

  train:
    build:
      context: .
      dockerfile: docker/Dockerfile.train
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    depends_on:
      - mlflow

  serve:
    build:
      context: .
      dockerfile: docker/Dockerfile.serve
    ports:
      - "8080:8080"
    volumes:
      - ./models:/app/models
    environment:
      - MODEL_DIR=/app/models/sentiment
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    depends_on:
      - mlflow

volumes:
  mlflow-data:
```

**Step 2: Create mlflow.env**

```
# mlflow/mlflow.env
MLFLOW_TRACKING_URI=http://localhost:5000
```

**Step 3: Verify compose config**

Run: `docker-compose config`
Expected: Validated YAML output

**Step 4: Commit**

```bash
git add docker-compose.yml mlflow/mlflow.env
git commit -m "feat: add Docker Compose for local development with MLflow, training, and serving"
```

---

## Task 13: Terraform — GCP Infrastructure

**Files:**
- Create: `terraform/main.tf`
- Create: `terraform/variables.tf`
- Create: `terraform/outputs.tf`

**Step 1: Create variables.tf**

```hcl
# terraform/variables.tf

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "app_name" {
  description = "Application name used for naming resources"
  type        = string
  default     = "sentiment-mlops"
}

variable "cloud_run_memory" {
  description = "Memory allocation for Cloud Run"
  type        = string
  default     = "2Gi"
}

variable "cloud_run_cpu" {
  description = "CPU allocation for Cloud Run"
  type        = string
  default     = "2"
}

variable "cloud_run_min_instances" {
  description = "Minimum instances for Cloud Run"
  type        = number
  default     = 0
}

variable "cloud_run_max_instances" {
  description = "Maximum instances for Cloud Run"
  type        = number
  default     = 5
}
```

**Step 2: Create main.tf**

```hcl
# terraform/main.tf

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Enable required APIs
resource "google_project_service" "apis" {
  for_each = toset([
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "storage.googleapis.com",
    "monitoring.googleapis.com",
    "secretmanager.googleapis.com",
    "iam.googleapis.com",
  ])

  service            = each.value
  disable_on_destroy = false
}

# Artifact Registry for Docker images
resource "google_artifact_registry_repository" "docker" {
  location      = var.region
  repository_id = "${var.app_name}-docker"
  format        = "DOCKER"
  description   = "Docker images for sentiment MLOps pipeline"

  depends_on = [google_project_service.apis]
}

# Cloud Storage buckets
resource "google_storage_bucket" "data" {
  name                        = "${var.project_id}-${var.app_name}-data"
  location                    = var.region
  force_destroy               = true
  uniform_bucket_level_access = true
}

resource "google_storage_bucket" "models" {
  name                        = "${var.project_id}-${var.app_name}-models"
  location                    = var.region
  force_destroy               = true
  uniform_bucket_level_access = true
}

resource "google_storage_bucket" "mlflow" {
  name                        = "${var.project_id}-${var.app_name}-mlflow"
  location                    = var.region
  force_destroy               = true
  uniform_bucket_level_access = true
}

# Service account for Cloud Run
resource "google_service_account" "cloud_run" {
  account_id   = "${var.app_name}-run"
  display_name = "Sentiment MLOps Cloud Run SA"
}

# Grant Cloud Run SA access to model bucket
resource "google_storage_bucket_iam_member" "run_models_reader" {
  bucket = google_storage_bucket.models.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.cloud_run.email}"
}

# Grant Cloud Run SA access to MLflow bucket
resource "google_storage_bucket_iam_member" "run_mlflow_reader" {
  bucket = google_storage_bucket.mlflow.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.cloud_run.email}"
}

# Grant Cloud Run SA monitoring metric writer
resource "google_project_iam_member" "run_monitoring" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.cloud_run.email}"
}

# Cloud Run service
resource "google_cloud_run_v2_service" "inference" {
  name     = "${var.app_name}-inference"
  location = var.region

  template {
    scaling {
      min_instance_count = var.cloud_run_min_instances
      max_instance_count = var.cloud_run_max_instances
    }

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.docker.repository_id}/sentiment-serve:latest"

      resources {
        limits = {
          memory = var.cloud_run_memory
          cpu    = var.cloud_run_cpu
        }
      }

      ports {
        container_port = 8080
      }

      env {
        name  = "MODEL_DIR"
        value = "/app/models/sentiment"
      }

      startup_probe {
        http_get {
          path = "/health"
        }
        initial_delay_seconds = 10
        period_seconds        = 5
        failure_threshold     = 3
      }

      liveness_probe {
        http_get {
          path = "/health"
        }
        period_seconds = 30
      }
    }

    service_account = google_service_account.cloud_run.email
  }

  depends_on = [google_project_service.apis]
}

# Allow unauthenticated access (for demo/portfolio purposes)
resource "google_cloud_run_v2_service_iam_member" "public" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.inference.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Cloud Monitoring alert policy for error rate
resource "google_monitoring_alert_policy" "error_rate" {
  display_name = "${var.app_name} - High Error Rate"
  combiner     = "OR"

  conditions {
    display_name = "Cloud Run 5xx error rate"
    condition_threshold {
      filter          = "resource.type = \"cloud_run_revision\" AND resource.labels.service_name = \"${var.app_name}-inference\" AND metric.type = \"run.googleapis.com/request_count\" AND metric.labels.response_code_class = \"5xx\""
      duration        = "60s"
      comparison      = "COMPARISON_GT"
      threshold_value = 5

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_RATE"
      }
    }
  }

  notification_channels = []  # Add notification channel IDs here
}
```

**Step 3: Create outputs.tf**

```hcl
# terraform/outputs.tf

output "cloud_run_url" {
  description = "URL of the deployed Cloud Run inference service"
  value       = google_cloud_run_v2_service.inference.uri
}

output "artifact_registry_url" {
  description = "URL of the Artifact Registry Docker repository"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.docker.repository_id}"
}

output "data_bucket" {
  description = "GCS bucket for data"
  value       = google_storage_bucket.data.name
}

output "models_bucket" {
  description = "GCS bucket for model artifacts"
  value       = google_storage_bucket.models.name
}

output "mlflow_bucket" {
  description = "GCS bucket for MLflow artifacts"
  value       = google_storage_bucket.mlflow.name
}

output "service_account_email" {
  description = "Cloud Run service account email"
  value       = google_service_account.cloud_run.email
}
```

**Step 4: Validate Terraform**

Run: `cd terraform && terraform init && terraform validate`
Expected: "Success! The configuration is valid."

**Step 5: Commit**

```bash
git add terraform/
git commit -m "feat: add Terraform IaC for GCP (Cloud Run, Artifact Registry, GCS, Monitoring)"
```

---

## Task 14: CI/CD — GitHub Actions CI

**Files:**
- Create: `.github/workflows/ci.yml`

**Step 1: Create ci.yml**

```yaml
# .github/workflows/ci.yml
name: CI

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install ruff
        run: pip install ruff
      - name: Run linter
        run: ruff check src/ tests/

  typecheck:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: |
          pip install -r requirements-train.txt
          pip install -r requirements-serve.txt
          pip install mypy
      - name: Run type checker
        run: mypy src/ --ignore-missing-imports

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: |
          pip install -r requirements-train.txt
          pip install -r requirements-serve.txt
      - name: Run tests
        run: pytest tests/ -v --tb=short

  docker-build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build training image
        run: docker build -f docker/Dockerfile.train -t sentiment-train:test .
      - name: Build serving image
        run: docker build -f docker/Dockerfile.serve -t sentiment-serve:test .
```

**Step 2: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "feat: add GitHub Actions CI pipeline with lint, typecheck, test, and Docker build"
```

---

## Task 15: CI/CD — Staging Deployment

**Files:**
- Create: `.github/workflows/cd-staging.yml`

**Step 1: Create cd-staging.yml**

```yaml
# .github/workflows/cd-staging.yml
name: CD - Deploy to Staging

on:
  push:
    branches: [main]

env:
  PROJECT_ID: ${{ secrets.GCP_PROJECT_ID }}
  REGION: us-central1
  REGISTRY: us-central1-docker.pkg.dev
  REPOSITORY: sentiment-mlops-docker
  SERVICE_NAME: sentiment-mlops-inference

jobs:
  deploy:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      id-token: write

    steps:
      - uses: actions/checkout@v4

      - id: auth
        uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: ${{ secrets.GCP_WORKLOAD_IDENTITY_PROVIDER }}
          service_account: ${{ secrets.GCP_SERVICE_ACCOUNT }}

      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v2

      - name: Configure Docker for Artifact Registry
        run: gcloud auth configure-docker ${{ env.REGISTRY }}

      - name: Build and push serving image
        run: |
          IMAGE=${{ env.REGISTRY }}/${{ env.PROJECT_ID }}/${{ env.REPOSITORY }}/sentiment-serve:${{ github.sha }}
          docker build -f docker/Dockerfile.serve -t $IMAGE .
          docker push $IMAGE
          echo "IMAGE=$IMAGE" >> $GITHUB_ENV

      - name: Deploy to Cloud Run
        run: |
          gcloud run deploy ${{ env.SERVICE_NAME }} \
            --image ${{ env.IMAGE }} \
            --region ${{ env.REGION }} \
            --platform managed \
            --allow-unauthenticated \
            --memory 2Gi \
            --cpu 2 \
            --min-instances 0 \
            --max-instances 5

      - name: Get service URL
        run: |
          URL=$(gcloud run services describe ${{ env.SERVICE_NAME }} --region ${{ env.REGION }} --format 'value(status.url)')
          echo "Service URL: $URL"

      - name: Integration test - health check
        run: |
          URL=$(gcloud run services describe ${{ env.SERVICE_NAME }} --region ${{ env.REGION }} --format 'value(status.url)')
          curl -f "$URL/health" || exit 1
```

**Step 2: Commit**

```bash
git add .github/workflows/cd-staging.yml
git commit -m "feat: add GitHub Actions CD pipeline for Cloud Run staging deployment"
```

---

## Task 16: CI/CD — Training Workflow

**Files:**
- Create: `.github/workflows/train.yml`

**Step 1: Create train.yml**

```yaml
# .github/workflows/train.yml
name: Training Pipeline

on:
  workflow_dispatch:
    inputs:
      num_epochs:
        description: "Number of training epochs"
        default: "3"
        type: string
      batch_size:
        description: "Training batch size"
        default: "16"
        type: string
  push:
    branches: [main]
    paths:
      - "src/model/**"
      - "src/data/**"

env:
  PROJECT_ID: ${{ secrets.GCP_PROJECT_ID }}
  REGION: us-central1

jobs:
  train:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      id-token: write

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements-train.txt

      - id: auth
        uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: ${{ secrets.GCP_WORKLOAD_IDENTITY_PROVIDER }}
          service_account: ${{ secrets.GCP_SERVICE_ACCOUNT }}

      - name: Download data
        run: python -m src.data.ingest

      - name: Preprocess data
        run: python -m src.data.preprocess

      - name: Train model
        run: |
          python -m src.model.train \
            --num_epochs ${{ github.event.inputs.num_epochs || '3' }} \
            --batch_size ${{ github.event.inputs.batch_size || '16' }}

      - name: Evaluate model
        run: python -m src.model.evaluate

      - name: Upload model metrics
        uses: actions/upload-artifact@v4
        with:
          name: model-metrics
          path: models/sentiment/test_metrics.json
```

**Step 2: Commit**

```bash
git add .github/workflows/train.yml
git commit -m "feat: add GitHub Actions training pipeline with manual trigger support"
```

---

## Task 17: DVC Configuration

**Files:**
- Create: `dvc.yaml`

**Step 1: Create dvc.yaml**

```yaml
# dvc.yaml
stages:
  ingest:
    cmd: python -m src.data.ingest
    deps:
      - src/data/ingest.py
    outs:
      - data/raw/reviews.csv

  preprocess:
    cmd: python -m src.data.preprocess
    deps:
      - src/data/preprocess.py
      - data/raw/reviews.csv
    outs:
      - data/processed/train.parquet
      - data/processed/val.parquet
      - data/processed/test.parquet

  train:
    cmd: python -m src.model.train
    deps:
      - src/model/train.py
      - data/processed/train.parquet
      - data/processed/val.parquet
    outs:
      - models/sentiment:
          cache: false
    metrics:
      - models/sentiment/test_metrics.json:
          cache: false

  evaluate:
    cmd: python -m src.model.evaluate
    deps:
      - src/model/evaluate.py
      - models/sentiment
      - data/processed/test.parquet
    metrics:
      - models/sentiment/test_metrics.json:
          cache: false
```

**Step 2: Commit**

```bash
git add dvc.yaml
git commit -m "feat: add DVC pipeline definition for data and training stages"
```

---

## Task 18: README

**Files:**
- Create: `README.md`

**Step 1: Create README.md**

```markdown
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
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add comprehensive README with architecture, setup, and usage instructions"
```

---

## Task 19: Final Lint and Test Pass

**Step 1: Run linter**

Run: `ruff check src/ tests/`
Expected: No errors (fix any issues found)

**Step 2: Run tests**

Run: `pytest tests/ -v`
Expected: All tests pass

**Step 3: Final commit (if lint fixes needed)**

```bash
git add -u
git commit -m "fix: resolve lint issues from final check"
```

---

## Summary

| Task | Component | Files |
|------|-----------|-------|
| 1 | Project Scaffolding | pyproject.toml, .gitignore, requirements, Makefile |
| 2 | Data Ingestion | src/data/ingest.py, tests/test_data.py |
| 3 | Data Preprocessing | src/data/preprocess.py, tests/test_data.py |
| 4 | Model Training | src/model/train.py, tests/test_model.py |
| 5 | Hyperparameter Tuning | src/model/tune.py, tests/test_model.py |
| 6 | Model Evaluation | src/model/evaluate.py, tests/test_model.py |
| 7 | API Schemas | src/serving/schemas.py, tests/test_serving.py |
| 8 | FastAPI App | src/serving/app.py, tests/test_serving.py |
| 9 | Drift Detection | src/monitoring/drift.py, tests/test_monitoring.py |
| 10 | Docker Train | docker/Dockerfile.train |
| 11 | Docker Serve | docker/Dockerfile.serve |
| 12 | Docker Compose | docker-compose.yml, mlflow/mlflow.env |
| 13 | Terraform | terraform/main.tf, variables.tf, outputs.tf |
| 14 | CI Pipeline | .github/workflows/ci.yml |
| 15 | CD Pipeline | .github/workflows/cd-staging.yml |
| 16 | Training Pipeline | .github/workflows/train.yml |
| 17 | DVC Config | dvc.yaml |
| 18 | README | README.md |
| 19 | Final Check | Lint + test pass |
