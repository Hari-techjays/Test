"""Model evaluation module for sentiment classifier."""

import json
import logging
import os

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import mlflow
from src.model.train import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, LABEL2ID

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
