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
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, id2label=ID2LABEL, label2id=LABEL2ID,
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
        report_to="none",
    )


def compute_metrics(pred: EvalPrediction) -> dict:
    from sklearn.metrics import accuracy_score, f1_score
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1}


def load_data(data_dir: str = DEFAULT_DATA_DIR) -> tuple:
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
        output_dir=output_dir, num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate,
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    mlflow.set_experiment("sentiment-analysis")
    with mlflow.start_run(run_name="distilbert-train"):
        mlflow.log_params({
            "model_name": MODEL_NAME, "num_epochs": num_epochs,
            "batch_size": batch_size, "learning_rate": learning_rate,
            "max_length": max_length, "train_size": len(train_df), "val_size": len(val_df),
        })
        trainer.train()
        metrics = trainer.evaluate()
        mlflow.log_metrics({
            "val_accuracy": metrics["eval_accuracy"],
            "val_f1_macro": metrics["eval_f1_macro"],
            "val_loss": metrics["eval_loss"],
        })
        trainer.save_model(output_dir)
        train_dataset.tokenizer.save_pretrained(output_dir)
        mlflow.pytorch.log_model(model, "model")
        logger.info(f"Training complete. Metrics: {metrics}")
    return metrics


if __name__ == "__main__":
    metrics = train()
    print(json.dumps(metrics, indent=2))
