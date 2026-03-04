"""Hyperparameter tuning with Optuna for sentiment model."""

import logging
import os
from collections.abc import Callable

import optuna
import pandas as pd
from transformers import Trainer

import mlflow
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
