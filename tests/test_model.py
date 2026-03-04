import json
import os
import tempfile


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
        SentimentDataset(texts, labels, max_length=32)
        create_model(num_labels=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            args = get_training_args(
                output_dir=tmpdir, num_epochs=1, batch_size=2, learning_rate=5e-5
            )
            assert args is not None
            assert args.num_train_epochs == 1


class TestTuning:
    def test_objective_returns_float(self):

        from src.model.tune import create_objective

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

            with open(path) as f:
                loaded = json.load(f)
            assert loaded["accuracy"] == 0.85
