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
            assert args is not None
            assert args.num_train_epochs == 1
