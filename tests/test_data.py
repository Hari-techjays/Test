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


class TestPreprocess:
    def setup_method(self):
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
        assert len(result) <= len(self.sample_df)
        assert all(len(t) > 0 for t in result["text"])

    def test_split_data_proportions(self):
        from src.data.preprocess import split_data
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
