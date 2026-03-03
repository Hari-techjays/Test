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
