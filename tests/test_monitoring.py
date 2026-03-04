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
        assert result["drift_detected"] == False

    def test_detect_length_drift_with_drift(self):
        from src.monitoring.drift import detect_length_drift

        reference = ["short"] * 100
        current = ["this is a very long text with many many words " * 10] * 50
        result = detect_length_drift(reference, current)
        assert result["drift_detected"] == True

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
