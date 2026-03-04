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
    test_df = pd.read_parquet(os.path.join("data", "processed", "test.parquet"))

    report = compute_drift_report(
        reference_texts=train_df["text"].tolist(),
        current_texts=test_df["text"].tolist(),
    )
    print(json.dumps(report, indent=2))
