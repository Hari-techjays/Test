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
    """Load Amazon product reviews from Hugging Face Datasets hub.

    Tries multiple strategies to download review data:
    1. Primary: McAuley-Lab/Amazon-Reviews-2023 with streaming
    2. Fallback: Try different split names
    3. Last resort: Generate synthetic review data for testing

    Args:
        dataset_name: HuggingFace dataset identifier.
        subset: Dataset subset/configuration name.
        sample_size: Number of reviews to load.
        seed: Random seed for reproducibility.

    Returns:
        pd.DataFrame with at least 'text' and 'rating' columns.
    """
    df = None

    # Strategy 1: Try loading with split slicing syntax
    strategies = [
        {"split": f"full[:{sample_size}]"},
        {"split": f"train[:{sample_size}]"},
        {"split": "full", "streaming": True},
        {"split": "train", "streaming": True},
    ]

    for strategy in strategies:
        try:
            streaming = strategy.pop("streaming", False)
            split = strategy["split"]
            logger.info(
                f"Trying to load {dataset_name}/{subset} "
                f"split={split} streaming={streaming}"
            )

            if streaming:
                dataset = load_dataset(
                    dataset_name,
                    subset,
                    split=split.replace(f"[:{sample_size}]", ""),
                    streaming=True,
                    trust_remote_code=True,
                )
                # Take sample_size items from streaming dataset
                rows = []
                for i, item in enumerate(dataset):
                    if i >= sample_size:
                        break
                    rows.append(item)
                df = pd.DataFrame(rows)
            else:
                dataset = load_dataset(
                    dataset_name,
                    subset,
                    split=split,
                    trust_remote_code=True,
                )
                df = dataset.to_pandas()

            if df is not None and len(df) > 0:
                logger.info(f"Successfully loaded {len(df)} rows with strategy: {strategy}")
                break
        except Exception as e:
            logger.warning(f"Strategy {strategy} failed: {e}")
            df = None
            continue

    # Strategy 2: Try alternative Amazon reviews datasets
    if df is None or len(df) == 0:
        alternative_datasets = [
            ("amazon_polarity", None, "test"),
            ("amazon_reviews_multi", "en", "test"),
        ]
        for alt_name, alt_subset, alt_split in alternative_datasets:
            try:
                logger.info(f"Trying alternative dataset: {alt_name}")
                if alt_subset:
                    dataset = load_dataset(
                        alt_name, alt_subset,
                        split=f"{alt_split}[:{sample_size}]",
                        trust_remote_code=True,
                    )
                else:
                    dataset = load_dataset(
                        alt_name,
                        split=f"{alt_split}[:{sample_size}]",
                        trust_remote_code=True,
                    )
                df = dataset.to_pandas()
                if len(df) > 0:
                    logger.info(f"Loaded {len(df)} rows from {alt_name}")
                    break
            except Exception as e:
                logger.warning(f"Alternative dataset {alt_name} failed: {e}")
                df = None
                continue

    # Strategy 3 (last resort): Generate synthetic data
    if df is None or len(df) == 0:
        logger.warning("All download strategies failed. Generating synthetic review data.")
        import random

        random.seed(seed)
        sample_texts = [
            "Great product, works perfectly! Highly recommend.",
            "Terrible quality. Broke after one week of use.",
            "Average product. Nothing special but does the job.",
            "Excellent value for money. Very satisfied with purchase.",
            "Not worth the price. Disappointed with this product.",
            "Good product overall. Minor issues but acceptable.",
            "Amazing quality and fast shipping. Love it!",
            "Product arrived damaged. Customer service was unhelpful.",
            "Decent product for the price. Would buy again.",
            "Worst purchase I've ever made. Complete waste of money.",
        ]
        ratings = [5, 1, 3, 5, 2, 4, 5, 1, 3, 1]
        n = min(sample_size, 1000)
        indices = [random.randint(0, len(sample_texts) - 1) for _ in range(n)]
        df = pd.DataFrame({
            "text": [sample_texts[i] for i in indices],
            "rating": [float(ratings[i]) for i in indices],
        })
        logger.info(f"Generated {len(df)} synthetic reviews")
        return df

    # Normalize column names
    column_map = {}
    if "text" in df.columns:
        column_map["text"] = "text"
    elif "reviewText" in df.columns:
        column_map["reviewText"] = "text"
    elif "content" in df.columns:
        column_map["content"] = "text"
    elif "review_body" in df.columns:
        column_map["review_body"] = "text"

    if "rating" in df.columns:
        column_map["rating"] = "rating"
    elif "overall" in df.columns:
        column_map["overall"] = "rating"
    elif "stars" in df.columns:
        column_map["stars"] = "rating"
    elif "star_rating" in df.columns:
        column_map["star_rating"] = "rating"
    elif "label" in df.columns:
        column_map["label"] = "rating"

    df = df.rename(columns=column_map)

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
        df: DataFrame containing review data.
        output_path: File path to save CSV to.

    Returns:
        The output path where the file was saved.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} reviews to {output_path}")
    return output_path


if __name__ == "__main__":
    df = load_reviews()
    save_raw(df)
