"""Data preprocessing module for review text cleaning and splitting."""
import html
import logging
import os
import re

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_PROCESSED_DIR = os.path.join("data", "processed")
LABEL_MAP = {1.0: "negative", 2.0: "negative", 3.0: "neutral", 4.0: "positive", 5.0: "positive"}


def clean_text(text: str | None) -> str:
    if text is None or not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def map_sentiment(rating: float) -> str:
    return LABEL_MAP[float(rating)]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"Preprocessing {len(df)} reviews")
    result = df.copy()
    result["text"] = result["text"].apply(clean_text)
    result = result[result["text"].str.len() > 0].reset_index(drop=True)
    result = result.dropna(subset=["rating"]).reset_index(drop=True)
    result["label"] = result["rating"].apply(map_sentiment)
    logger.info(f"After preprocessing: {len(result)} reviews")
    logger.info(f"Label distribution:\n{result['label'].value_counts()}")
    return result


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test_ratio = 1.0 - train_ratio - val_ratio

    # Use stratified splitting when possible; fall back to simple splitting
    # for small datasets where some classes have too few members.
    min_class_count = df["label"].value_counts().min()
    use_stratify = min_class_count >= 2

    stratify_col = df["label"] if use_stratify else None
    train_df, temp_df = train_test_split(
        df, test_size=(val_ratio + test_ratio), random_state=seed, stratify=stratify_col
    )

    # For very small temp splits, just divide evenly instead of using train_test_split
    if len(temp_df) < 2:
        val_df = temp_df
        test_df = temp_df.iloc[0:0]  # empty DataFrame with same columns
    else:
        relative_val_ratio = val_ratio / (val_ratio + test_ratio)
        stratify_temp = (
            temp_df["label"]
            if use_stratify and temp_df["label"].value_counts().min() >= 2
            else None
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=(1 - relative_val_ratio), random_state=seed, stratify=stratify_temp
        )

    logger.info(f"Split sizes — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: str = DEFAULT_PROCESSED_DIR,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for name, df in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(output_dir, f"{name}.parquet")
        df.to_parquet(path, index=False)
        logger.info(f"Saved {name} split ({len(df)} rows) to {path}")


if __name__ == "__main__":
    raw_path = os.path.join("data", "raw", "reviews.csv")
    df = pd.read_csv(raw_path)
    processed = preprocess(df)
    train, val, test = split_data(processed)
    save_splits(train, val, test)
