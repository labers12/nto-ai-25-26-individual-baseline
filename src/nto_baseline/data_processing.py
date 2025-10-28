"""
Data loading and merging script.
"""

from typing import Any

import pandas as pd

from . import config


def load_and_merge_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads all raw CSV files, merges train and test sets,
    and joins user and book metadata.

    Returns:
        A tuple containing:
        - The merged DataFrame with train/test data and metadata.
        - The book_genres DataFrame.
        - The genres DataFrame.
    """
    print("Loading data...")

    # Define dtypes for memory optimization
    dtype_spec: dict[str, Any] = {
        "user_id": "int32",
        "book_id": "int32",
        "rating": "float32",
        "gender": "category",
        "age": "float32",
        "author_id": "int32",
        "publication_year": "float32",
        "language": "category",
        "publisher": "category",
        "avg_rating": "float32",
        "genre_id": "int16",
    }

    # Load datasets
    train_df = pd.read_csv(
        config.RAW_DATA_DIR / "train.csv",
        dtype={k: v for k, v in dtype_spec.items() if k in ["user_id", "book_id", "rating"]},
    )
    test_df = pd.read_csv(
        config.RAW_DATA_DIR / "test.csv",
        dtype={k: v for k, v in dtype_spec.items() if k in ["user_id", "book_id"]},
    )
    user_data_df = pd.read_csv(
        config.RAW_DATA_DIR / "user_data.csv",
        dtype={k: v for k, v in dtype_spec.items() if k in ["user_id", "gender", "age"]},
    )
    book_data_df = pd.read_csv(
        config.RAW_DATA_DIR / "book_data.csv",
        dtype={
            k: v
            for k, v in dtype_spec.items()
            if k
            in [
                "book_id",
                "author_id",
                "publication_year",
                "language",
                "avg_rating",
                "publisher",
            ]
        },
    )
    book_genres_df = pd.read_csv(
        config.RAW_DATA_DIR / "book_genres.csv",
        dtype={k: v for k, v in dtype_spec.items() if k in ["book_id", "genre_id"]},
    )
    genres_df = pd.read_csv(config.RAW_DATA_DIR / "genres.csv")

    print("Data loaded. Merging datasets...")

    # Combine train and test
    train_df["source"] = "train"
    test_df["source"] = "test"
    combined_df = pd.concat([train_df, test_df], ignore_index=True, sort=False)

    # Join metadata
    combined_df = combined_df.merge(user_data_df, on="user_id", how="left")
    combined_df = combined_df.merge(book_data_df, on="book_id", how="left")

    print(f"Merged data shape: {combined_df.shape}")
    return combined_df, book_genres_df, genres_df
