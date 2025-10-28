"""
Feature engineering script.
"""

import pandas as pd

from . import config


def add_aggregate_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates and adds aggregate features based on the training data.
    These features include means and counts for users, books, and authors.
    """
    print("Adding aggregate features...")

    # User-based aggregates
    user_agg = train_df.groupby("user_id")[config.TARGET].agg(["mean", "count"]).reset_index()
    user_agg.columns = ["user_id", "user_mean_rating", "user_ratings_count"]

    # Book-based aggregates
    book_agg = train_df.groupby("book_id")[config.TARGET].agg(["mean", "count"]).reset_index()
    book_agg.columns = ["book_id", "book_mean_rating", "book_ratings_count"]

    # Author-based aggregates
    author_agg = train_df.groupby("author_id")[config.TARGET].agg(["mean"]).reset_index()
    author_agg.columns = ["author_id", "author_mean_rating"]

    # Merge aggregates into the main dataframe
    df = df.merge(user_agg, on="user_id", how="left")
    df = df.merge(book_agg, on="book_id", how="left")
    return df.merge(author_agg, on="author_id", how="left")


def add_genre_features(df: pd.DataFrame, book_genres_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates and adds genre-based features.
    Currently, this is the count of genres per book.
    """
    print("Adding genre features...")
    genre_counts = book_genres_df.groupby("book_id")["genre_id"].count().reset_index()
    genre_counts.columns = ["book_id", "book_genres_count"]
    return df.merge(genre_counts, on="book_id", how="left")


def handle_missing_values(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing values in the dataframe according to the specified strategy.
    """
    print("Handling missing values...")

    # Calculate global mean from training data for filling
    global_mean = train_df[config.TARGET].mean()

    # Fill age with the median
    age_median = df["age"].median()
    df["age"] = df["age"].fillna(age_median)

    # Fill aggregate features for "cold start" users/items
    df["user_mean_rating"] = df["user_mean_rating"].fillna(global_mean)
    df["book_mean_rating"] = df["book_mean_rating"].fillna(global_mean)
    df["author_mean_rating"] = df["author_mean_rating"].fillna(global_mean)

    df["user_ratings_count"] = df["user_ratings_count"].fillna(0)
    df["book_ratings_count"] = df["book_ratings_count"].fillna(0)

    # Fill missing avg_rating from book_data with global mean
    df["avg_rating"] = df["avg_rating"].fillna(global_mean)

    # Fill genre counts with 0
    df["book_genres_count"] = df["book_genres_count"].fillna(0)

    # Fill remaining categorical features with a special value
    for col in config.CAT_FEATURES:
        if df[col].dtype.name in ("category", "object") and df[col].isna().any():
            df[col] = df[col].astype(str).fillna("-1").astype("category")
        elif pd.api.types.is_numeric_dtype(df[col].dtype) and df[col].isna().any():
            df[col] = df[col].fillna(-1)

    return df


def create_features(df: pd.DataFrame, book_genres_df: pd.DataFrame) -> pd.DataFrame:
    """
    Main feature engineering pipeline.
    """
    print("Starting feature engineering pipeline...")
    train_df = df[df["source"] == "train"].copy()

    df = add_aggregate_features(df, train_df)
    df = add_genre_features(df, book_genres_df)
    df = handle_missing_values(df, train_df)

    # Convert categorical columns to pandas 'category' dtype for LightGBM
    for col in config.CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    print("Feature engineering complete.")
    return df
