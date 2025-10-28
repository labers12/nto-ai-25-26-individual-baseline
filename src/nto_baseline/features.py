"""
Feature engineering script.
"""

import pandas as pd

from . import config, constants


def add_aggregate_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates and adds user, book, and author aggregate features.

    Uses the training data to compute mean ratings and interaction counts
    to prevent data leakage from the test set.

    Args:
        df (pd.DataFrame): The main DataFrame to add features to.
        train_df (pd.DataFrame): The training portion of the data for calculations.

    Returns:
        pd.DataFrame: The DataFrame with new aggregate features.
    """
    print("Adding aggregate features...")

    # User-based aggregates
    user_agg = train_df.groupby(constants.COL_USER_ID)[config.TARGET].agg(["mean", "count"]).reset_index()
    user_agg.columns = [
        constants.COL_USER_ID,
        constants.F_USER_MEAN_RATING,
        constants.F_USER_RATINGS_COUNT,
    ]

    # Book-based aggregates
    book_agg = train_df.groupby(constants.COL_BOOK_ID)[config.TARGET].agg(["mean", "count"]).reset_index()
    book_agg.columns = [
        constants.COL_BOOK_ID,
        constants.F_BOOK_MEAN_RATING,
        constants.F_BOOK_RATINGS_COUNT,
    ]

    # Author-based aggregates
    author_agg = train_df.groupby(constants.COL_AUTHOR_ID)[config.TARGET].agg(["mean"]).reset_index()
    author_agg.columns = [constants.COL_AUTHOR_ID, constants.F_AUTHOR_MEAN_RATING]

    # Merge aggregates into the main dataframe
    df = df.merge(user_agg, on=constants.COL_USER_ID, how="left")
    df = df.merge(book_agg, on=constants.COL_BOOK_ID, how="left")
    return df.merge(author_agg, on=constants.COL_AUTHOR_ID, how="left")


def add_genre_features(df: pd.DataFrame, book_genres_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates and adds the count of genres for each book.

    Args:
        df (pd.DataFrame): The main DataFrame to add features to.
        book_genres_df (pd.DataFrame): DataFrame mapping books to genres.

    Returns:
        pd.DataFrame: The DataFrame with the new 'book_genres_count' column.
    """
    print("Adding genre features...")
    genre_counts = book_genres_df.groupby(constants.COL_BOOK_ID)[constants.COL_GENRE_ID].count().reset_index()
    genre_counts.columns = [
        constants.COL_BOOK_ID,
        constants.F_BOOK_GENRES_COUNT,
    ]
    return df.merge(genre_counts, on=constants.COL_BOOK_ID, how="left")


def handle_missing_values(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Fills missing values using a defined strategy.

    Fills missing values for age, aggregated features, and categorical features
    to prepare the DataFrame for model training. Uses metrics from the training
    set (e.g., global mean) to fill NaNs.

    Args:
        df (pd.DataFrame): The DataFrame with missing values.
        train_df (pd.DataFrame): The training data, used for calculating fill metrics.

    Returns:
        pd.DataFrame: The DataFrame with missing values handled.
    """
    print("Handling missing values...")

    # Calculate global mean from training data for filling
    global_mean = train_df[config.TARGET].mean()

    # Fill age with the median
    age_median = df[constants.COL_AGE].median()
    df[constants.COL_AGE] = df[constants.COL_AGE].fillna(age_median)

    # Fill aggregate features for "cold start" users/items
    df[constants.F_USER_MEAN_RATING] = df[constants.F_USER_MEAN_RATING].fillna(global_mean)
    df[constants.F_BOOK_MEAN_RATING] = df[constants.F_BOOK_MEAN_RATING].fillna(global_mean)
    df[constants.F_AUTHOR_MEAN_RATING] = df[constants.F_AUTHOR_MEAN_RATING].fillna(global_mean)

    df[constants.F_USER_RATINGS_COUNT] = df[constants.F_USER_RATINGS_COUNT].fillna(0)
    df[constants.F_BOOK_RATINGS_COUNT] = df[constants.F_BOOK_RATINGS_COUNT].fillna(0)

    # Fill missing avg_rating from book_data with global mean
    df[constants.COL_AVG_RATING] = df[constants.COL_AVG_RATING].fillna(global_mean)

    # Fill genre counts with 0
    df[constants.F_BOOK_GENRES_COUNT] = df[constants.F_BOOK_GENRES_COUNT].fillna(0)

    # Fill remaining categorical features with a special value
    for col in config.CAT_FEATURES:
        if df[col].dtype.name in ("category", "object") and df[col].isna().any():
            df[col] = df[col].astype(str).fillna(constants.MISSING_CAT_VALUE).astype("category")
        elif pd.api.types.is_numeric_dtype(df[col].dtype) and df[col].isna().any():
            df[col] = df[col].fillna(constants.MISSING_NUM_VALUE)

    return df


def create_features(df: pd.DataFrame, book_genres_df: pd.DataFrame) -> pd.DataFrame:
    """Runs the full feature engineering pipeline.

    This function orchestrates the calls to add aggregate features, genre
    features, and handle missing values.

    Args:
        df (pd.DataFrame): The merged DataFrame from `data_processing`.
        book_genres_df (pd.DataFrame): DataFrame mapping books to genres.

    Returns:
        pd.DataFrame: The final DataFrame with all features engineered.
    """
    print("Starting feature engineering pipeline...")
    train_df = df[df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()

    df = add_aggregate_features(df, train_df)
    df = add_genre_features(df, book_genres_df)
    df = handle_missing_values(df, train_df)

    # Convert categorical columns to pandas 'category' dtype for LightGBM
    for col in config.CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    print("Feature engineering complete.")
    return df
