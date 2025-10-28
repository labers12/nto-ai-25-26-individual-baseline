"""
Inference script to generate predictions for the test set.
"""

import lightgbm as lgb
import numpy as np

from . import config
from .data_processing import load_and_merge_data
from .features import create_features


def predict() -> None:
    """
    Main function to run the inference pipeline and generate a submission file.
    """
    # Load and process data
    merged_df, book_genres_df, _ = load_and_merge_data()
    featured_df = create_features(merged_df, book_genres_df)

    # Separate test set for prediction
    test_set = featured_df[featured_df["source"] == "test"].copy()

    # Define features
    features = [col for col in test_set.columns if col not in ["source", config.TARGET, "rating_predict"]]
    X_test = test_set[features]

    # Generate predictions from all fold models
    test_preds = []
    print(f"Loading {config.N_SPLITS} models and generating predictions...")

    for fold in range(config.N_SPLITS):
        model_path = config.MODEL_DIR / f"lgb_fold_{fold}.txt"
        print(f"Loading model from {model_path}")
        model = lgb.Booster(model_file=str(model_path))
        fold_preds = model.predict(X_test)
        test_preds.append(fold_preds)

    # Average the predictions
    avg_preds = np.mean(test_preds, axis=0)

    # Clip predictions to be within the valid rating range [0, 10]
    clipped_preds = np.clip(avg_preds, 0, 10)

    # Create submission file
    submission_df = test_set[["user_id", "book_id"]].copy()
    submission_df["rating_predict"] = clipped_preds

    # Ensure submission directory exists
    config.SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = config.SUBMISSION_DIR / "baseline_submission.csv"

    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file created at: {submission_path}")


if __name__ == "__main__":
    predict()
