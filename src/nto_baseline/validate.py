"""
Script to validate the format of the submission file.
"""

import pandas as pd

from . import config


def validate() -> None:
    """
    Performs checks on the submission file.
    """
    print("Validating submission file...")

    try:
        # Load test data and submission file
        test_df = pd.read_csv(config.RAW_DATA_DIR / "test.csv")
        sub_df = pd.read_csv(config.SUBMISSION_DIR / "baseline_submission.csv")

        # 1. Check length
        assert len(sub_df) == len(test_df), f"Submission length mismatch. Expected {len(test_df)}, got {len(sub_df)}."
        print("✅ Length check passed.")

        # 2. Check for missing values in prediction
        assert not sub_df["rating_predict"].isna().any(), "Missing values found in 'rating_predict'."
        print("✅ No missing values check passed.")

        # 3. Check that the set of (user_id, book_id) pairs match
        test_keys = test_df[["user_id", "book_id"]].copy().set_index(["user_id", "book_id"])
        sub_keys = sub_df[["user_id", "book_id"]].copy().set_index(["user_id", "book_id"])

        assert test_keys.index.equals(
            sub_keys.index
        ), "The set of (user_id, book_id) pairs does not match the test set."
        print("✅ (user_id, book_id) pair matching check passed.")

        # 4. Check prediction range
        assert sub_df["rating_predict"].between(0, 10).all(), "Predictions are not within the [0, 10] range."
        print("✅ Prediction range [0, 10] check passed.")

        print("\nValidation successful! The submission file appears to be in the correct format.")

    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure the required files exist.")
    except AssertionError as e:
        print(f"Validation failed: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    validate()
