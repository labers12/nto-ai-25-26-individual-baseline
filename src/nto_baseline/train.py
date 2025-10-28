"""
Main training script for the LightGBM model.
"""

import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold

from . import config
from .data_processing import load_and_merge_data
from .features import create_features


def train() -> None:
    """
    Main function to run the training pipeline.
    """
    # Load and process data
    merged_df, book_genres_df, _ = load_and_merge_data()
    featured_df = create_features(merged_df, book_genres_df)

    # Separate train and test sets
    train_set = featured_df[featured_df["source"] == "train"].copy()

    # Define features (X) and target (y)
    features = [col for col in train_set.columns if col not in ["source", config.TARGET, "rating_predict"]]
    X = train_set[features]
    y = train_set[config.TARGET]
    groups = train_set["user_id"]

    # Ensure model directory exists
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Cross-validation training
    gkf = GroupKFold(n_splits=config.N_SPLITS)

    print(f"Starting training with {config.N_SPLITS}-fold GroupKFold CV...")

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"--- Fold {fold + 1}/{config.N_SPLITS} ---")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # Initialize and train the model
        model = lgb.LGBMRegressor(**config.LGB_PARAMS)

        # Update fit params with early stopping callback for the current fold
        fit_params = config.LGB_FIT_PARAMS.copy()
        fit_params["callbacks"] = [lgb.early_stopping(stopping_rounds=50, verbose=False)]

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=fit_params["eval_metric"],
            callbacks=fit_params["callbacks"],
        )

        # Evaluate the model
        val_preds = model.predict(X_val)
        rmse = mean_squared_error(y_val, val_preds, squared=False)
        mae = mean_absolute_error(y_val, val_preds)
        print(f"Fold {fold + 1} Validation RMSE: {rmse:.4f}, MAE: {mae:.4f}")

        # Save the trained model
        model_path = config.MODEL_DIR / f"lgb_fold_{fold}.txt"
        model.booster_.save_model(str(model_path))
        print(f"Model for fold {fold + 1} saved to {model_path}")

    print("Training complete.")


if __name__ == "__main__":
    train()
