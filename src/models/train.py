import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.config import MODEL_PATH, FEATURE_COLS, EXPERIMENT_PATH
from src.models.evaluate import evaluate_model
from src.utils.logger import get_logger

logger = get_logger("training")


def train_model(df: pd.DataFrame):

    logger.info("Starting training pipeline...")

    # Ensure chronological sorting
    if "order_purchase_timestamp" not in df.columns:
        raise ValueError("order_purchase_timestamp column missing from dataset")

    df = df.sort_values("order_purchase_timestamp")

    # Chronological split (80/20)
    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    logger.info(f"Train size: {train_df.shape}")
    logger.info(f"Test size: {test_df.shape}")

    # Feature matrix
    X_train = train_df[FEATURE_COLS].copy()
    X_test = test_df[FEATURE_COLS].copy()

    y_train = train_df["low_rating"]
    y_test = test_df["low_rating"]

    # ---------------------------
    # Median Imputation (Train stats only)
    # ---------------------------
    for col in FEATURE_COLS:
        median = X_train[col].median()
        X_train[col] = X_train[col].fillna(median)
        X_test[col] = X_test[col].fillna(median)

    # ---------------------------
    # Model Definition
    # ---------------------------
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    )

    logger.info("Fitting RandomForest model...")
    model.fit(X_train, y_train)

    # ---------------------------
    # Evaluation
    # ---------------------------
    metrics = evaluate_model(model, X_test, y_test)

    logger.info("Saving trained model...")
    joblib.dump(model, MODEL_PATH)

    # ---------------------------
    # Save experiment results
    # ---------------------------
    results_df = pd.DataFrame([metrics])
    results_df.to_csv(EXPERIMENT_PATH, index=False)

    logger.info("Training complete.")

    return metrics