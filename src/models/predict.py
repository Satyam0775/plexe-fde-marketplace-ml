import joblib
import pandas as pd
from src.config import MODEL_PATH, FEATURE_COLS


def load_model():
    return joblib.load(MODEL_PATH)


def predict(data: dict):
    model = load_model()
    df = pd.DataFrame([data])

    # Ensure all features exist
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0

    df = df[FEATURE_COLS]

    probability = model.predict_proba(df)[0][1]
    prediction = model.predict(df)[0]

    # Feature importance explanation
    importances = model.feature_importances_
    feature_importance = dict(zip(FEATURE_COLS, importances))

    # Sort by importance
    sorted_features = sorted(
        feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )

    top_features = sorted_features[:2]

    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "top_contributing_features": top_features
    }