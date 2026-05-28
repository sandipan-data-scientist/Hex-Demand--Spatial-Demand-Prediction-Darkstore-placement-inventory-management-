# Loads the saved model once at startup and exposes a predict function.
# Loading once and reusing is critical for API performance.
# If we loaded the model on every request the API would be extremely slow.

import joblib
import numpy as np
import os

# Resolve path relative to this file so it works regardless of
# where the process is started from (local or inside Docker)
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'model.pkl')

# Load model into memory once when this module is imported
model = joblib.load(MODEL_PATH)


def classify_demand(value: float) -> str:
    # Simple tiering based on fixed thresholds
    # In production these thresholds would be derived from
    # the training data's percentile distribution
    if value >= 10:
        return "High"
    elif value >= 5:
        return "Medium"
    else:
        return "Low"


def predict_single(lag_1: float, hour: int) -> dict:
    features = np.array([[lag_1, hour]])
    prediction = float(model.predict(features)[0])
    prediction = max(prediction, 0.0)  # demand cannot be negative
    return {
        "predicted_demand": round(prediction, 2),
        "demand_tier"     : classify_demand(prediction),
    }


def predict_batch(records: list) -> list:
    results = []
    for record in records:
        result = predict_single(record.lag_1, record.hour)
        results.append(result)
    return results