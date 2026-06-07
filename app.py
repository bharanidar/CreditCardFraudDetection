"""
Flask API for Credit Card Fraud Detection

- Loads trained model from models/fraud_model.pkl
- Provides a health-check GET endpoint at '/'
- Provides a POST '/predict' endpoint that accepts JSON with transaction features
  and returns whether the transaction is predicted as Fraud or Not Fraud.

The code is intentionally simple and well commented for beginners.
"""

import os
import joblib
from typing import Any, Dict

from flask import Flask, request, jsonify
import pandas as pd

MODEL_PATH = os.path.join("models", "fraud_model.pkl")

app = Flask(__name__)
model = None


def load_model(path: str):
    """Load a trained model from disk. Exits the app if the model is not found.
    """
    if not os.path.exists(path):
        app.logger.error(f"Model file not found at '{path}'. Please train the model first.")
        return None
    try:
        m = joblib.load(path)
        app.logger.info(f"Loaded model from {path}")
        return m
    except Exception as e:
        app.logger.error(f"Failed to load model: {e}")
        return None


@app.route("/", methods=["GET"])
def health_check():
    """Simple health-check endpoint."""
    return jsonify({"message": "Credit Card Fraud Detection API Running"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """Predict whether a transaction is fraud or not.

    Expects JSON body with keys matching the feature columns used during training.
    Example input (one row):
    {
        "Time": 12345,
        "V1": -1.3598071336738,
        "V2": -0.0727811733098497,
        ...
    }

    Returns JSON with 'prediction': 'Fraud' or 'Not Fraud'.
    """
    global model

    if model is None:
        return jsonify({"error": "Model not loaded. Make sure models/fraud_model.pkl exists."}), 500

    # Parse JSON input. Expect either a dict (single sample) or list of dicts.
    try:
        payload = request.get_json()
        if payload is None:
            return jsonify({"error": "Empty request body or invalid JSON."}), 400

        # If user sends a single dict, convert to list to build a DataFrame
        if isinstance(payload, dict):
            data = [payload]
        elif isinstance(payload, list):
            data = payload
        else:
            return jsonify({"error": "Invalid JSON format. Send a dict or a list of dicts."}), 400

        df = pd.DataFrame(data)
    except Exception as e:
        app.logger.error(f"Error parsing input JSON: {e}")
        return jsonify({"error": "Failed to parse input JSON."}), 400

    # Validate that dataframe has the right number of features if model exposes n_features_in_
    try:
        if hasattr(model, "n_features_in_"):
            expected = int(model.n_features_in_)
            if df.shape[1] != expected:
                return (
                    jsonify(
                        {
                            "error": "Feature mismatch",
                            "details": f"Model expects {expected} features but received {df.shape[1]} columns." 
                        }
                    ),
                    400,
                )

        # Ensure columns are numeric where possible
        df = df.apply(pd.to_numeric, errors="raise")

        preds = model.predict(df)

        # If multiple rows provided, return list of predictions
        def label(p: Any) -> str:
            return "Fraud" if int(p) == 1 else "Not Fraud"

        if len(preds) == 1:
            return jsonify({"prediction": label(preds[0])}), 200
        else:
            return jsonify({"predictions": [label(p) for p in preds]}), 200

    except ValueError as ve:
        app.logger.error(f"Value error during prediction: {ve}")
        return jsonify({"error": "Invalid feature values. Ensure all features are numeric."}), 400
    except Exception as e:
        app.logger.error(f"Unexpected error during prediction: {e}")
        return jsonify({"error": "Server error during prediction."}), 500


# Load model at startup
model = load_model(MODEL_PATH)


if __name__ == "__main__":
    # Use a simple dev server. In production use a WSGI server like gunicorn.
    app.run(host="0.0.0.0", port=5001, debug=True)