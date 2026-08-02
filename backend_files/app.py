"""
SuperKart Sales Forecasting - Flask Backend API
Serves the trained ML model via REST endpoints for single and batch inference.
"""

import os
import io
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

# Initialize Flask application
superkart_api = Flask(__name__)

# Load the serialized model safely at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "superkart_model.joblib")
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

# Expected feature columns (must match training order)
FEATURE_COLS = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category",
]


# Health-check endpoint
@superkart_api.get("/")
def health():
    """Returns a simple health-check message."""
    status_msg = "SuperKart Forecasting API is running."
    if model is None:
        status_msg += " (Warning: superkart_model.joblib not loaded yet)"
    return jsonify({"status": "ok", "message": status_msg})


# Online (single-record) inference endpoint
@superkart_api.post("/v1/predict")
def predict():
    """
    Accepts a JSON payload with one record and returns the predicted
    Product_Store_Sales_Total.
    """
    try:
        if model is None:
            return jsonify({"error": "Model file superkart_model.joblib is missing. Please train and upload the model file."}), 500

        data = request.get_json(force=True)
        missing = [col for col in FEATURE_COLS if col not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400
        input_df = pd.DataFrame([{col: data[col] for col in FEATURE_COLS}])
        prediction = model.predict(input_df)[0]
        return jsonify({"prediction": round(float(prediction), 2)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# Batch inference endpoint
@superkart_api.post("/v1/predictbatch")
def predict_batch():
    """
    Accepts a CSV file upload and returns predictions for every row.
    Returns a JSON object mapping row-index to predicted sales.
    """
    try:
        if model is None:
            return jsonify({"error": "Model file superkart_model.joblib is missing. Please train and upload the model file."}), 500

        if "file" not in request.files:
            return jsonify({"error": "No file part in request. Use key='file'."}), 400
        file = request.files["file"]
        content = file.read()
        batch_df = pd.read_csv(io.BytesIO(content))
        missing = [col for col in FEATURE_COLS if col not in batch_df.columns]
        if missing:
            return jsonify({"error": f"Missing columns in CSV: {missing}"}), 400
        input_df = batch_df[FEATURE_COLS]
        predictions = model.predict(input_df)
        result = {str(i): round(float(p), 2) for i, p in enumerate(predictions)}
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# Entry point
if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860, debug=False)
