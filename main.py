import io
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from google.cloud import storage

app = Flask(__name__)
model = None


def _load_model():
    file_path = "model.joblib"
    model = joblib.load(file_path)
    return model


def load_model():
    storage_client = storage.Client()
    bucket = storage_client.bucket("my-first-project-466020-bucket")
    blob = bucket.blob("advertising_roi_artifact/model.joblib")
    model_bytes = blob.download_as_bytes()
    model = joblib.load(io.BytesIO(model_bytes))
    return model


def preprocess(input_json):
    try:
        df = pd.DataFrame(input_json, index=[0])
        df.fillna(0, inplace=True)
        return df

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict", methods=["POST"])
def predict():
    model = load_model()
    try:
        input_json = request.get_json()
        expected_order = model.feature_names_in_

        df_preprocessed = preprocess(input_json)
        df_preprocessed = df_preprocessed[expected_order]

        y_predictions = model.predict(df_preprocessed)
        response = {"predictions": y_predictions.tolist()}
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5051)))
