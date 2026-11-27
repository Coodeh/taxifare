import os
import io
import joblib
import numpy as np

from google.cloud import storage

def load_model_rf(model_path):

    if model_path.startswith("gs://"):

        bucket_name, blob_name = model_path[5:].split("/", 1)

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        return joblib.load(io.BytesIO(blob.download_as_bytes()))

    return joblib.load(model_path)

def predict_rf(X):

    model_path = os.getenv(
        "RF_MODEL_PATH",
        "models/randomforest_pipeline.joblib"
    )

    model = load_model_rf(model_path)

    return np.asarray(model.predict(X)).ravel()
