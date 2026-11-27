import os
import io
import joblib
import tempfile
import numpy as np

from pathlib import Path
from google.cloud import storage
from scipy.sparse import issparse

def load_model_dl(model_path):

    from tensorflow import keras

    if model_path.startswith("gs://"):

        bucket_name, blob_name = model_path[5:].split("/", 1)

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        with tempfile.TemporaryDirectory() as tmpdir:

            local_path = Path(tmpdir) / "model.keras"

            blob.download_to_filename(str(local_path))

            return keras.models.load_model(
                local_path,
                compile=False
            )

    return keras.models.load_model(
        model_path,
        compile=False
    )

def load_preprocessor(preprocessor_path):

    if preprocessor_path.startswith("gs://"):

        bucket_name, blob_name = preprocessor_path[5:].split("/", 1)

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        return joblib.load(io.BytesIO(blob.download_as_bytes()))

    return joblib.load(preprocessor_path)

def predict_dl(X):

    model_path = os.getenv(
        "KERAS_MODEL_PATH",
        "models/model_dl.keras"
    )

    preprocessor_path = os.getenv(
        "PREPROCESSOR_PATH",
        "models/preprocessor.joblib"
    )

    model = load_model_dl(model_path)

    preprocessor = load_preprocessor(preprocessor_path)

    X_processed = preprocessor.transform(X)

    if issparse(X_processed):
        X_processed = X_processed.toarray()

    X_processed = np.asarray(X_processed, dtype=np.float32)

    predictions = model.predict(X_processed, verbose=0)

    return predictions.ravel()
