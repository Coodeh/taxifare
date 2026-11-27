import io
import joblib
import pandas as pd

from google.cloud import storage


def load_model(model_path):
    """Load a model from a local file or Google Cloud Storage."""

    model_path = str(model_path)

    if model_path.startswith("gs://"):

        bucket_name, blob_name = (
            model_path[5:].split("/", 1)
        )

        client = storage.Client()

        bucket = client.bucket(bucket_name)

        blob = bucket.blob(blob_name)

        model_bytes = blob.download_as_bytes()

        model = joblib.load(
            io.BytesIO(model_bytes)
        )

    else:

        model = joblib.load(model_path)

    return model


def predict(model,X):
    prediction = model.predict(X)
    return prediction

if __name__ == "__main__":
    X_dummy = pd.DataFrame([{
          'pickup_datetime': '2013-07-06 17:18:00 UTC',
          'pickup_longitude': -73.950655,
          'pickup_latitude': 40.783282,
          'dropoff_longitude': -73.984365,
          'dropoff_latitude': 40.769802,
          'passenger_count': 1
    }])

    model = load_model("models/linreg_pipeline.joblib")

    print(predict(model, X_dummy))
