from fastapi import FastAPI
from pathlib import Path

import pandas as pd
import os

from taxifare.predict import load_model
from taxifare.predict import predict as model_predict


app = FastAPI()

LOCAL_MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "linreg_pipeline.joblib"
)

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    str(LOCAL_MODEL_PATH)
)

model = load_model(MODEL_PATH)



@app.get("/")
def root():
    """Check that the API is running."""

    return {"message": "Welcome To WagonCap API"}


@app.get("/predict")
def predict(
    pickup_datetime: str,
    pickup_longitude: float,
    pickup_latitude: float,
    dropoff_longitude: float,
    dropoff_latitude: float,
    passenger_count: int
):
    """Predict the taxi fare using the trained pipeline."""

    X = pd.DataFrame([{
        "pickup_datetime": pickup_datetime,
        "pickup_longitude": pickup_longitude,
        "pickup_latitude": pickup_latitude,
        "dropoff_longitude": dropoff_longitude,
        "dropoff_latitude": dropoff_latitude,
        "passenger_count": passenger_count
    }])

    prediction = model_predict(model, X)

    """Round to 2 digits"""
    fare = round(float(prediction[0]), 2)

    return {"fare": fare}
