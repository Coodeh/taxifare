from fastapi import FastAPI
from pathlib import Path

import pandas as pd
import os
import numpy as np

from scipy.sparse import issparse

from taxifare.predict import load_model
from taxifare.predict import predict as model_predict

from taxifare.predict_rf import load_model_rf

from taxifare.predict_dl import load_model_dl
from taxifare.predict_dl import load_preprocessor


app = FastAPI()


# ==========================================
# MODEL PATHS
# ==========================================

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    str(MODELS_DIR / "linreg_pipeline.joblib")
)

RF_MODEL_PATH = os.getenv(
    "RF_MODEL_PATH",
    str(MODELS_DIR / "randomforest_pipeline.joblib")
)

KERAS_MODEL_PATH = os.getenv(
    "KERAS_MODEL_PATH",
    str(MODELS_DIR / "model_dl.keras")
)

PREPROCESSOR_PATH = os.getenv(
    "PREPROCESSOR_PATH",
    str(MODELS_DIR / "preprocessor.joblib")
)


# ==========================================
# LOAD MODELS
# ==========================================

model = load_model(MODEL_PATH)

model_rf = load_model_rf(RF_MODEL_PATH)

model_dl = load_model_dl(KERAS_MODEL_PATH)

preprocessor = load_preprocessor(PREPROCESSOR_PATH)


# ==========================================
# ROOT
# ==========================================

@app.get("/")
def root():
    """Check that the API is running."""

    return {"message": "Welcome To WagonCap API"}


# ==========================================
# LINEAR REGRESSION
# ==========================================

@app.get("/predict")
def predict(
    pickup_datetime: str,
    pickup_longitude: float,
    pickup_latitude: float,
    dropoff_longitude: float,
    dropoff_latitude: float,
    passenger_count: int
):
    """Predict the taxi fare using Linear Regression."""

    X = pd.DataFrame([{
        "pickup_datetime": pickup_datetime,
        "pickup_longitude": pickup_longitude,
        "pickup_latitude": pickup_latitude,
        "dropoff_longitude": dropoff_longitude,
        "dropoff_latitude": dropoff_latitude,
        "passenger_count": passenger_count
    }])

    prediction = model_predict(model, X)

    fare = round(float(prediction[0]), 2)

    return {"fare": fare}


# ==========================================
# RANDOM FOREST
# ==========================================

@app.get("/predict_rf")
def predict_rf(
    pickup_datetime: str,
    pickup_longitude: float,
    pickup_latitude: float,
    dropoff_longitude: float,
    dropoff_latitude: float,
    passenger_count: int
):
    """Predict the taxi fare using Random Forest."""

    X = pd.DataFrame([{
        "pickup_datetime": pickup_datetime,
        "pickup_longitude": pickup_longitude,
        "pickup_latitude": pickup_latitude,
        "dropoff_longitude": dropoff_longitude,
        "dropoff_latitude": dropoff_latitude,
        "passenger_count": passenger_count
    }])

    prediction = model_rf.predict(X)

    fare = round(float(prediction[0]), 2)

    return {"fare": fare}


# ==========================================
# DEEP LEARNING
# ==========================================

@app.get("/predict_dl")
def predict_dl(
    pickup_datetime: str,
    pickup_longitude: float,
    pickup_latitude: float,
    dropoff_longitude: float,
    dropoff_latitude: float,
    passenger_count: int
):
    """Predict the taxi fare using the Keras model."""

    X = pd.DataFrame([{
        "pickup_datetime": pickup_datetime,
        "pickup_longitude": pickup_longitude,
        "pickup_latitude": pickup_latitude,
        "dropoff_longitude": dropoff_longitude,
        "dropoff_latitude": dropoff_latitude,
        "passenger_count": passenger_count
    }])

    X_processed = preprocessor.transform(X)

    if issparse(X_processed):
        X_processed = X_processed.toarray()

    X_processed = np.asarray(
        X_processed,
        dtype=np.float32
    )

    prediction = model_dl.predict(
        X_processed,
        verbose=0
    ).ravel()

    fare = round(float(prediction[0]), 2)

    return {"fare": fare}
