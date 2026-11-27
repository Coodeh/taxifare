from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from pathlib import Path

import pandas as pd
import os
import numpy as np
import io

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
# DEF REQUIRED COLUMNS FROM FILE
# ==========================================

REQUIRED_COLUMNS = [
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count"
]
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


def read_csv(file):
    """Read and validate the uploaded CSV file."""

    try:
        content = file.file.read(5_000_001)

        if len(content) > 5_000_000:
            raise HTTPException(
                status_code=413,
                detail="Maximum file size: 5 MB"
            )

        df = pd.read_csv(
            io.BytesIO(content),
            nrows=10001
        )

    except (ValueError, UnicodeError):
        raise HTTPException(
            status_code=400,
            detail="Invalid CSV file"
        )

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="Empty CSV file"
        )

    if len(df) > 10000:
        raise HTTPException(
            status_code=413,
            detail="Maximum 10000 rows"
        )

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise HTTPException(
            status_code=422,
            detail=f"Missing columns: {missing_columns}"
        )

    X = df[REQUIRED_COLUMNS].copy()

    try:
        X["pickup_datetime"] = pd.to_datetime(
            X["pickup_datetime"],
            utc=True,
            errors="raise"
        )

        for col in REQUIRED_COLUMNS[1:]:
            X[col] = pd.to_numeric(
                X[col],
                errors="raise"
            )

    except (ValueError, TypeError):
        raise HTTPException(
            status_code=422,
            detail="Invalid date or numerical values"
        )

    if X.isna().any().any():
        raise HTTPException(
            status_code=422,
            detail="Missing values in CSV"
        )

    if not np.isfinite(
        X[REQUIRED_COLUMNS[1:]].to_numpy()
    ).all():
        raise HTTPException(
            status_code=422,
            detail="Non-finite values in CSV"
        )

    return df, X

def return_csv(df, predictions):
    """Add predictions and return a downloadable CSV."""

    predictions = np.asarray(predictions).ravel()

    if len(predictions) != len(df):
        raise HTTPException(
            status_code=500,
            detail="Invalid number of predictions"
        )

    if not np.isfinite(predictions).all():
        raise HTTPException(
            status_code=500,
            detail="Invalid model predictions"
        )

    df["fare"] = np.round(predictions, 2)

    return Response(
        content=df.to_csv(
            index=False,
            float_format="%.2f"
        ),
        media_type="text/csv",
        headers={
            "Content-Disposition":
                "attachment; filename=predictions.csv"
        }
    )

@app.post("/predict_batch")
def predict_batch(file: UploadFile = File(...)):
    """Predict multiple taxi fares using Ridge."""

    df, X = read_csv(file)

    predictions = model_predict(model, X)

    return return_csv(df, predictions)


@app.post("/predict_batch_rf")
def predict_batch_rf(file: UploadFile = File(...)):
    """Predict multiple taxi fares using Random Forest."""

    df, X = read_csv(file)

    predictions = model_rf.predict(X)

    return return_csv(df, predictions)


@app.post("/predict_batch_dl")
def predict_batch_dl(file: UploadFile = File(...)):
    """Predict multiple taxi fares using Keras."""

    df, X = read_csv(file)

    X_processed = preprocessor.transform(X)

    if issparse(X_processed):
        X_processed = X_processed.toarray()

    X_processed = np.asarray(
        X_processed,
        dtype=np.float32
    )

    predictions = model_dl.predict(
        X_processed,
        batch_size=256,
        verbose=0
    ).ravel()

    return return_csv(df, predictions)
