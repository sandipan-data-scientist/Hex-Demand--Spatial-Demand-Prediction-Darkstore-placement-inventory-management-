# FastAPI application with three endpoints:
#   GET  /           -> health check
#   POST /predict    -> single prediction
#   POST /predict/batch -> multiple predictions in one call

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import os

from app.schemas import (
    PredictRequest,
    PredictResponse,
    BatchPredictRequest,
    BatchPredictResponse,
)
from app.predict import predict_single, predict_batch

app = FastAPI(
    title       = "Hex-Demand Forecast API",
    description = (
        "Predicts grocery order demand density per H3 hexagon "
        "for a given hour using a trained Random Forest model. "
        "Built for Zepto-style dark store operations."
    ),
    version     = "1.0.0",
    docs_url    = "/docs",    # Swagger UI lives here
    redoc_url   = "/redoc",   # ReDoc lives here
)


# ---- Health Check ----

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Hex-Demand API is running"}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}


# ---- Single Prediction ----

@app.post(
    "/predict",
    response_model = PredictResponse,
    tags           = ["Prediction"],
    summary        = "Predict demand for a single hexagon-hour",
    description    = (
        "Accepts the previous hour's demand (lag_1) and the current "
        "hour of day, returns predicted demand count and demand tier."
    )
)
def predict(request: PredictRequest):
    try:
        result = predict_single(lag_1=request.lag_1, hour=request.hour)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---- Batch Prediction ----

@app.post(
    "/predict/batch",
    response_model = BatchPredictResponse,
    tags           = ["Prediction"],
    summary        = "Predict demand for multiple hexagon-hour records",
    description    = (
        "Accepts a list of hexagon-hour records and returns a list of "
        "predictions. Useful for generating a full city snapshot in one call."
    )
)
def predict_batch_endpoint(request: BatchPredictRequest):
    try:
        results = predict_batch(request.records)
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---- Data Summary Endpoint ----

@app.get(
    "/data/summary",
    tags    = ["Data"],
    summary = "Returns summary stats from the processed hex demand CSV"
)
def data_summary():
    csv_path = os.path.join(
        os.path.dirname(__file__), '..', 'artifacts', 'hex_demand_processed.csv'
    )
    if not os.path.exists(csv_path):
        raise HTTPException(
            status_code = 404,
            detail      = "hex_demand_processed.csv not found. Run train_and_save.py first."
        )
    df = pd.read_csv(csv_path)
    return {
        "total_records"    : len(df),
        "unique_hexagons"  : int(df['h3_index'].nunique()),
        "mean_demand"      : round(float(df['demand'].mean()), 2),
        "max_demand"       : int(df['demand'].max()),
        "hours_covered"    : sorted(df['hour'].unique().tolist()),
    }