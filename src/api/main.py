"""
FastAPI Application for Heart Disease Prediction

This module provides a RESTful API for heart disease prediction
with health monitoring, logging, and Prometheus metrics.
"""

import os
import sys
import logging
import time
from pathlib import Path
from contextlib import asynccontextmanager

import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.schemas import (  # noqa: E402
    HeartDiseaseInput,
    PredictionResponse,
    BatchPredictionInput,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    RiskLevel,
)
from src.data.preprocessing import HeartDiseasePreprocessor  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
    ["endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "prediction_request_latency_seconds", "Request latency in seconds", ["endpoint"]
)

PREDICTION_DISTRIBUTION = Counter(
    "prediction_distribution_total",
    "Distribution of predictions",
    ["prediction", "risk_level"],
)

# Global model and preprocessor
model = None
preprocessor = None
model_info = None


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def load_model():
    """Load the trained model and preprocessor."""
    global model, preprocessor, model_info

    project_root = get_project_root()
    model_path = project_root / "models" / "final_model.joblib"
    preprocessor_path = project_root / "models" / "preprocessor.joblib"

    # Check for environment variable override
    if os.environ.get("MODEL_PATH"):
        model_path = Path(os.environ["MODEL_PATH"])
    if os.environ.get("PREPROCESSOR_PATH"):
        preprocessor_path = Path(os.environ["PREPROCESSOR_PATH"])

    try:
        if model_path.exists():
            model_data = joblib.load(model_path)
            model = model_data["model"]
            model_info = {
                "model_name": model_data.get("model_name", "unknown"),
                "version": model_data.get("version", "1.0.0"),
                "created_at": model_data.get("created_at", "unknown"),
            }
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning(f"Model file not found at {model_path}")
            model = None
            model_info = {
                "model_name": "not_loaded",
                "version": "0.0.0",
                "created_at": "N/A",
            }

        if preprocessor_path.exists():
            preprocessor = HeartDiseasePreprocessor.load(preprocessor_path)
            logger.info(f"Preprocessor loaded from {preprocessor_path}")
        else:
            logger.warning(f"Preprocessor file not found at {preprocessor_path}")
            preprocessor = None

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model = None
        preprocessor = None
        model_info = {"model_name": "error", "version": "0.0.0", "created_at": "N/A"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Heart Disease Prediction API...")
    load_model()
    yield
    # Shutdown
    logger.info("Shutting down Heart Disease Prediction API...")


# Create FastAPI application
app = FastAPI(
    title="Heart Disease Prediction API",
    description="""
    A machine learning API for predicting heart disease risk based on patient health data.

    ## Features
    - Single patient prediction
    - Batch prediction
    - Health monitoring
    - Prometheus metrics

    ## Model Information
    The API uses a trained classification model on the UCI Heart Disease dataset.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware for logging all requests."""
    start_time = time.time()

    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")

    response = await call_next(request)

    # Calculate latency
    latency = time.time() - start_time

    # Log response
    logger.info(f"Response: {response.status_code} - Latency: {latency:.3f}s")

    return response


def get_risk_level(probability: float) -> RiskLevel:
    """
    Determine risk level based on probability.

    Args:
        probability: Predicted probability of heart disease

    Returns:
        RiskLevel enum value
    """
    if probability < 0.3:
        return RiskLevel.LOW
    elif probability < 0.7:
        return RiskLevel.MODERATE
    else:
        return RiskLevel.HIGH


def prepare_input(data: HeartDiseaseInput) -> pd.DataFrame:
    """Convert a validated request into a one-row DataFrame for inference."""
    input_dict = data.model_dump()
    return pd.DataFrame([input_dict])


def _predict_from_dataframe(df: pd.DataFrame) -> tuple[int, float, RiskLevel, float]:
    """Run inference and return the prediction payload values."""
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])
    risk_level = get_risk_level(probability)
    confidence = max(probability, 1 - probability)
    return prediction, probability, risk_level, confidence


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Heart Disease Prediction API",
        "version": "1.0.0",
        "description": "ML-powered heart disease risk prediction",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "model_info": "/model/info",
            "metrics": "/metrics",
            "docs": "/docs",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Returns the current health status of the API.
    """
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        model_version=model_info.get("version", "0.0.0") if model_info else "0.0.0",
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """
    Get model information.

    Returns details about the loaded model.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
    ]

    return ModelInfoResponse(
        model_name=model_info.get("model_name", "unknown"),
        model_version=model_info.get("version", "1.0.0"),
        features=features,
        created_at=model_info.get("created_at", "unknown"),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(data: HeartDiseaseInput):
    """
    Predict heart disease risk for a single patient.

    - **age**: Age in years (20-100)
    - **sex**: Sex (1 = male, 0 = female)
    - **cp**: Chest pain type (0-3)
    - **trestbps**: Resting blood pressure (mm Hg)
    - **chol**: Serum cholesterol (mg/dl)
    - **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
    - **restecg**: Resting ECG results (0-2)
    - **thalach**: Maximum heart rate achieved
    - **exang**: Exercise induced angina (1 = yes, 0 = no)
    - **oldpeak**: ST depression induced by exercise
    - **slope**: Slope of peak exercise ST segment (0-2)
    - **ca**: Number of major vessels (0-4)
    - **thal**: Thalassemia (0-3)
    """
    start_time = time.time()

    if model is None:
        REQUEST_COUNT.labels(endpoint="predict", status="error").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Prepare input
        df = prepare_input(data)

        # Apply preprocessing if available
        if preprocessor is not None:
            df = preprocessor.transform(df)

        prediction, probability, risk_level, confidence = _predict_from_dataframe(df)

        # Update metrics
        REQUEST_COUNT.labels(endpoint="predict", status="success").inc()
        REQUEST_LATENCY.labels(endpoint="predict").observe(time.time() - start_time)
        PREDICTION_DISTRIBUTION.labels(
            prediction=str(prediction), risk_level=risk_level.value
        ).inc()

        logger.info(
            f"Prediction: {prediction}, Probability: {probability:.3f}, Risk: {risk_level.value}"
        )

        return PredictionResponse(
            prediction=prediction,
            probability=round(probability, 4),
            risk_level=risk_level,
            model_version=model_info.get("version", "1.0.0"),
            confidence=round(confidence, 4),
        )

    except Exception as e:
        REQUEST_COUNT.labels(endpoint="predict", status="error").inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def batch_predict(data: BatchPredictionInput):
    """
    Predict heart disease risk for multiple patients.

    Accepts a list of patient data and returns predictions for all.
    """
    start_time = time.time()

    if model is None:
        REQUEST_COUNT.labels(endpoint="batch_predict", status="error").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        predictions = []

        for instance in data.instances:
            # Prepare input
            df = prepare_input(instance)

            # Apply preprocessing if available
            if preprocessor is not None:
                df = preprocessor.transform(df)

            prediction, probability, risk_level, confidence = _predict_from_dataframe(df)

            predictions.append(
                PredictionResponse(
                    prediction=prediction,
                    probability=round(probability, 4),
                    risk_level=risk_level,
                    model_version=model_info.get("version", "1.0.0"),
                    confidence=round(confidence, 4),
                )
            )

            # Update prediction distribution
            PREDICTION_DISTRIBUTION.labels(
                prediction=str(prediction), risk_level=risk_level.value
            ).inc()

        # Update metrics
        REQUEST_COUNT.labels(endpoint="batch_predict", status="success").inc()
        REQUEST_LATENCY.labels(endpoint="batch_predict").observe(
            time.time() - start_time
        )

        logger.info(f"Batch prediction completed: {len(predictions)} instances")

        return BatchPredictionResponse(predictions=predictions, count=len(predictions))

    except Exception as e:
        REQUEST_COUNT.labels(endpoint="batch_predict", status="error").inc()
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus format for scraping.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500, content={"error": "Internal server error", "detail": str(exc)}
    )


# Run with: uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
