"""
Pydantic Schemas for Heart Disease API

Defines request and response models for the prediction API.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from enum import Enum


class ChestPainType(int, Enum):
    """Chest pain type categories."""

    TYPICAL_ANGINA = 0
    ATYPICAL_ANGINA = 1
    NON_ANGINAL_PAIN = 2
    ASYMPTOMATIC = 3


class RestingECG(int, Enum):
    """Resting ECG result categories."""

    NORMAL = 0
    ST_T_ABNORMALITY = 1
    LEFT_VENTRICULAR_HYPERTROPHY = 2


class SlopeType(int, Enum):
    """Slope of peak exercise ST segment."""

    UPSLOPING = 0
    FLAT = 1
    DOWNSLOPING = 2


class ThalType(int, Enum):
    """Thalassemia type."""

    NORMAL = 0
    FIXED_DEFECT = 1
    REVERSIBLE_DEFECT = 2


class RiskLevel(str, Enum):
    """Risk level categories."""

    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"


class HeartDiseaseInput(BaseModel):
    """
    Input schema for heart disease prediction.

    All features required for the prediction model.
    """

    age: int = Field(..., ge=20, le=100, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (1 = male, 0 = female)")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: int = Field(
        ..., ge=80, le=250, description="Resting blood pressure (mm Hg)"
    )
    chol: int = Field(..., ge=100, le=600, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(
        ...,
        ge=0,
        le=1,
        description="Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)",
    )
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: int = Field(..., ge=60, le=220, description="Maximum heart rate achieved")
    exang: int = Field(
        ..., ge=0, le=1, description="Exercise induced angina (1 = yes, 0 = no)"
    )
    oldpeak: float = Field(
        ..., ge=0.0, le=10.0, description="ST depression induced by exercise"
    )
    slope: int = Field(
        ..., ge=0, le=2, description="Slope of peak exercise ST segment (0-2)"
    )
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels (0-4)")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia (0-3)")

    @field_validator("oldpeak")
    @classmethod
    def round_oldpeak(cls, v):
        """Round oldpeak to 1 decimal place."""
        return round(v, 1)

    class Config:
        json_schema_extra = {
            "example": {
                "age": 63,
                "sex": 1,
                "cp": 3,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 0,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 0,
                "ca": 0,
                "thal": 1,
            }
        }


class PredictionResponse(BaseModel):
    """
    Response schema for heart disease prediction.
    """

    prediction: int = Field(
        ..., description="Binary prediction (0 = no disease, 1 = disease)"
    )
    probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of heart disease"
    )
    risk_level: RiskLevel = Field(..., description="Risk level category")
    model_version: str = Field(..., description="Version of the prediction model")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")

    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 1,
                "probability": 0.85,
                "risk_level": "High",
                "model_version": "1.0.0",
                "confidence": 0.85,
            }
        }


class BatchPredictionInput(BaseModel):
    """
    Input schema for batch predictions.
    """

    instances: List[HeartDiseaseInput] = Field(
        ..., description="List of patient data for prediction"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "instances": [
                    {
                        "age": 63,
                        "sex": 1,
                        "cp": 3,
                        "trestbps": 145,
                        "chol": 233,
                        "fbs": 1,
                        "restecg": 0,
                        "thalach": 150,
                        "exang": 0,
                        "oldpeak": 2.3,
                        "slope": 0,
                        "ca": 0,
                        "thal": 1,
                    },
                    {
                        "age": 45,
                        "sex": 0,
                        "cp": 1,
                        "trestbps": 120,
                        "chol": 200,
                        "fbs": 0,
                        "restecg": 1,
                        "thalach": 170,
                        "exang": 0,
                        "oldpeak": 0.5,
                        "slope": 1,
                        "ca": 0,
                        "thal": 2,
                    },
                ]
            }
        }


class BatchPredictionResponse(BaseModel):
    """
    Response schema for batch predictions.
    """

    predictions: List[PredictionResponse] = Field(
        ..., description="List of predictions"
    )
    count: int = Field(..., description="Number of predictions")


class HealthResponse(BaseModel):
    """
    Response schema for health check endpoint.
    """

    status: str = Field(..., description="Service health status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: str = Field(..., description="Model version")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_version": "1.0.0",
            }
        }


class ModelInfoResponse(BaseModel):
    """
    Response schema for model information endpoint.
    """

    model_name: str = Field(..., description="Name of the model")
    model_version: str = Field(..., description="Model version")
    features: List[str] = Field(..., description="List of input features")
    created_at: str = Field(..., description="Model creation timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "random_forest",
                "model_version": "1.0.0",
                "features": [
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
                ],
                "created_at": "2024-01-15T10:30:00",
            }
        }


class ErrorResponse(BaseModel):
    """
    Response schema for error responses.
    """

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")

    class Config:
        json_schema_extra = {
            "example": {"error": "Prediction failed", "detail": "Model not loaded"}
        }
