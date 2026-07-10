"""
Unit Tests for FastAPI Application

Tests for API endpoints, request validation, and response formats.
"""

import pytest
import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient  # noqa: E402
from src.api.main import app, get_risk_level, prepare_input  # noqa: E402
from src.api.schemas import (  # noqa: E402
    HeartDiseaseInput,
    PredictionResponse,
    HealthResponse,
    RiskLevel,
)


class TestAPISchemas:
    """Tests for Pydantic schemas."""

    def test_valid_heart_disease_input(self):
        """Test valid input schema."""
        data = HeartDiseaseInput(
            age=63,
            sex=1,
            cp=3,
            trestbps=145,
            chol=233,
            fbs=1,
            restecg=0,
            thalach=150,
            exang=0,
            oldpeak=2.3,
            slope=0,
            ca=0,
            thal=1,
        )

        assert data.age == 63
        assert data.sex == 1
        assert data.oldpeak == 2.3

    def test_invalid_age(self):
        """Test validation for age out of range."""
        with pytest.raises(ValueError):
            HeartDiseaseInput(
                age=15,  # Too young
                sex=1,
                cp=3,
                trestbps=145,
                chol=233,
                fbs=1,
                restecg=0,
                thalach=150,
                exang=0,
                oldpeak=2.3,
                slope=0,
                ca=0,
                thal=1,
            )

    def test_invalid_sex(self):
        """Test validation for invalid sex value."""
        with pytest.raises(ValueError):
            HeartDiseaseInput(
                age=63,
                sex=2,  # Invalid
                cp=3,
                trestbps=145,
                chol=233,
                fbs=1,
                restecg=0,
                thalach=150,
                exang=0,
                oldpeak=2.3,
                slope=0,
                ca=0,
                thal=1,
            )

    def test_oldpeak_rounding(self):
        """Test that oldpeak is rounded to 1 decimal."""
        data = HeartDiseaseInput(
            age=63,
            sex=1,
            cp=3,
            trestbps=145,
            chol=233,
            fbs=1,
            restecg=0,
            thalach=150,
            exang=0,
            oldpeak=2.345,  # Should be rounded to 2.3
            slope=0,
            ca=0,
            thal=1,
        )

        assert data.oldpeak == 2.3

    def test_prediction_response_schema(self):
        """Test prediction response schema."""
        response = PredictionResponse(
            prediction=1,
            probability=0.85,
            risk_level=RiskLevel.HIGH,
            model_version="1.0.0",
            confidence=0.85,
        )

        assert response.prediction == 1
        assert response.risk_level == RiskLevel.HIGH

    def test_health_response_schema(self):
        """Test health response schema."""
        response = HealthResponse(
            status="healthy", model_loaded=True, model_version="1.0.0"
        )

        assert response.status == "healthy"


class TestRiskLevel:
    """Tests for risk level calculation."""

    def test_low_risk(self):
        """Test low risk level."""
        assert get_risk_level(0.1) == RiskLevel.LOW
        assert get_risk_level(0.25) == RiskLevel.LOW

    def test_moderate_risk(self):
        """Test moderate risk level."""
        assert get_risk_level(0.35) == RiskLevel.MODERATE
        assert get_risk_level(0.5) == RiskLevel.MODERATE
        assert get_risk_level(0.65) == RiskLevel.MODERATE

    def test_high_risk(self):
        """Test high risk level."""
        assert get_risk_level(0.75) == RiskLevel.HIGH
        assert get_risk_level(0.9) == RiskLevel.HIGH

    def test_boundary_values(self):
        """Test boundary values."""
        assert get_risk_level(0.3) == RiskLevel.MODERATE
        assert get_risk_level(0.7) == RiskLevel.HIGH


class TestPrepareInput:
    """Tests for input preparation."""

    def test_prepare_input_returns_dataframe(self):
        """Test that prepare_input returns a DataFrame."""
        import pandas as pd

        data = HeartDiseaseInput(
            age=63,
            sex=1,
            cp=3,
            trestbps=145,
            chol=233,
            fbs=1,
            restecg=0,
            thalach=150,
            exang=0,
            oldpeak=2.3,
            slope=0,
            ca=0,
            thal=1,
        )

        df = prepare_input(data)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "age" in df.columns


class TestAPIEndpoints:
    """Tests for API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def sample_input(self):
        """Sample valid input data."""
        return {
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

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data

    def test_predict_endpoint_valid_input(self, client, sample_input):
        """Test prediction endpoint with valid input."""
        # Mock model for testing
        with patch("src.api.main.model") as mock_model, patch(
            "src.api.main.preprocessor"
        ) as mock_preprocessor, patch("src.api.main.model_info", {"version": "1.0.0"}):

            mock_model.predict.return_value = np.array([1])
            mock_model.predict_proba.return_value = np.array([[0.15, 0.85]])
            mock_preprocessor.transform.return_value = None

            response = client.post("/predict", json=sample_input)

            assert response.status_code == 200
            data = response.json()
            assert "prediction" in data
            assert "probability" in data
            assert "risk_level" in data

    def test_predict_endpoint_invalid_input(self, client):
        """Test prediction endpoint with invalid input."""
        invalid_input = {
            "age": 15,  # Invalid age
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

        response = client.post("/predict", json=invalid_input)

        assert response.status_code == 422  # Validation error

    def test_predict_endpoint_missing_field(self, client):
        """Test prediction endpoint with missing field."""
        incomplete_input = {
            "age": 63,
            "sex": 1,
            "cp": 3,
            "trestbps": 145,
            # Missing other required fields
        }

        response = client.post("/predict", json=incomplete_input)

        assert response.status_code == 422  # Validation error

    def test_batch_predict_endpoint(self, client, sample_input):
        """Test batch prediction endpoint."""
        batch_input = {"instances": [sample_input, sample_input]}

        with patch("src.api.main.model") as mock_model, patch(
            "src.api.main.preprocessor"
        ) as mock_preprocessor, patch("src.api.main.model_info", {"version": "1.0.0"}):

            mock_model.predict.return_value = np.array([1])
            mock_model.predict_proba.return_value = np.array([[0.15, 0.85]])
            mock_preprocessor.transform.return_value = None

            response = client.post("/predict/batch", json=batch_input)

            assert response.status_code == 200
            data = response.json()
            assert "predictions" in data
            assert "count" in data
            assert data["count"] == 2

    def test_model_info_endpoint_no_model(self, client):
        """Test model info endpoint when model is not loaded."""
        with patch("src.api.main.model", None):
            response = client.get("/model/info")
            assert response.status_code == 503

    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")

        assert response.status_code == 200
        assert (
            "text/plain" in response.headers["content-type"]
            or "text/openmetrics-text" in response.headers["content-type"]
        )

    def test_predict_model_not_loaded(self, client, sample_input):
        """Test prediction when model is not loaded."""
        with patch("src.api.main.model", None):
            response = client.post("/predict", json=sample_input)

            assert response.status_code == 503


class TestAPIValidation:
    """Tests for input validation edge cases."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_boundary_age_values(self, client):
        """Test boundary age values."""
        base_input = {
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

        # Test minimum valid age
        min_age_input = {**base_input, "age": 20}
        response = client.post("/predict", json=min_age_input)
        # Should either succeed or fail gracefully (model not loaded)
        assert response.status_code in [200, 503]

        # Test maximum valid age
        max_age_input = {**base_input, "age": 100}
        response = client.post("/predict", json=max_age_input)
        assert response.status_code in [200, 503]

    def test_boundary_oldpeak_values(self, client):
        """Test boundary oldpeak values."""
        base_input = {
            "age": 63,
            "sex": 1,
            "cp": 3,
            "trestbps": 145,
            "chol": 233,
            "fbs": 1,
            "restecg": 0,
            "thalach": 150,
            "exang": 0,
            "slope": 0,
            "ca": 0,
            "thal": 1,
        }

        # Test minimum valid oldpeak
        min_oldpeak_input = {**base_input, "oldpeak": 0.0}
        response = client.post("/predict", json=min_oldpeak_input)
        assert response.status_code in [200, 503]

        # Test maximum valid oldpeak
        max_oldpeak_input = {**base_input, "oldpeak": 10.0}
        response = client.post("/predict", json=max_oldpeak_input)
        assert response.status_code in [200, 503]

    def test_negative_values(self, client):
        """Test that negative values are rejected."""
        invalid_input = {
            "age": -1,
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

        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422

    def test_string_instead_of_number(self, client):
        """Test that string values are rejected."""
        invalid_input = {
            "age": "sixty",
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

        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
