"""
Unit Tests for Data Processing Module

Tests for data loading, cleaning, and preprocessing functions.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import (  # noqa: E402
    HeartDiseasePreprocessor,
    handle_missing_values,
    analyze_data_quality,
    prepare_data,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
)


class TestDataPreprocessing:
    """Tests for data preprocessing functions."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 100

        data = {
            "age": np.random.randint(30, 80, n_samples),
            "sex": np.random.randint(0, 2, n_samples),
            "cp": np.random.randint(0, 4, n_samples),
            "trestbps": np.random.randint(90, 200, n_samples),
            "chol": np.random.randint(120, 400, n_samples),
            "fbs": np.random.randint(0, 2, n_samples),
            "restecg": np.random.randint(0, 3, n_samples),
            "thalach": np.random.randint(70, 200, n_samples),
            "exang": np.random.randint(0, 2, n_samples),
            "oldpeak": np.random.uniform(0, 6, n_samples).round(1),
            "slope": np.random.randint(0, 3, n_samples),
            "ca": np.random.randint(0, 4, n_samples),
            "thal": np.random.randint(0, 3, n_samples),
            "target": np.random.randint(0, 2, n_samples),
        }

        return pd.DataFrame(data)

    @pytest.fixture
    def data_with_missing(self, sample_data):
        """Create sample data with missing values."""
        df = sample_data.copy()
        # Introduce missing values
        df.loc[0:5, "age"] = np.nan
        df.loc[10:15, "chol"] = np.nan
        df.loc[20:25, "ca"] = np.nan
        return df

    def test_sample_data_shape(self, sample_data):
        """Test that sample data has correct shape."""
        assert sample_data.shape == (100, 14)
        assert TARGET_COLUMN in sample_data.columns

    def test_sample_data_types(self, sample_data):
        """Test that sample data has correct dtypes."""
        assert sample_data["age"].dtype in [np.int32, np.int64]
        assert sample_data["oldpeak"].dtype == np.float64

    def test_analyze_data_quality(self, sample_data):
        """Test data quality analysis function."""
        quality_report = analyze_data_quality(sample_data)

        assert "total_samples" in quality_report
        assert quality_report["total_samples"] == 100
        assert "missing_values" in quality_report
        assert "duplicates" in quality_report
        assert "target_distribution" in quality_report

    def test_handle_missing_values_no_missing(self, sample_data):
        """Test handling data without missing values."""
        result = handle_missing_values(sample_data)
        assert result.isnull().sum().sum() == 0
        assert len(result) == len(sample_data)

    def test_handle_missing_values_with_missing(self, data_with_missing):
        """Test handling data with missing values."""
        assert data_with_missing.isnull().sum().sum() > 0
        result = handle_missing_values(data_with_missing)
        assert result.isnull().sum().sum() == 0
        assert len(result) == len(data_with_missing)

    def test_preprocessor_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = HeartDiseasePreprocessor(scale_features=True)
        assert preprocessor.scale_features is True
        assert preprocessor.is_fitted is False

    def test_preprocessor_fit(self, sample_data):
        """Test preprocessor fitting."""
        X = sample_data.drop(columns=[TARGET_COLUMN])
        preprocessor = HeartDiseasePreprocessor(scale_features=True)

        preprocessor.fit(X)

        assert preprocessor.is_fitted is True
        assert preprocessor.feature_names is not None

    def test_preprocessor_transform(self, sample_data):
        """Test preprocessor transformation."""
        X = sample_data.drop(columns=[TARGET_COLUMN])
        preprocessor = HeartDiseasePreprocessor(scale_features=True)

        preprocessor.fit(X)
        X_transformed = preprocessor.transform(X)

        assert X_transformed.shape == X.shape
        assert not X_transformed.isnull().any().any()

    def test_preprocessor_fit_transform(self, sample_data):
        """Test preprocessor fit_transform."""
        X = sample_data.drop(columns=[TARGET_COLUMN])
        preprocessor = HeartDiseasePreprocessor(scale_features=True)

        X_transformed = preprocessor.fit_transform(X)

        assert preprocessor.is_fitted is True
        assert X_transformed.shape == X.shape

    def test_preprocessor_transform_without_fit(self, sample_data):
        """Test that transform raises error without fitting."""
        X = sample_data.drop(columns=[TARGET_COLUMN])
        preprocessor = HeartDiseasePreprocessor()

        with pytest.raises(ValueError, match="must be fitted"):
            preprocessor.transform(X)

    def test_preprocessor_scaling(self, sample_data):
        """Test that scaling normalizes features."""
        X = sample_data.drop(columns=[TARGET_COLUMN])
        preprocessor = HeartDiseasePreprocessor(scale_features=True)

        X_transformed = preprocessor.fit_transform(X)

        # Check that numeric features are scaled
        for col in NUMERIC_FEATURES:
            if col in X_transformed.columns:
                # Mean should be close to 0 and std close to 1
                assert abs(X_transformed[col].mean()) < 0.5
                assert abs(X_transformed[col].std() - 1) < 0.5

    def test_preprocessor_save_load(self, sample_data, tmp_path):
        """Test preprocessor serialization."""
        X = sample_data.drop(columns=[TARGET_COLUMN])
        preprocessor = HeartDiseasePreprocessor(scale_features=True)
        preprocessor.fit(X)

        # Save
        save_path = tmp_path / "preprocessor.joblib"
        preprocessor.save(save_path)
        assert save_path.exists()

        # Load
        loaded = HeartDiseasePreprocessor.load(save_path)
        assert loaded.is_fitted is True

        # Transform should produce same results
        X_orig = preprocessor.transform(X)
        X_loaded = loaded.transform(X)
        pd.testing.assert_frame_equal(X_orig, X_loaded)

    def test_prepare_data(self, sample_data):
        """Test data preparation function."""
        X_train, X_test, y_train, y_test, preprocessor = prepare_data(
            sample_data, test_size=0.2, random_state=42
        )

        # Check shapes
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20

        # Check preprocessor is fitted
        assert preprocessor.is_fitted

        # Check no missing values
        assert not X_train.isnull().any().any()
        assert not X_test.isnull().any().any()

    def test_target_column_not_in_features(self, sample_data):
        """Test that target column is not in features after prepare_data."""
        X_train, X_test, y_train, y_test, _ = prepare_data(sample_data)

        assert TARGET_COLUMN not in X_train.columns
        assert TARGET_COLUMN not in X_test.columns

    def test_stratified_split(self, sample_data):
        """Test that train/test split preserves class distribution."""
        X_train, X_test, y_train, y_test, _ = prepare_data(
            sample_data, test_size=0.2, random_state=42
        )

        train_ratio = y_train.mean()
        test_ratio = y_test.mean()

        # Ratios should be similar (within 10%)
        assert abs(train_ratio - test_ratio) < 0.1


class TestFeatureValidation:
    """Tests for feature validation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        return pd.DataFrame(
            {
                "age": [45, 55, 65],
                "sex": [1, 0, 1],
                "cp": [0, 1, 2],
                "trestbps": [120, 140, 160],
                "chol": [200, 250, 300],
                "fbs": [0, 1, 0],
                "restecg": [0, 1, 2],
                "thalach": [150, 140, 130],
                "exang": [0, 1, 0],
                "oldpeak": [1.0, 2.0, 3.0],
                "slope": [0, 1, 2],
                "ca": [0, 1, 2],
                "thal": [1, 2, 0],
                "target": [0, 1, 1],
            }
        )

    def test_numeric_features_exist(self, sample_data):
        """Test that all numeric features exist."""
        for feature in NUMERIC_FEATURES:
            assert feature in sample_data.columns

    def test_categorical_features_exist(self, sample_data):
        """Test that all categorical features exist."""
        for feature in CATEGORICAL_FEATURES:
            assert feature in sample_data.columns

    def test_binary_features_range(self, sample_data):
        """Test that binary features are 0 or 1."""
        binary_features = ["sex", "fbs", "exang"]
        for feature in binary_features:
            assert sample_data[feature].isin([0, 1]).all()

    def test_categorical_features_range(self, sample_data):
        """Test categorical features are within expected range."""
        assert sample_data["cp"].between(0, 3).all()
        assert sample_data["restecg"].between(0, 2).all()
        assert sample_data["slope"].between(0, 2).all()
        assert sample_data["ca"].between(0, 4).all()
        assert sample_data["thal"].between(0, 3).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
