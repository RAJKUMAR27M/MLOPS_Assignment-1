"""
Unit Tests for Model Training and Evaluation

Tests for model training, evaluation, and prediction functions.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.models.train import evaluate_model, cross_validate_model, MODEL_CONFIGS
from src.models.evaluate import (
    compute_all_metrics,
    compute_specificity,
    find_optimal_threshold,
    compare_models,
)


class TestModelTraining:
    """Tests for model training functions."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 200

        # Create features with some correlation to target
        X = pd.DataFrame(
            {
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
            }
        )

        # Create target with some signal from features
        y = pd.Series(
            ((X["age"] > 55) & (X["cp"] > 1)).astype(int)
            | (np.random.random(n_samples) > 0.7).astype(int)
        )

        return X, y

    @pytest.fixture
    def train_test_split(self, sample_data):
        """Split sample data into train and test sets."""
        X, y = sample_data
        split_idx = int(len(X) * 0.8)

        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]

        return X_train, X_test, y_train, y_test

    def test_model_configs_exist(self):
        """Test that model configurations are defined."""
        assert "logistic_regression" in MODEL_CONFIGS
        assert "random_forest" in MODEL_CONFIGS
        assert "gradient_boosting" in MODEL_CONFIGS

    def test_model_config_structure(self):
        """Test model config structure."""
        for model_name, config in MODEL_CONFIGS.items():
            assert "model_class" in config
            assert "default_params" in config
            assert "param_grid" in config

    def test_logistic_regression_training(self, train_test_split):
        """Test logistic regression model training."""
        X_train, X_test, y_train, y_test = train_test_split

        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        assert 0.0 <= accuracy <= 1.0
        assert len(predictions) == len(y_test)

    def test_random_forest_training(self, train_test_split):
        """Test random forest model training."""
        X_train, X_test, y_train, y_test = train_test_split

        model = RandomForestClassifier(random_state=42, n_estimators=50)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        assert 0.0 <= accuracy <= 1.0
        assert len(predictions) == len(y_test)

    def test_evaluate_model(self, train_test_split):
        """Test model evaluation function."""
        X_train, X_test, y_train, y_test = train_test_split

        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "roc_auc" in metrics

        # Check metric ranges
        for metric_name, value in metrics.items():
            assert 0.0 <= value <= 1.0, f"{metric_name} out of range: {value}"

    def test_cross_validate_model(self, sample_data):
        """Test cross-validation function."""
        X, y = sample_data

        model = LogisticRegression(random_state=42, max_iter=1000)
        cv_results = cross_validate_model(model, X, y, cv=3)

        assert "cv_accuracy_mean" in cv_results
        assert "cv_accuracy_std" in cv_results
        assert "cv_roc_auc_mean" in cv_results

        # Check that std is non-negative
        assert cv_results["cv_accuracy_std"] >= 0

    def test_model_predict_proba(self, train_test_split):
        """Test that models produce probability predictions."""
        X_train, X_test, y_train, y_test = train_test_split

        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)

        probas = model.predict_proba(X_test)

        assert probas.shape == (len(X_test), 2)
        assert np.allclose(probas.sum(axis=1), 1.0)
        assert (probas >= 0).all() and (probas <= 1).all()


class TestModelEvaluation:
    """Tests for model evaluation functions."""

    @pytest.fixture
    def predictions(self):
        """Create sample predictions for testing."""
        np.random.seed(42)
        n_samples = 100

        y_true = np.random.randint(0, 2, n_samples)
        y_pred = np.random.randint(0, 2, n_samples)
        y_prob = np.random.uniform(0, 1, n_samples)

        return y_true, y_pred, y_prob

    def test_compute_all_metrics(self, predictions):
        """Test computation of all metrics."""
        y_true, y_pred, y_prob = predictions

        metrics = compute_all_metrics(y_true, y_pred, y_prob)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "specificity" in metrics
        assert "roc_auc" in metrics
        assert "brier_score" in metrics

    def test_compute_specificity(self, predictions):
        """Test specificity computation."""
        y_true, y_pred, _ = predictions

        specificity = compute_specificity(y_true, y_pred)

        assert 0.0 <= specificity <= 1.0

    def test_compute_specificity_all_zeros(self):
        """Test specificity when all predictions are zeros."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])

        specificity = compute_specificity(y_true, y_pred)
        assert specificity == 1.0  # All negatives correctly predicted

    def test_find_optimal_threshold(self, predictions):
        """Test finding optimal threshold."""
        y_true, _, y_prob = predictions

        threshold, score = find_optimal_threshold(y_true, y_prob, metric="f1")

        assert 0.1 <= threshold <= 0.9
        assert 0.0 <= score <= 1.0

    def test_find_optimal_threshold_different_metrics(self, predictions):
        """Test finding optimal threshold with different metrics."""
        y_true, _, y_prob = predictions

        metrics = ["f1", "youden", "precision", "recall"]

        for metric in metrics:
            threshold, score = find_optimal_threshold(y_true, y_prob, metric=metric)
            assert 0.1 <= threshold <= 0.9

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.8, 0.15, 0.85])

        metrics = compute_all_metrics(y_true, y_pred, y_prob)

        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1_score"] == 1.0

    def test_worst_predictions(self):
        """Test metrics with completely wrong predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])

        metrics = compute_all_metrics(y_true, y_pred)

        assert metrics["accuracy"] == 0.0


class TestModelComparison:
    """Tests for model comparison functions."""

    @pytest.fixture
    def model_results(self):
        """Create sample model results."""
        model1 = LogisticRegression()
        model2 = RandomForestClassifier()

        results = {
            "logistic_regression": (
                model1,
                {
                    "test_accuracy": 0.85,
                    "test_roc_auc": 0.90,
                    "test_precision": 0.82,
                    "test_recall": 0.88,
                },
            ),
            "random_forest": (
                model2,
                {
                    "test_accuracy": 0.88,
                    "test_roc_auc": 0.92,
                    "test_precision": 0.85,
                    "test_recall": 0.90,
                },
            ),
        }

        return results

    def test_compare_models(self, model_results):
        """Test model comparison function."""
        comparison_df = compare_models(model_results)

        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == 2
        assert "accuracy" in comparison_df.columns
        assert "roc_auc" in comparison_df.columns

    def test_compare_models_sorted(self, model_results):
        """Test that models are sorted by ROC-AUC."""
        comparison_df = compare_models(model_results)

        # Should be sorted descending by roc_auc
        assert comparison_df.index[0] == "random_forest"


class TestModelPersistence:
    """Tests for model saving and loading."""

    @pytest.fixture
    def trained_model(self):
        """Create and train a simple model."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)

        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        return model, X, y

    def test_model_save_load(self, trained_model):
        """Test saving and loading a model."""
        import joblib

        model, X, y = trained_model

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            temp_path = f.name

        try:
            joblib.dump({"model": model, "version": "1.0.0"}, temp_path)

            loaded_data = joblib.load(temp_path)
            loaded_model = loaded_data["model"]

            # Predictions should match
            orig_pred = model.predict(X)
            loaded_pred = loaded_model.predict(X)

            np.testing.assert_array_equal(orig_pred, loaded_pred)
        finally:
            Path(temp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
