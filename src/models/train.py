"""
Model Training Script with MLflow Integration

This module trains classification models for heart disease prediction
with comprehensive experiment tracking using MLflow.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.preprocessing import (  # noqa: E402
    load_raw_data,
    prepare_data,
    HeartDiseasePreprocessor,
    handle_missing_values,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


# Model configurations
MODEL_CONFIGS = {
    "logistic_regression": {
        "model_class": LogisticRegression,
        "default_params": {"random_state": 42, "max_iter": 1000},
        "param_grid": {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear", "saga"],
        },
    },
    "random_forest": {
        "model_class": RandomForestClassifier,
        "default_params": {"random_state": 42, "n_jobs": -1},
        "param_grid": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
    },
    "gradient_boosting": {
        "model_class": GradientBoostingClassifier,
        "default_params": {"random_state": 42},
        "param_grid": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
        },
    },
    "svm": {
        "model_class": SVC,
        "default_params": {"random_state": 42, "probability": True},
        "param_grid": {
            "C": [0.1, 1, 10],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"],
        },
    },
}


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Evaluate model performance on test data.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels

    Returns:
        Dict containing evaluation metrics
    """
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    )

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_prob)

    return metrics


def cross_validate_model(
    model, X: pd.DataFrame, y: pd.Series, cv: int = 5
) -> Dict[str, float]:
    """
    Perform cross-validation and return metrics.

    Args:
        model: Model to evaluate
        X: Features
        y: Labels
        cv: Number of cross-validation folds

    Returns:
        Dict containing cross-validation metrics
    """
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    scoring_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    cv_results = {}

    for metric in scoring_metrics:
        scores = cross_val_score(model, X, y, cv=cv_strategy, scoring=metric)
        cv_results[f"cv_{metric}_mean"] = scores.mean()
        cv_results[f"cv_{metric}_std"] = scores.std()

    return cv_results


def plot_confusion_matrix(
    y_true: pd.Series, y_pred: np.ndarray, save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the plot

    Returns:
        matplotlib Figure
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    ax.set_xticklabels(["No Disease", "Disease"])
    ax.set_yticklabels(["No Disease", "Disease"])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Confusion matrix saved to {save_path}")

    return fig


def plot_roc_curve(
    y_true: pd.Series, y_prob: np.ndarray, save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot ROC curve.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        save_path: Path to save the plot

    Returns:
        matplotlib Figure
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {auc:.3f})")
    ax.plot(
        [0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random classifier"
    )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"ROC curve saved to {save_path}")

    return fig


def plot_feature_importance(
    model, feature_names: list, save_path: Optional[Path] = None
) -> Optional[plt.Figure]:
    """
    Plot feature importance for tree-based models.

    Args:
        model: Trained model
        feature_names: List of feature names
        save_path: Path to save the plot

    Returns:
        matplotlib Figure or None if not applicable
    """
    if not hasattr(model, "feature_importances_"):
        logger.info("Model does not have feature importances")
        return None

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(indices)), importances[indices], align="center")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Feature Importance")
    ax.set_title("Feature Importance Ranking")
    ax.invert_yaxis()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Feature importance plot saved to {save_path}")

    return fig


def train_model_with_mlflow(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    tune_hyperparameters: bool = False,
    experiment_name: str = "heart-disease-classification",
) -> Tuple[Any, Dict[str, float]]:
    """
    Train a model with MLflow tracking.

    Args:
        model_name: Name of the model to train
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        tune_hyperparameters: Whether to perform hyperparameter tuning
        experiment_name: MLflow experiment name

    Returns:
        Tuple of (trained model, metrics dictionary)
    """
    project_root = get_project_root()

    # Use a local file-based MLflow tracking store for reliable runs in this workspace.
    tracking_dir = project_root / "mlruns"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_dir.as_uri())
    mlflow.set_experiment(experiment_name)

    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}"
        )

    config = MODEL_CONFIGS[model_name]

    with mlflow.start_run(
        run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ):
        # Log parameters
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("tune_hyperparameters", tune_hyperparameters)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])

        if tune_hyperparameters:
            logger.info(f"Tuning hyperparameters for {model_name}...")

            # Perform grid search
            base_model = config["model_class"](**config["default_params"])
            grid_search = GridSearchCV(
                base_model,
                config["param_grid"],
                cv=5,
                scoring="roc_auc",
                n_jobs=-1,
                verbose=1,
            )
            grid_search.fit(X_train, y_train)

            model = grid_search.best_estimator_
            best_params = grid_search.best_params_

            # Log best parameters
            for param, value in best_params.items():
                mlflow.log_param(f"best_{param}", value)

            mlflow.log_param("best_cv_score", grid_search.best_score_)
            logger.info(f"Best parameters: {best_params}")
            logger.info(f"Best CV score: {grid_search.best_score_:.4f}")

        else:
            # Train with default parameters
            model = config["model_class"](**config["default_params"])
            model.fit(X_train, y_train)

            # Log default parameters
            for param, value in config["default_params"].items():
                mlflow.log_param(param, value)

        # Cross-validation metrics
        logger.info("Performing cross-validation...")
        cv_metrics = cross_validate_model(model, X_train, y_train)
        for metric_name, value in cv_metrics.items():
            mlflow.log_metric(metric_name, value)

        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_metrics = evaluate_model(model, X_test, y_test)
        for metric_name, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", value)

        # Print metrics
        logger.info(f"\n{model_name} Test Metrics:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        # Create artifacts directory
        artifacts_dir = project_root / "screenshots" / model_name
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Plot and log confusion matrix
        y_pred = model.predict(X_test)
        cm_path = artifacts_dir / "confusion_matrix.png"
        plot_confusion_matrix(y_test, y_pred, cm_path)
        mlflow.log_artifact(str(cm_path))

        # Plot and log ROC curve
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            roc_path = artifacts_dir / "roc_curve.png"
            plot_roc_curve(y_test, y_prob, roc_path)
            mlflow.log_artifact(str(roc_path))

        # Plot feature importance
        fi_path = artifacts_dir / "feature_importance.png"
        fig = plot_feature_importance(model, list(X_train.columns), fi_path)
        if fig:
            mlflow.log_artifact(str(fi_path))

        # Log classification report
        report = classification_report(y_test, y_pred)
        report_path = artifacts_dir / "classification_report.txt"
        report_path.write_text(report)
        mlflow.log_artifact(str(report_path))

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Close all plots
        plt.close("all")

        logger.info(f"MLflow run completed. Run ID: {mlflow.active_run().info.run_id}")

        return model, {**cv_metrics, **test_metrics}


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    tune_hyperparameters: bool = False,
) -> Dict[str, Tuple[Any, Dict[str, float]]]:
    """
    Train all available models.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        tune_hyperparameters: Whether to tune hyperparameters

    Returns:
        Dict mapping model names to (model, metrics) tuples
    """
    results = {}

    for model_name in MODEL_CONFIGS.keys():
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {model_name}")
        logger.info(f"{'='*50}")

        try:
            model, metrics = train_model_with_mlflow(
                model_name, X_train, y_train, X_test, y_test, tune_hyperparameters
            )
            results[model_name] = (model, metrics)
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")

    return results


def select_best_model(
    results: Dict[str, Tuple[Any, Dict[str, float]]],
) -> Tuple[str, Any, Dict[str, float]]:
    """
    Select the best model based on test ROC-AUC.

    Args:
        results: Dict mapping model names to (model, metrics) tuples

    Returns:
        Tuple of (best model name, best model, best metrics)
    """
    best_model_name = None
    best_model = None
    best_metrics = None
    best_score = 0

    for model_name, (model, metrics) in results.items():
        # Look for roc_auc first, then test_roc_auc, then accuracy
        score = metrics.get(
            "roc_auc", metrics.get("test_roc_auc", metrics.get("accuracy", 0))
        )
        if score > best_score:
            best_score = score
            best_model_name = model_name
            best_model = model
            best_metrics = metrics

    logger.info(f"\nBest model: {best_model_name} (ROC-AUC: {best_score:.4f})")
    return best_model_name, best_model, best_metrics


def save_model(model, preprocessor: HeartDiseasePreprocessor, model_name: str) -> Path:
    """
    Save the final model and preprocessor.

    Args:
        model: Trained model
        preprocessor: Fitted preprocessor
        model_name: Name of the model

    Returns:
        Path to saved model
    """
    project_root = get_project_root()
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = models_dir / "final_model.joblib"
    joblib.dump(
        {
            "model": model,
            "model_name": model_name,
            "version": "1.0.0",
            "created_at": datetime.now().isoformat(),
        },
        model_path,
    )
    logger.info(f"Model saved to {model_path}")

    # Save preprocessor
    preprocessor_path = models_dir / "preprocessor.joblib"
    preprocessor.save(preprocessor_path)

    return model_path


def main():
    """Main training function."""
    logger.info("=" * 60)
    logger.info("Heart Disease Model Training Pipeline")
    logger.info("=" * 60)

    # Load and preprocess data
    logger.info("\nLoading and preprocessing data...")
    df = load_raw_data()
    df = handle_missing_values(df)

    X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)

    logger.info("\nDataset statistics:")
    logger.info(f"  Training samples: {len(X_train)}")
    logger.info(f"  Test samples: {len(X_test)}")
    logger.info(f"  Features: {X_train.shape[1]}")

    # Train models
    logger.info("\nTraining models with MLflow tracking...")
    results = train_all_models(
        X_train, y_train, X_test, y_test, tune_hyperparameters=True
    )

    # Select best model
    best_model_name, best_model, best_metrics = select_best_model(results)

    # Save best model
    if best_model is not None:
        save_model(best_model, preprocessor, best_model_name)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"\nBest Model: {best_model_name}")
    if best_metrics:
        logger.info("Test Metrics:")
        for metric, value in best_metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {metric}: {value:.4f}")

    logger.info("\nArtifacts saved to: models/")
    logger.info("MLflow runs saved to: mlruns.db")
    logger.info("\nTo view MLflow UI, run: mlflow ui --host 0.0.0.0 --port 5000")


if __name__ == "__main__":
    main()
