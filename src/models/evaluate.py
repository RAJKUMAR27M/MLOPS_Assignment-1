"""
Model Evaluation Utilities

This module provides comprehensive model evaluation functions
including metrics computation, visualization, and reporting.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import logging

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def compute_all_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute all classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)

    Returns:
        Dict containing all metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "specificity": compute_specificity(y_true, y_pred),
    }

    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        metrics["average_precision"] = average_precision_score(y_true, y_prob)
        metrics["brier_score"] = compute_brier_score(y_true, y_prob)

    return metrics


def compute_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute specificity (true negative rate).

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Specificity score
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute Brier score for calibration assessment.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities

    Returns:
        Brier score (lower is better)
    """
    return np.mean((y_prob - y_true) ** 2)


def generate_classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, target_names: List[str] = None
) -> str:
    """
    Generate detailed classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Names for target classes

    Returns:
        Classification report string
    """
    if target_names is None:
        target_names = ["No Disease", "Disease"]

    return classification_report(y_true, y_pred, target_names=target_names)


def plot_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    save_dir: Optional[Path] = None,
) -> Dict[str, plt.Figure]:
    """
    Generate all evaluation plots.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        save_dir: Directory to save plots

    Returns:
        Dict mapping plot names to Figure objects
    """
    figures = {}

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # Confusion Matrix
    fig_cm = plot_confusion_matrix_detailed(y_true, y_pred)
    figures["confusion_matrix"] = fig_cm
    if save_dir:
        fig_cm.savefig(save_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")

    # ROC Curve
    fig_roc = plot_roc_curve_detailed(y_true, y_prob)
    figures["roc_curve"] = fig_roc
    if save_dir:
        fig_roc.savefig(save_dir / "roc_curve.png", dpi=150, bbox_inches="tight")

    # Precision-Recall Curve
    fig_pr = plot_precision_recall_curve(y_true, y_prob)
    figures["precision_recall_curve"] = fig_pr
    if save_dir:
        fig_pr.savefig(
            save_dir / "precision_recall_curve.png", dpi=150, bbox_inches="tight"
        )

    # Calibration Curve
    fig_cal = plot_calibration_curve(y_true, y_prob)
    figures["calibration_curve"] = fig_cal
    if save_dir:
        fig_cal.savefig(
            save_dir / "calibration_curve.png", dpi=150, bbox_inches="tight"
        )

    # Prediction Distribution
    fig_dist = plot_prediction_distribution(y_true, y_prob)
    figures["prediction_distribution"] = fig_dist
    if save_dir:
        fig_dist.savefig(
            save_dir / "prediction_distribution.png", dpi=150, bbox_inches="tight"
        )

    plt.close("all")

    return figures


def plot_confusion_matrix_detailed(
    y_true: np.ndarray, y_pred: np.ndarray
) -> plt.Figure:
    """
    Plot detailed confusion matrix with percentages.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        matplotlib Figure
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Absolute values
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].set_title("Confusion Matrix (Counts)")
    axes[0].set_xticklabels(["No Disease", "Disease"])
    axes[0].set_yticklabels(["No Disease", "Disease"])

    # Normalized values
    sns.heatmap(cm_normalized, annot=True, fmt=".2%", cmap="Blues", ax=axes[1])
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    axes[1].set_title("Confusion Matrix (Percentages)")
    axes[1].set_xticklabels(["No Disease", "Disease"])
    axes[1].set_yticklabels(["No Disease", "Disease"])

    plt.tight_layout()
    return fig


def plot_roc_curve_detailed(y_true: np.ndarray, y_prob: np.ndarray) -> plt.Figure:
    """
    Plot detailed ROC curve with threshold annotations.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities

    Returns:
        matplotlib Figure
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    # Find optimal threshold (Youden's J statistic)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {auc:.3f})")
    ax.plot(
        [0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random classifier"
    )

    # Mark optimal threshold
    ax.scatter(fpr[optimal_idx], tpr[optimal_idx], color="red", s=100, zorder=5)
    ax.annotate(
        f"Optimal threshold: {optimal_threshold:.2f}",
        xy=(fpr[optimal_idx], tpr[optimal_idx]),
        xytext=(fpr[optimal_idx] + 0.1, tpr[optimal_idx] - 0.1),
        arrowprops=dict(arrowstyle="->", color="red"),
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_precision_recall_curve(y_true: np.ndarray, y_prob: np.ndarray) -> plt.Figure:
    """
    Plot precision-recall curve.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities

    Returns:
        matplotlib Figure
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(recall, precision, color="blue", lw=2, label=f"PR curve (AP = {ap:.3f})")

    # Baseline (random classifier)
    baseline = y_true.sum() / len(y_true)
    ax.axhline(
        y=baseline, color="gray", linestyle="--", label=f"Baseline ({baseline:.2f})"
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_calibration_curve(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> plt.Figure:
    """
    Plot calibration curve.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration

    Returns:
        matplotlib Figure
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(prob_pred, prob_true, "s-", color="blue", label="Model")
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve (Reliability Diagram)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_prediction_distribution(y_true: np.ndarray, y_prob: np.ndarray) -> plt.Figure:
    """
    Plot prediction probability distributions.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Separate predictions by true class
    probs_class_0 = y_prob[y_true == 0]
    probs_class_1 = y_prob[y_true == 1]

    ax.hist(
        probs_class_0,
        bins=30,
        alpha=0.5,
        label="No Disease",
        color="green",
        density=True,
    )
    ax.hist(
        probs_class_1, bins=30, alpha=0.5, label="Disease", color="red", density=True
    )

    ax.axvline(x=0.5, color="black", linestyle="--", label="Decision threshold")

    ax.set_xlabel("Predicted Probability of Disease")
    ax.set_ylabel("Density")
    ax.set_title("Prediction Distribution by True Class")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def find_optimal_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, metric: str = "f1"
) -> Tuple[float, float]:
    """
    Find optimal classification threshold.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        metric: Metric to optimize ('f1', 'youden', 'precision', 'recall')

    Returns:
        Tuple of (optimal threshold, metric value)
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_score = 0

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "youden":
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = sensitivity + specificity - 1
        elif metric == "precision":
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score


def generate_model_report(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Generate comprehensive model evaluation report.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model
        output_path: Path to save the report

    Returns:
        Dict containing the complete report
    """
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    )

    report = {
        "model_name": model_name,
        "test_samples": len(y_test),
        "metrics": compute_all_metrics(y_test.values, y_pred, y_prob),
        "classification_report": generate_classification_report(y_test.values, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    if y_prob is not None:
        optimal_threshold, optimal_f1 = find_optimal_threshold(
            y_test.values, y_prob, "f1"
        )
        report["optimal_threshold"] = optimal_threshold
        report["optimal_f1"] = optimal_f1

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            # Convert non-serializable items
            report_serializable = {
                k: v if not isinstance(v, np.ndarray) else v.tolist()
                for k, v in report.items()
            }
            json.dump(report_serializable, f, indent=2, default=str)

        logger.info(f"Report saved to {output_path}")

    return report


def compare_models(results: Dict[str, Tuple[Any, Dict[str, float]]]) -> pd.DataFrame:
    """
    Compare multiple models.

    Args:
        results: Dict mapping model names to (model, metrics) tuples

    Returns:
        DataFrame with comparison
    """
    comparison_data = []

    for model_name, (model, metrics) in results.items():
        row = {"model": model_name}
        for metric_name, value in metrics.items():
            if "test_" in metric_name:
                row[metric_name.replace("test_", "")] = value
        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)
    df = df.set_index("model")
    df = df.sort_values("roc_auc", ascending=False)

    return df


if __name__ == "__main__":
    # Example usage
    print("Model Evaluation Module")
    print("This module provides evaluation utilities for heart disease models.")
