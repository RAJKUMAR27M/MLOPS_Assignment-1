"""
Data Preprocessing Pipeline for Heart Disease Dataset

This module provides a complete preprocessing pipeline including:
- Missing value handling
- Feature encoding
- Feature scaling
- Train/test splitting
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import joblib
import logging

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Feature definitions
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET_COLUMN = "target"

# Feature descriptions for documentation
FEATURE_DESCRIPTIONS = {
    "age": "Age in years",
    "sex": "Sex (1 = male; 0 = female)",
    "cp": "Chest pain type (0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic)",
    "trestbps": "Resting blood pressure (mm Hg on admission)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)",
    "restecg": "Resting ECG results (0: normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy)",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise induced angina (1 = yes; 0 = no)",
    "oldpeak": "ST depression induced by exercise relative to rest",
    "slope": "Slope of peak exercise ST segment (0: upsloping, 1: flat, 2: downsloping)",
    "ca": "Number of major vessels (0-3) colored by fluoroscopy",
    "thal": "Thalassemia (0: normal, 1: fixed defect, 2: reversible defect)",
    "target": "Diagnosis of heart disease (0 = no disease, 1 = disease)",
}


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def load_raw_data(data_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load raw data from CSV file.

    Args:
        data_path: Path to the CSV file. If None, uses default location.

    Returns:
        pd.DataFrame: Raw dataset
    """
    if data_path is None:
        project_root = get_project_root()
        data_path = project_root / "data" / "raw" / "heart_cleveland.csv"

    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")

    return df


def analyze_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze data quality including missing values and outliers.

    Args:
        df: Input DataFrame

    Returns:
        Dict containing data quality metrics
    """
    quality_report = {
        "total_samples": len(df),
        "total_features": len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percentages": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        "duplicates": df.duplicated().sum(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }

    # Numeric feature statistics
    numeric_stats = {}
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            numeric_stats[col] = {
                "mean": df[col].mean(),
                "std": df[col].std(),
                "min": df[col].min(),
                "max": df[col].max(),
                "median": df[col].median(),
            }
    quality_report["numeric_stats"] = numeric_stats

    # Target distribution
    if TARGET_COLUMN in df.columns:
        quality_report["target_distribution"] = (
            df[TARGET_COLUMN].value_counts().to_dict()
        )
        quality_report["target_balance"] = (
            df[TARGET_COLUMN].value_counts(normalize=True).to_dict()
        )

    return quality_report


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.

    Strategy:
    - Numeric features: Impute with median
    - Categorical features: Impute with mode

    Args:
        df: Input DataFrame with missing values

    Returns:
        pd.DataFrame: DataFrame with imputed values
    """
    df = df.copy()

    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() == 0:
        logger.info("No missing values found")
        return df

    logger.info(f"Missing values found:\n{missing[missing > 0]}")

    # Impute numeric features with median
    for col in NUMERIC_FEATURES:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"Imputed {col} with median: {median_val}")

    # Impute categorical features with mode
    for col in CATEGORICAL_FEATURES:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            logger.info(f"Imputed {col} with mode: {mode_val}")

    return df


def remove_outliers(
    df: pd.DataFrame, columns: list = None, threshold: float = 3.0
) -> pd.DataFrame:
    """
    Remove outliers using Z-score method.

    Args:
        df: Input DataFrame
        columns: List of columns to check for outliers. If None, uses NUMERIC_FEATURES.
        threshold: Z-score threshold for outlier detection

    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    df = df.copy()

    if columns is None:
        columns = [col for col in NUMERIC_FEATURES if col in df.columns]

    initial_count = len(df)

    for col in columns:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        df = df[z_scores < threshold]

    removed = initial_count - len(df)
    if removed > 0:
        logger.info(f"Removed {removed} outliers ({removed/initial_count*100:.1f}%)")

    return df


class HeartDiseasePreprocessor:
    """
    Complete preprocessing pipeline for Heart Disease dataset.

    This class provides methods for:
    - Data cleaning and imputation
    - Feature scaling
    - Train/test splitting
    - Pipeline persistence
    """

    def __init__(self, scale_features: bool = True, remove_outliers: bool = False):
        """
        Initialize the preprocessor.

        Args:
            scale_features: Whether to scale numeric features
            remove_outliers: Whether to remove outliers
        """
        self.scale_features = scale_features
        self.remove_outliers_flag = remove_outliers
        self.scaler = StandardScaler() if scale_features else None
        self.imputer_numeric = SimpleImputer(strategy="median")
        self.imputer_categorical = SimpleImputer(strategy="most_frequent")
        self.is_fitted = False
        self.feature_names = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "HeartDiseasePreprocessor":
        """
        Fit the preprocessor on training data.

        Args:
            X: Training features
            y: Training labels (optional)

        Returns:
            self
        """
        self.feature_names = list(X.columns)

        # Fit imputers
        numeric_cols = [col for col in NUMERIC_FEATURES if col in X.columns]
        categorical_cols = [col for col in CATEGORICAL_FEATURES if col in X.columns]

        if numeric_cols:
            self.imputer_numeric.fit(X[numeric_cols])

        if categorical_cols:
            self.imputer_categorical.fit(X[categorical_cols])

        # Fit scaler on imputed data
        if self.scale_features and numeric_cols:
            X_imputed = X.copy()
            X_imputed[numeric_cols] = self.imputer_numeric.transform(X[numeric_cols])
            self.scaler.fit(X_imputed[numeric_cols])

        self.is_fitted = True
        logger.info("Preprocessor fitted successfully")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted preprocessor.

        Args:
            X: Features to transform

        Returns:
            pd.DataFrame: Transformed features
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        X = X.copy()

        numeric_cols = [col for col in NUMERIC_FEATURES if col in X.columns]
        categorical_cols = [col for col in CATEGORICAL_FEATURES if col in X.columns]

        # Apply imputation
        if numeric_cols:
            X[numeric_cols] = self.imputer_numeric.transform(X[numeric_cols])

        if categorical_cols:
            X[categorical_cols] = self.imputer_categorical.transform(
                X[categorical_cols]
            )

        # Apply scaling
        if self.scale_features and numeric_cols:
            X[numeric_cols] = self.scaler.transform(X[numeric_cols])

        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Args:
            X: Features to fit and transform
            y: Labels (optional)

        Returns:
            pd.DataFrame: Transformed features
        """
        return self.fit(X, y).transform(X)

    def save(self, path: Path) -> None:
        """
        Save the preprocessor to disk.

        Args:
            path: Path to save the preprocessor
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {
                "scaler": self.scaler,
                "imputer_numeric": self.imputer_numeric,
                "imputer_categorical": self.imputer_categorical,
                "scale_features": self.scale_features,
                "remove_outliers_flag": self.remove_outliers_flag,
                "is_fitted": self.is_fitted,
                "feature_names": self.feature_names,
            },
            path,
        )

        logger.info(f"Preprocessor saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "HeartDiseasePreprocessor":
        """
        Load a preprocessor from disk.

        Args:
            path: Path to load the preprocessor from

        Returns:
            HeartDiseasePreprocessor: Loaded preprocessor
        """
        data = joblib.load(path)

        preprocessor = cls(
            scale_features=data["scale_features"],
            remove_outliers=data["remove_outliers_flag"],
        )
        preprocessor.scaler = data["scaler"]
        preprocessor.imputer_numeric = data["imputer_numeric"]
        preprocessor.imputer_categorical = data["imputer_categorical"]
        preprocessor.is_fitted = data["is_fitted"]
        preprocessor.feature_names = data["feature_names"]

        logger.info(f"Preprocessor loaded from {path}")
        return preprocessor


def prepare_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42, scale: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, HeartDiseasePreprocessor]:
    """
    Prepare data for model training.

    Args:
        df: Raw DataFrame
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        scale: Whether to scale features

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, preprocessor)
    """
    # Separate features and target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    logger.info(f"Train target distribution:\n{y_train.value_counts()}")

    # Create and fit preprocessor
    preprocessor = HeartDiseasePreprocessor(scale_features=scale)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor


def save_processed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Optional[Path] = None,
) -> None:
    """
    Save processed data to disk.

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        output_dir: Output directory path
    """
    if output_dir is None:
        project_root = get_project_root()
        output_dir = project_root / "data" / "processed"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Combine X and y for saving
    train_data = X_train.copy()
    train_data[TARGET_COLUMN] = y_train.values

    test_data = X_test.copy()
    test_data[TARGET_COLUMN] = y_test.values

    train_data.to_csv(output_dir / "train.csv", index=False)
    test_data.to_csv(output_dir / "test.csv", index=False)

    logger.info(f"Processed data saved to {output_dir}")


def main():
    """Main function for data preprocessing."""
    logger.info("=" * 50)
    logger.info("Heart Disease Data Preprocessing")
    logger.info("=" * 50)

    # Load raw data
    df = load_raw_data()

    # Analyze data quality
    quality_report = analyze_data_quality(df)
    logger.info("\nData Quality Report:")
    logger.info(f"Total samples: {quality_report['total_samples']}")
    logger.info(f"Missing values: {sum(quality_report['missing_values'].values())}")
    logger.info(f"Duplicates: {quality_report['duplicates']}")

    # Handle missing values
    df = handle_missing_values(df)

    # Prepare data
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)

    # Save processed data
    save_processed_data(X_train, X_test, y_train, y_test)

    # Save preprocessor
    project_root = get_project_root()
    preprocessor.save(project_root / "models" / "preprocessor.joblib")

    logger.info("\n" + "=" * 50)
    logger.info("Preprocessing Complete!")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
