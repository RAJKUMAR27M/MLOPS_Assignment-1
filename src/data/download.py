"""
Heart Disease UCI Dataset Download Script

This script downloads the Heart Disease dataset from UCI Machine Learning Repository
and saves it to the data/raw directory.
"""

import requests
import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Dataset URLs
DATASET_URLS = {
    "cleveland": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    "hungarian": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data",
    "switzerland": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data",
    "va": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data",
}

# Alternative URL (Kaggle-style combined dataset)
COMBINED_DATASET_URL = (
    "https://raw.githubusercontent.com/plotly/datasets/master/heart_disease_uci.csv"
)

# Column names for the UCI dataset
COLUMN_NAMES = [
    "age",  # Age in years
    "sex",  # Sex (1 = male; 0 = female)
    "cp",  # Chest pain type (0-3)
    "trestbps",  # Resting blood pressure (mm Hg)
    "chol",  # Serum cholesterol (mg/dl)
    "fbs",  # Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
    "restecg",  # Resting ECG results (0-2)
    "thalach",  # Maximum heart rate achieved
    "exang",  # Exercise induced angina (1 = yes; 0 = no)
    "oldpeak",  # ST depression induced by exercise
    "slope",  # Slope of peak exercise ST segment (0-2)
    "ca",  # Number of major vessels colored by fluoroscopy (0-3)
    "thal",  # Thalassemia (0 = normal; 1 = fixed defect; 2 = reversible defect)
    "target",  # Diagnosis of heart disease (0 = no disease, 1-4 = disease present)
]


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def download_file(url: str, save_path: Path) -> bool:
    """
    Download a file from URL and save it locally.

    Args:
        url: URL to download from
        save_path: Local path to save the file

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Downloading from {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(response.text)
        logger.info(f"Saved to {save_path}")
        return True
    except requests.RequestException as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def download_cleveland_dataset() -> pd.DataFrame:
    """
    Download the Cleveland Heart Disease dataset (most commonly used).

    Returns:
        pd.DataFrame: The downloaded dataset
    """
    project_root = get_project_root()
    raw_data_dir = project_root / "data" / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = raw_data_dir / "heart_cleveland.csv"

    # Try to download from UCI repository
    url = DATASET_URLS["cleveland"]

    try:
        logger.info(f"Downloading Cleveland dataset from {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Parse the data (UCI format uses '?' for missing values)
        from io import StringIO

        df = pd.read_csv(StringIO(response.text), names=COLUMN_NAMES, na_values="?")

        # Convert target to binary (0 = no disease, 1 = disease)
        df["target"] = (df["target"] > 0).astype(int)

        # Save to CSV
        df.to_csv(csv_path, index=False)
        logger.info(f"Dataset saved to {csv_path}")
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Target distribution:\n{df['target'].value_counts()}")

        return df

    except Exception as e:
        logger.warning(f"Failed to download from UCI: {e}")
        logger.info("Trying alternative source...")

        # Try alternative source
        try:
            response = requests.get(COMBINED_DATASET_URL, timeout=30)
            response.raise_for_status()

            from io import StringIO

            df = pd.read_csv(StringIO(response.text))

            # Standardize column names
            df.columns = df.columns.str.lower().str.replace(" ", "_")

            # Ensure target is binary
            if "num" in df.columns:
                df["target"] = (df["num"] > 0).astype(int)
                df = df.drop("num", axis=1)

            df.to_csv(csv_path, index=False)
            logger.info(f"Dataset saved to {csv_path}")

            return df

        except Exception as e2:
            logger.error(f"Failed to download from alternative source: {e2}")
            raise


def download_combined_dataset() -> pd.DataFrame:
    """
    Download combined Heart Disease dataset from all centers.

    Returns:
        pd.DataFrame: Combined dataset from all centers
    """
    project_root = get_project_root()
    raw_data_dir = project_root / "data" / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    all_dfs = []

    for name, url in DATASET_URLS.items():
        try:
            logger.info(f"Downloading {name} dataset...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            from io import StringIO

            df = pd.read_csv(StringIO(response.text), names=COLUMN_NAMES, na_values="?")
            df["source"] = name
            all_dfs.append(df)

            # Save individual dataset
            df.to_csv(raw_data_dir / f"heart_{name}.csv", index=False)

        except Exception as e:
            logger.warning(f"Failed to download {name}: {e}")

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df["target"] = (combined_df["target"] > 0).astype(int)

        # Save combined dataset
        combined_path = raw_data_dir / "heart_combined.csv"
        combined_df.to_csv(combined_path, index=False)
        logger.info(f"Combined dataset saved to {combined_path}")
        logger.info(f"Combined dataset shape: {combined_df.shape}")

        return combined_df
    else:
        raise RuntimeError("Failed to download any dataset")


def create_sample_dataset() -> pd.DataFrame:
    """
    Create a sample dataset for testing when download fails.

    Returns:
        pd.DataFrame: Sample dataset
    """
    import numpy as np

    np.random.seed(42)
    n_samples = 303  # Same as Cleveland dataset

    data = {
        "age": np.random.randint(29, 77, n_samples),
        "sex": np.random.randint(0, 2, n_samples),
        "cp": np.random.randint(0, 4, n_samples),
        "trestbps": np.random.randint(94, 200, n_samples),
        "chol": np.random.randint(126, 564, n_samples),
        "fbs": np.random.randint(0, 2, n_samples),
        "restecg": np.random.randint(0, 3, n_samples),
        "thalach": np.random.randint(71, 202, n_samples),
        "exang": np.random.randint(0, 2, n_samples),
        "oldpeak": np.random.uniform(0, 6.2, n_samples).round(1),
        "slope": np.random.randint(0, 3, n_samples),
        "ca": np.random.randint(0, 4, n_samples),
        "thal": np.random.randint(0, 3, n_samples),
        "target": np.random.randint(0, 2, n_samples),
    }

    df = pd.DataFrame(data)

    project_root = get_project_root()
    raw_data_dir = project_root / "data" / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = raw_data_dir / "heart_cleveland.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Sample dataset created at {csv_path}")

    return df


def main():
    """Main function to download the dataset."""
    logger.info("=" * 50)
    logger.info("Heart Disease UCI Dataset Download")
    logger.info("=" * 50)

    try:
        # Download Cleveland dataset (most commonly used)
        df = download_cleveland_dataset()

        logger.info("\n" + "=" * 50)
        logger.info("Download Complete!")
        logger.info("=" * 50)
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Features: {list(df.columns)}")
        logger.info(f"\nData types:\n{df.dtypes}")
        logger.info(f"\nBasic statistics:\n{df.describe()}")

    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.info("Creating sample dataset for development...")
        df = create_sample_dataset()
        logger.info("Sample dataset created successfully!")


if __name__ == "__main__":
    main()
