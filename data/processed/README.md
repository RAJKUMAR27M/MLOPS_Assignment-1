# Processed Data Directory

This directory contains preprocessed and split data ready for model training.

## Files

- `train.csv` - Training dataset (~80% of data)
- `test.csv` - Test dataset (~20% of data)

## Preprocessing Applied

1. Missing value imputation (median for numeric, mode for categorical)
2. Feature scaling (StandardScaler for numeric features)
3. Stratified train/test split

## Usage

```python
import pandas as pd

train_df = pd.read_csv('data/processed/train.csv')
test_df = pd.read_csv('data/processed/test.csv')
```
