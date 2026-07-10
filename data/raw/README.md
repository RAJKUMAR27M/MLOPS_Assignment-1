# Raw Data Directory

This directory contains the raw, unprocessed data downloaded from the UCI Machine Learning Repository.

## Files

- `heart_cleveland.csv` - Cleveland Heart Disease dataset (primary dataset)
- `heart_combined.csv` - Combined dataset from all centers (optional)

## Download Instructions

Run the following command to download the data:

```bash
python -m src.data.download
```

## Dataset Information

- **Source**: UCI Machine Learning Repository
- **URL**: https://archive.ics.uci.edu/ml/datasets/heart+Disease
- **Samples**: ~303 (Cleveland)
- **Features**: 14 attributes
