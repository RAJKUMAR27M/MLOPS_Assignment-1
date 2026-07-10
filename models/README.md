# Models Directory

This directory contains trained model artifacts.

## Files

- `final_model.joblib` - Best performing trained model
- `preprocessor.joblib` - Fitted data preprocessor

## Loading Models

```python
import joblib

# Load model
model_data = joblib.load('models/final_model.joblib')
model = model_data['model']
model_name = model_data['model_name']
version = model_data['version']

# Load preprocessor
from src.data.preprocessing import HeartDiseasePreprocessor
preprocessor = HeartDiseasePreprocessor.load('models/preprocessor.joblib')

# Make prediction
X_processed = preprocessor.transform(X_new)
prediction = model.predict(X_processed)
probability = model.predict_proba(X_processed)[:, 1]
```

## Model Information

- **Model Type**: Random Forest Classifier (or best performing model)
- **Version**: 1.0.0
- **Metrics**: See REPORT.md for performance details
