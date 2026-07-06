# Heart Disease Prediction MLOps Project

A production-ready machine learning solution for predicting heart disease risk using modern MLOps best practices.

## 🎯 Project Overview

This project implements an end-to-end ML pipeline for binary classification of heart disease presence/absence based on the UCI Heart Disease dataset. It demonstrates:

- **Data Engineering**: Automated data acquisition, cleaning, and preprocessing
- **Model Development**: Multiple classification models with hyperparameter tuning
- **Experiment Tracking**: MLflow integration for reproducible experiments
- **CI/CD Pipeline**: GitHub Actions for automated testing and deployment
- **Containerization**: Docker-based model serving with FastAPI
- **Kubernetes Deployment**: Production-ready deployment manifests
- **Monitoring**: Prometheus + Grafana integration for observability

## 📁 Project Structure

```
MLops_Assignment/
├── .github/
│   └── workflows/
│       └── ci-cd.yml           # GitHub Actions CI/CD pipeline
├── data/
│   ├── raw/                    # Raw downloaded data
│   └── processed/              # Cleaned and processed data
├── notebooks/
│   ├── 01_data_acquisition_eda.ipynb
│   └── 02_model_training.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py         # Data acquisition script
│   │   └── preprocessing.py    # Data preprocessing pipeline
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py            # Model training script
│   │   └── evaluate.py         # Model evaluation utilities
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py             # FastAPI application
│   │   └── schemas.py          # Pydantic models
│   └── utils/
│       ├── __init__.py
│       └── logging_config.py   # Logging configuration
├── tests/
│   ├── __init__.py
│   ├── test_data_processing.py
│   ├── test_model.py
│   └── test_api.py
├── deployment/
│   ├── kubernetes/
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   └── ingress.yaml
│   └── helm/
│       └── heart-disease-api/
├── monitoring/
│   ├── prometheus/
│   │   └── prometheus.yml
│   └── grafana/
│       └── dashboard.json
├── models/                     # Saved model artifacts
├── mlruns/                     # MLflow experiment tracking
├── screenshots/                # Deployment screenshots
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── setup.py
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Kubernetes (Minikube/Docker Desktop) for local deployment

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/heart-disease-mlops.git
   cd heart-disease-mlops
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download data**
   ```bash
   python -m src.data.download
   ```

5. **Run EDA notebook**
   ```bash
   jupyter notebook notebooks/01_data_acquisition_eda.ipynb
   ```

### Training Models

```bash
# Train with MLflow tracking
python -m src.models.train
```

### Running the API Locally

```bash
# Start FastAPI server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker Deployment

```bash
# Build image
docker build -t heart-disease-api:latest .

# Run container
docker run -p 8000:8000 heart-disease-api:latest
```

### Kubernetes Deployment

```bash
# Deploy to Kubernetes
kubectl apply -f deployment/kubernetes/

# Check deployment status
kubectl get pods -l app=heart-disease-api
```

## 📊 API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Prediction Endpoint
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "thal": 1
  }'
```

### Response Format
```json
{
  "prediction": 1,
  "probability": 0.85,
  "risk_level": "High",
  "model_version": "1.0.0"
}
```

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v --cov=src

# Run specific test file
pytest tests/test_api.py -v
```

## 📈 MLflow Tracking

```bash
# Start MLflow UI
mlflow ui --host 0.0.0.0 --port 5000
```

Navigate to `http://localhost:5000` to view experiment tracking dashboard.

## 📉 Monitoring

### Prometheus Metrics
- Request count by endpoint
- Request latency histograms
- Prediction distribution
- Error rates

### Grafana Dashboard
Import `monitoring/grafana/dashboard.json` for pre-configured visualizations.

## 🔧 Configuration

Environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_PATH` | Path to saved model | `models/final_model.pkl` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `MLFLOW_TRACKING_URI` | MLflow server URI | `./mlruns` |

## 📝 License

MIT License - See LICENSE file for details.

## 👥 Contributors

- Your Name

## 📚 References

- [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
