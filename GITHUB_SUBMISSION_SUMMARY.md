# GitHub Submission Summary

## Project Title
Heart Disease Prediction MLOps Assignment

## What is included
- End-to-end machine learning pipeline for heart disease prediction
- Data preprocessing and model training workflow
- MLflow experiment tracking integration
- FastAPI-based prediction service
- Docker and Kubernetes deployment assets
- Prometheus and Grafana monitoring configuration
- Automated tests and CI/CD workflow

## Key implementation highlights
- Reproducible training pipeline with saved model and preprocessor artifacts
- Production-style API serving with health and prediction endpoints
- Containerized deployment setup for local and orchestrated environments
- Comprehensive documentation and report files for assignment submission

## Verification
- Unit tests: 60 passed
- API health endpoint: available locally
- MLflow UI: available locally

## Run quickly
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.models.train
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

## Submission notes
This repository is structured to satisfy the assignment requirements for MLOps implementation, deployment readiness, testing, monitoring, and documentation.
