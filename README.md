# volleyball-role-random-forest

A compact, open-source volleyball analytics toolkit. It uses a Random Forest model to predict a player's position from match stats and exposes a lightweight API for quick decisions on the court.

## What it does
- Predicts playing position from per-match stats
- REST API: POST /predict and GET /health
- MLflow-backed training workflow for experiments and deployment

## Run locally
- Prereqs: Python 3.8+, Git
- Install dependencies
  - Backend: cd backend/app; python -m venv venv; source venv/bin/activate; pip install -r requirements.txt
  - Pipeline: cd ../../pipeline; python -m venv venv; source venv/bin/activate; pip install -r requirements.txt
- Run API
  - Set ARTIFACT_URI to your model path (MLflow) and optional MLFLOW_TRACKING_URI
  - uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
- Quick test
  - curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"sets_per_match":25,"receives_per_match":15,"blocks_per_match":3,"digs_per_match":8,"attacks_per_match":12}'

## Train the model (optional)
- python pipeline/flows/ml_orchestration.py
- The pipeline logs artifacts with MLflow.

## Project structure (high level)
- backend/app: FastAPI server
- pipeline: ML flow
- data: dataset placeholder

## Future improvements
- Tests, Dockerization, additional metrics, and improved explainability.

## Contributor
- Ahmad Nasiruddin Dzulkifli (nasirdzul@gmail.com)
