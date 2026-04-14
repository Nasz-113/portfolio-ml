import os

import joblib
import mlflow
import mlflow.artifacts
import pandas as pd
from fastapi import FastAPI

if os.environ.get("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

# Feature order must match training after column drops (see ml_orchestration.load_clean_data)
FEATURE_COLUMNS = [
    "Sets Per Match",
    "Receives Per Match",
    "Blocks Per Match",
    "Digs Per Match",
    "Attacks Per Match",
]

def _load_preprocessors(run_id: str):
    base = mlflow.artifacts.download_artifacts("models:/random-forest-classifier-model-dev/latest")
    scaler = joblib.load(os.path.join(base, "standard_scaler.joblib"))
    label_encoder = joblib.load(os.path.join(base, "label_encoder.joblib"))
    return scaler, label_encoder

_scaler, _label_encoder = _load_preprocessors(_run_id)

app = FastAPI()
model = mlflow.pyfunc.load_model("models:/random-forest-classifier-model-dev/latest")

@app.get("/health")
def health_check():
    return {"message": "OK"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        return {"error": f"Missing keys: {missing}", "expected": FEATURE_COLUMNS}
    df = df[FEATURE_COLUMNS]
    x = _scaler.transform(df)
    y_idx = model.predict(x)
    labels = _label_encoder.inverse_transform(y_idx.astype(int))
    return {"prediction": labels.tolist()}
