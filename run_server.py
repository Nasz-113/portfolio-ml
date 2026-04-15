import os
import mlflow.pyfunc
from fastapi import FastAPI
import pandas as pd
from sklearn.preprocessing import StandardScaler

if os.environ.get("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

app = FastAPI()
model = mlflow.pyfunc.load_model("models:/random-forest-classifier-model-dev/latest")

@app.get("/health")
def health_check():
    return {"message": "OK"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    scaler = StandardScaler()
    df = scaler.fit_transform(df)
    return {"prediction": model.predict(df).tolist()}