import os
import mlflow
import pandas as pd
import json, tempfile
from pathlib import Path
from schemas.predict_input import PredictInput
from schemas.predict_output import PredictOutput

if os.environ.get("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

model_uri = os.environ.get("ARTIFACT_URI")
model = mlflow.pyfunc.load_model(model_uri)

def load_label_map_from_run(artifact_uri: str) -> dict[int, str]:
    # Download artifact logged at artifact_path="metadata", file name "label_map.json"
    label_dir = Path(mlflow.artifacts.download_artifacts(artifact_uri=f"{artifact_uri}/extra_files"))
    label_map_file = next(label_dir.glob("*.json"))
    print(label_map_file)
    with open(label_map_file, "r") as f:
        raw = json.load(f)
    # JSON keys come back as strings -> convert to int
    return {int(k): v for k, v in raw.items()}

def predict_position(input_data: PredictInput) -> PredictOutput:
    df = pd.DataFrame([input_data.model_dump()])
    pred_idx = int(model.predict(df)[0]) 
    label_map = load_label_map_from_run(model_uri)
    position = label_map.get(pred_idx, str(pred_idx))
    return PredictOutput(position=position)