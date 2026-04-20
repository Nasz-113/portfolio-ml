#!/usr/bin/env python
# coding: utf-8
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from prefect import flow, task
import mlflow
from sklearn.pipeline import Pipeline
import json, tempfile

if os.environ.get("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("vb-position-prediction")
mlflow.autolog()
if mlflow.active_run() is not None:
    mlflow.end_run()

# Load and clean data
@task
def load_data():
    playerStats = pd.read_csv("/home/nas/portfolio/portfolio-ml/data/playerStats.csv")
    playerStats.drop(columns=["Player Name", "Team"],inplace=True)
    X=playerStats.drop(columns=["Position"])
    y=playerStats["Position"]
    X = X[["Age","Sets Per Match", "Receives Per Match", "Serves Per Match", "Blocks Per Match", "Digs Per Match", "Attacks Per Match"]]
    X = X.drop(columns=["Age", "Serves Per Match"])
    X = X.rename(columns={
        "Sets Per Match": "sets_per_match",
        "Receives Per Match": "receives_per_match",
        "Blocks Per Match": "blocks_per_match",
        "Digs Per Match": "digs_per_match",
        "Attacks Per Match": "attacks_per_match",
        "Serves Per Match": "serves_per_match",
    })
    return X, y
    
# Encode and scale data
@task
def label_encoding(y_train, y_test):
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    return y_train_encoded, y_test_encoded, le

# Split for training & testing
@task
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

@task
def pipeline():
    train_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model_rf", RandomForestClassifier(
                bootstrap=True,
                ccp_alpha=0.0,
                class_weight=None,
                criterion="gini",
                max_depth=30,
                max_features="log2",
                max_leaf_nodes=None,
                max_samples=None,
                min_impurity_decrease=0.0,
                min_samples_leaf=4,
                min_samples_split=5,
                min_weight_fraction_leaf=0.0,
                monotonic_cst=None,
                n_estimators=443,
                n_jobs=None,
                oob_score=False,
                random_state=42,
                verbose=0,
                warm_start=False,
            ))
        ]
    )
    return train_pipeline

@task
def train_random_forest(train_pipeline, X_train, y_train, X_test, y_test, le):
    with mlflow.start_run(run_name="RandomForest_best_model"):

        train_pipeline.fit(X_train, y_train)
        preds = train_pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)

        # log label map for API decoding
        label_map = {int(i): c for i, c in enumerate(le.classes_)}
        with tempfile.NamedTemporaryFile("w", prefix="label_map", suffix=".json", delete=False) as f:
            json.dump(label_map, f)
            p = f.name
        mlflow.log_artifact(p, artifact_path="metadata")
        mlflow.sklearn.log_model(
            train_pipeline,
            "random_forest_model",
            registered_model_name="random_forest_model_dev",
            input_example=X_train,
            extra_files=[p]
        )


@flow
def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    y_train_encoded, y_test_encoded, le= label_encoding(y_train,y_test)
    train_pipeline = pipeline()
    train_random_forest(train_pipeline,X_train, y_train_encoded, X_test, y_test_encoded, le)

if __name__ == "__main__" :
    main()