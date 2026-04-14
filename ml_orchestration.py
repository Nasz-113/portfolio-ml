#!/usr/bin/env python
# coding: utf-8

import os
import tempfile

import joblib
import mlflow
import pandas as pd
from prefect import flow, task
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

mlflow.set_experiment("mlops-learn1")
mlflow.autolog()
if mlflow.active_run() is not None:
    mlflow.end_run()


# Load and clean data (no scaling / encoding — avoids leakage before split)
@task
def load_clean_data():
    player_stats = pd.read_csv("data/playerStats.csv")
    player_stats.drop(columns=["Player Name", "Team"], inplace=True)
    y = player_stats["Position"]
    x = player_stats.drop(columns=["Position"])
    x = x[
        [
            "Age",
            "Sets Per Match",
            "Receives Per Match",
            "Serves Per Match",
            "Blocks Per Match",
            "Digs Per Match",
            "Attacks Per Match",
        ]
    ]
    x = x.drop(columns=["Age", "Serves Per Match"])
    return x, y


# Split before any fit on the full dataset
@task
def split_data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    return x_train, x_test, y_train, y_test


# Fit encoders on training only; transform train and test
@task
def fit_preprocess(x_train, x_test, y_train, y_test):
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled, y_train_enc, y_test_enc, scaler, label_encoder


def _log_preprocessors_mlflow(scaler, label_encoder):
    with tempfile.TemporaryDirectory() as tmp:
        scaler_path = os.path.join(tmp, "standard_scaler.joblib")
        encoder_path = os.path.join(tmp, "label_encoder.joblib")
        joblib.dump(scaler, scaler_path)
        joblib.dump(label_encoder, encoder_path)
        mlflow.log_artifacts(tmp, artifact_path="preprocessing")


@task
def train_random_forest(
    x_train_scaled,
    y_train_enc,
    x_test_scaled,
    y_test_enc,
    scaler,
    label_encoder,
):
    with mlflow.start_run(run_name="RandomForest_best_model"):
        _log_preprocessors_mlflow(scaler, label_encoder)

        best_rf = RandomForestClassifier(
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
        )
        best_rf.fit(x_train_scaled, y_train_enc)
        preds = best_rf.predict(x_test_scaled)
        acc = accuracy_score(y_test_enc, preds)
        f1 = f1_score(y_test_enc, preds, average="weighted")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)
        mlflow.sklearn.log_model(
            best_rf,
            "random_forest_model",
            registered_model_name="random_forest_model_dev",
            input_example=x_train_scaled,
        )
        return best_rf


@flow
def main():
    x, y = load_clean_data()
    x_train, x_test, y_train, y_test = split_data(x, y)
    x_train_s, x_test_s, y_train_e, y_test_e, scaler, le = fit_preprocess(
        x_train, x_test, y_train, y_test
    )
    best_rf = train_random_forest(
        x_train_s, y_train_e, x_test_s, y_test_e, scaler, le
    )
    preds = best_rf.predict(x_test_s)
    acc = accuracy_score(y_test_e, preds)
    f1 = f1_score(y_test_e, preds, average="weighted")
    print(f"Accuracy: {acc}")
    print(f"F1 Score: {f1}")


if __name__ == "__main__":
    main()
