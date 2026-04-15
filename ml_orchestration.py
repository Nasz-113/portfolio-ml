#!/usr/bin/env python
# coding: utf-8

import mlflow
mlflow.set_experiment("mlops-learn1")
mlflow.autolog()
if mlflow.active_run() is not None:
    mlflow.end_run()

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from prefect import flow, task

# Load and clean data
@task
def load_clean_data():
    playerStats = pd.read_csv("data/playerStats.csv")
    playerStats.drop(columns=["Player Name", "Team"],inplace=True)
    X=playerStats.drop(columns=["Position"])
    y=playerStats["Position"]
    X = X[["Age","Sets Per Match", "Receives Per Match", "Serves Per Match", "Blocks Per Match", "Digs Per Match", "Attacks Per Match"]]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X = X.drop(columns=["Age", "Serves Per Match"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y_encoded

# Split for training & testing
@task
def split_data(X_scaled, y_encoded):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

@task
def train_random_forest(X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name="RandomForest_best_model"):
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
        best_rf.fit(X_train, y_train)
        preds = best_rf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)
        mlflow.sklearn.log_model(best_rf, "random_forest_model", registered_model_name="random_forest_model_dev", input_example=X_train)
        return best_rf
@flow
def main():
    X_scaled, y_encoded = load_clean_data()
    X_train, X_test, y_train, y_test = split_data(X_scaled, y_encoded)
    best_rf = train_random_forest(X_train, y_train, X_test, y_test)
    preds = best_rf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    print(f"Accuracy: {acc}")  
    print(f"F1 Score: {f1}")
    
if __name__ == "__main__":
    main()