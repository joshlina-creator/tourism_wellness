"""
Model Training Script (with MLflow Experiment Tracking)
---------------------------------------------------------
Builds a preprocessing + classification pipeline to predict `ProdTaken`
(whether a customer will purchase the Wellness Tourism Package).

Trains several candidate models, logs parameters/metrics/artifacts for each
run to MLflow, selects the best model by test-set F1 score, registers it in
the MLflow Model Registry, and saves the final pipeline to disk for
deployment (also optionally pushed to the Hugging Face Hub as a model repo).
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from huggingface_hub import HfApi, create_repo

TRAIN_PATH = "tourism_project/data/train.csv"
TEST_PATH = "tourism_project/data/test.csv"
TARGET = "ProdTaken"
MODEL_DIR = "tourism_project/model_building"
DEPLOY_DIR = "tourism_project/deployment"
BEST_MODEL_LOCAL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
DEPLOY_MODEL_PATH = os.path.join(DEPLOY_DIR, "best_model.joblib")

NUMERIC_FEATURES = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome",
]
CATEGORICAL_FEATURES = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]

HF_USERNAME = os.environ.get("HF_USERNAME", "your-hf-username")
MODEL_REPO_ID = f"{HF_USERNAME}/tourism-wellness-package-model"


def build_preprocessor():
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )


def get_candidate_models():
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=6, class_weight="balanced", random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=10, class_weight="balanced",
            random_state=42, n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            eval_metric="logloss", random_state=42,
        ),
    }


def evaluate(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def train_and_track():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train, y_train = train_df.drop(columns=TARGET), train_df[TARGET]
    X_test, y_test = test_df.drop(columns=TARGET), test_df[TARGET]

    mlflow.set_tracking_uri("sqlite:///tourism_project/model_building/mlflow.db")
    mlflow.set_experiment("Tourism_Wellness_Package_Prediction")

    results = {}
    best_model_name, best_pipeline, best_f1 = None, None, -1.0

    for name, model in get_candidate_models().items():
        with mlflow.start_run(run_name=name):
            pipeline = Pipeline(
                steps=[("preprocessor", build_preprocessor()), ("model", model)]
            )
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            metrics = evaluate(y_test, y_pred, y_proba)

            mlflow.log_param("model_type", name)
            for param_name, param_value in model.get_params().items():
                try:
                    mlflow.log_param(param_name, param_value)
                except Exception:
                    pass
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            mlflow.sklearn.log_model(pipeline, artifact_path="model", serialization_format="pickle")

            results[name] = metrics
            print(f"[{name}] " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_model_name = name
                best_pipeline = pipeline

    print(f"\nBest model: {best_model_name} (F1={best_f1:.4f})")

    with mlflow.start_run(run_name=f"best_model_{best_model_name}"):
        mlflow.log_param("model_type", best_model_name)
        for metric_name, metric_value in results[best_model_name].items():
            mlflow.log_metric(metric_name, metric_value)
        mlflow.sklearn.log_model(
            best_pipeline, artifact_path="model", serialization_format="pickle",
            registered_model_name="tourism_wellness_package_model",
        )

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DEPLOY_DIR, exist_ok=True)
    joblib.dump(best_pipeline, BEST_MODEL_LOCAL_PATH)
    joblib.dump(best_pipeline, DEPLOY_MODEL_PATH)
    print(f"Saved best pipeline to {BEST_MODEL_LOCAL_PATH} and {DEPLOY_MODEL_PATH}")

    with open(os.path.join(MODEL_DIR, "model_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        api = HfApi(token=hf_token)
        create_repo(repo_id=MODEL_REPO_ID, token=hf_token, exist_ok=True, private=False)
        api.upload_file(
            path_or_fileobj=BEST_MODEL_LOCAL_PATH, path_in_repo="best_model.joblib",
            repo_id=MODEL_REPO_ID,
        )
        print(f"Model pushed to: https://huggingface.co/{MODEL_REPO_ID}")
    else:
        print("HF_TOKEN not found. Skipping model upload to the Hugging Face Hub.")

    return results, best_model_name


if __name__ == "__main__":
    train_and_track()
