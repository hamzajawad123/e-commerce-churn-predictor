"""Promote MLflow @candidate models to @champion after reviewing metrics."""
import os
from getpass import getpass

import mlflow
from mlflow import MlflowClient

DAGSHUB_REPO_OWNER = "hamzajawad123"
DAGSHUB_REPO_NAME = "e-commerce-churn-predictor"

os.environ["MLFLOW_TRACKING_URI"] = f"https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_REPO_OWNER
os.environ["MLFLOW_TRACKING_PASSWORD"] = getpass("DagsHub access token: ")
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

client = MlflowClient()

for model_name in ["churn_model", "ltv_model"]:
    candidate = client.get_model_version_by_alias(model_name, "candidate")
    client.set_registered_model_alias(model_name, "champion", candidate.version)
    print(f"{model_name}: v{candidate.version} (run_id={candidate.run_id}) promoted candidate -> champion")
