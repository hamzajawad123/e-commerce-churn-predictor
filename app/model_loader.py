"""Load @champion churn and LTV models from MLflow (CatBoost or sklearn)."""
import os
import threading

import mlflow
import mlflow.catboost
import mlflow.sklearn

MODEL_NAME_CHURN = os.getenv("MODEL_NAME_CHURN", "churn_model")
MODEL_NAME_LTV = os.getenv("MODEL_NAME_LTV", "ltv_model")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")

_lock = threading.Lock()
_churn_model = None
_ltv_model = None


def _ensure_tracking_uri():
    if not os.getenv("MLFLOW_TRACKING_URI"):
        raise RuntimeError(
            "MLFLOW_TRACKING_URI not set -- check .env is loaded before this module is used."
        )
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])


def _load_model(model_name: str):
    uri = f"models:/{model_name}@{MODEL_ALIAS}"
    try:
        return mlflow.catboost.load_model(uri)
    except Exception:
        return mlflow.sklearn.load_model(uri)


def get_churn_model():
    global _churn_model
    if _churn_model is None:
        with _lock:
            if _churn_model is None:
                _ensure_tracking_uri()
                _churn_model = _load_model(MODEL_NAME_CHURN)
    return _churn_model


def get_ltv_model():
    global _ltv_model
    if _ltv_model is None:
        with _lock:
            if _ltv_model is None:
                _ensure_tracking_uri()
                _ltv_model = _load_model(MODEL_NAME_LTV)
    return _ltv_model


def reload_models():
    # Clear cache so the next request picks up a newly promoted champion
    global _churn_model, _ltv_model
    with _lock:
        _churn_model = None
        _ltv_model = None
