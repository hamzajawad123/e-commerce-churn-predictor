"""API tests with stubbed models; needs Feast online store materialized."""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier, CatBoostRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("FEAST_REPO_PATH", str(ROOT / "feature_repo"))
os.environ["PREDICTIONS_DB_PATH"] = "test_predictions_log.db"

CAT_COLS = ["Gender", "Country", "City", "Signup_Quarter"]


def _fake_churn_model(sample_features: dict):
    X = pd.DataFrame([sample_features] * 40)
    for c in X.columns:
        if c not in CAT_COLS:
            X[c] = X[c].astype(float) + np.random.randn(40)
    y = np.random.randint(0, 2, 40)
    return CatBoostClassifier(verbose=0, iterations=10).fit(X, y, cat_features=CAT_COLS)


def _fake_ltv_model(sample_features: dict):
    X = pd.DataFrame([sample_features] * 40)
    for c in X.columns:
        if c not in CAT_COLS:
            X[c] = X[c].astype(float) + np.random.randn(40)
    y = np.random.rand(40) * 8
    return CatBoostRegressor(verbose=0, iterations=10).fit(X, y, cat_features=CAT_COLS)


@pytest.fixture
def client():
    if os.path.exists("test_predictions_log.db"):
        os.remove("test_predictions_log.db")

    import app.feature_client as feature_client
    import app.model_loader as model_loader

    model_loader._churn_model = _fake_churn_model(feature_client.get_churn_features(0))
    model_loader._ltv_model = _fake_ltv_model(feature_client.get_ltv_features(0))

    from app import main
    from fastapi.testclient import TestClient

    with TestClient(main.app) as c:
        yield c

    if os.path.exists("test_predictions_log.db"):
        os.remove("test_predictions_log.db")


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["churn_model_loaded"] is True
    assert body["ltv_model_loaded"] is True


def test_predict_churn_known_customer(client):
    r = client.get("/predict_churn/0")
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_tier"] in {"Low", "Medium", "High"}


def test_predict_ltv_known_customer(client):
    r = client.get("/predict_ltv/0")
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_ltv"] >= 0
    assert body["ltv_tier"] in {"Low", "High"}
    assert body["ltv_confidence"] in {"low", "normal"}


def test_predict_full_logs_to_db(client):
    r = client.get("/predict_full/0")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {
        "customer_id", "churn_probability", "risk_tier", "predicted_ltv",
        "ltv_tier", "ltv_confidence", "retention_strategy",
    }

    r2 = client.get("/high_risk_customers")
    assert r2.status_code == 200
    assert isinstance(r2.json(), list)


def test_missing_customer_returns_404(client):
    r = client.get("/predict_churn/999999999")
    assert r.status_code == 404
