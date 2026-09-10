import os
import sqlite3
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

load_dotenv()

# Put src/ on path so retention_policy imports cleanly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from . import feature_client, model_loader  # noqa: E402
import retention_policy  # noqa: E402
from .schemas import (  # noqa: E402
    ChurnResponse,
    FullPredictionResponse,
    HealthResponse,
    HighRiskCustomer,
    LTVResponse,
)

DB_PATH = os.getenv("PREDICTIONS_DB_PATH", "predictions_log.db")
CAT_COLS = ["Gender", "Country", "City", "Signup_Quarter"]


def _model_feature_names(model) -> list[str]:
    # CatBoost: feature_names_; sklearn: feature_names_in_
    if hasattr(model, "feature_names_"):
        return list(model.feature_names_)
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    prep = getattr(model, "named_steps", {}).get("prep")
    if prep is not None and hasattr(prep, "feature_names_in_"):
        return list(prep.feature_names_in_)
    raise RuntimeError("Could not determine model feature names for prediction.")


def _prepare_model_frame(features: dict, model):
    # Match trained column order and cast categoricals to strings
    import pandas as pd

    X = pd.DataFrame([features])
    names = _model_feature_names(model)
    for col in names:
        if col not in X.columns:
            X[col] = None
    X = X[names]
    for col in CAT_COLS:
        if col in X.columns:
            X[col] = X[col].map(lambda v: "nan" if v is None else str(v))
    return X


def _init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions_log (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER NOT NULL,
            churn_probability REAL NOT NULL,
            risk_tier TEXT NOT NULL,
            predicted_ltv REAL NOT NULL,
            ltv_tier TEXT NOT NULL,
            ltv_confidence TEXT NOT NULL,
            retention_strategy TEXT NOT NULL,
            predicted_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def _log_prediction(row: dict):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO predictions_log
        (customer_id, churn_probability, risk_tier, predicted_ltv,
         ltv_tier, ltv_confidence, retention_strategy, predicted_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row["customer_id"], row["churn_probability"], row["risk_tier"],
            row["predicted_ltv"], row["ltv_tier"], row["ltv_confidence"],
            row["retention_strategy"], row["predicted_at"],
        ),
    )
    conn.commit()
    conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_db()
    yield


app = FastAPI(title="E-Commerce Churn & LTV Predictor", lifespan=lifespan)


def _churn_probability_for(customer_id: int) -> tuple[float, dict]:
    features = feature_client.get_churn_features(customer_id)
    if not feature_client.customer_exists(features):
        raise HTTPException(
            status_code=404,
            detail=f"Customer {customer_id} not found in the online feature store.",
        )
    model = model_loader.get_churn_model()
    X = _prepare_model_frame(features, model)
    proba = float(model.predict_proba(X)[:, 1][0])
    return proba, features


def _predicted_ltv_for(customer_id: int) -> tuple[float, dict]:
    features = feature_client.get_ltv_features(customer_id)
    if not feature_client.customer_exists(features):
        raise HTTPException(
            status_code=404,
            detail=f"Customer {customer_id} not found in the online feature store.",
        )
    model = model_loader.get_ltv_model()
    import numpy as np

    X = _prepare_model_frame(features, model)
    pred_log = model.predict(X)[0]
    pred_raw = float(np.expm1(pred_log))
    return pred_raw, features


@app.get("/health", response_model=HealthResponse)
def health():
    churn_loaded, ltv_loaded = True, True
    try:
        model_loader.get_churn_model()
    except Exception:
        churn_loaded = False
    try:
        model_loader.get_ltv_model()
    except Exception:
        ltv_loaded = False
    return HealthResponse(status="ok", churn_model_loaded=churn_loaded, ltv_model_loaded=ltv_loaded)


@app.get("/predict_churn/{customer_id}", response_model=ChurnResponse)
def predict_churn(customer_id: int):
    proba, _ = _churn_probability_for(customer_id)
    return ChurnResponse(
        customer_id=customer_id,
        churn_probability=round(proba, 4),
        risk_tier=retention_policy.risk_tier(proba),
    )


@app.get("/predict_ltv/{customer_id}", response_model=LTVResponse)
def predict_ltv(customer_id: int):
    pred_ltv, features = _predicted_ltv_for(customer_id)
    signup_quarter = features.get("Signup_Quarter", "")
    return LTVResponse(
        customer_id=customer_id,
        predicted_ltv=round(pred_ltv, 2),
        ltv_tier=retention_policy.ltv_tier(pred_ltv),
        ltv_confidence=retention_policy.ltv_confidence(signup_quarter),
    )


@app.get("/predict_full/{customer_id}", response_model=FullPredictionResponse)
def predict_full(customer_id: int):
    proba, churn_features = _churn_probability_for(customer_id)
    pred_ltv, ltv_features = _predicted_ltv_for(customer_id)
    signup_quarter = churn_features.get("Signup_Quarter", "")

    decision = retention_policy.full_decision(proba, pred_ltv, signup_quarter)
    now = datetime.now(timezone.utc).isoformat()

    row = {
        "customer_id": customer_id,
        "churn_probability": round(proba, 4),
        "risk_tier": decision["risk_tier"],
        "predicted_ltv": round(pred_ltv, 2),
        "ltv_tier": decision["ltv_tier"],
        "ltv_confidence": decision["ltv_confidence"],
        "retention_strategy": decision["retention_strategy"],
        "predicted_at": now,
    }
    _log_prediction(row)

    return FullPredictionResponse(
        customer_id=customer_id,
        churn_probability=row["churn_probability"],
        risk_tier=row["risk_tier"],
        predicted_ltv=row["predicted_ltv"],
        ltv_tier=row["ltv_tier"],
        ltv_confidence=row["ltv_confidence"],
        retention_strategy=row["retention_strategy"],
    )


@app.get("/high_risk_customers", response_model=list[HighRiskCustomer])
def high_risk_customers(limit: int = 50):
    # Latest high-risk row per customer (deduped)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT p.customer_id, p.churn_probability, p.risk_tier, p.predicted_ltv,
               p.ltv_tier, p.retention_strategy, p.predicted_at
        FROM predictions_log p
        INNER JOIN (
            SELECT customer_id, MAX(predicted_at) AS max_at
            FROM predictions_log
            WHERE risk_tier = 'High'
            GROUP BY customer_id
        ) latest
          ON p.customer_id = latest.customer_id
         AND p.predicted_at = latest.max_at
        WHERE p.risk_tier = 'High'
        ORDER BY p.predicted_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()
    return [HighRiskCustomer(**dict(r)) for r in rows]
