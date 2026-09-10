"""Retrain models on real+synthetic data; promote @champion only if real-Q4 improves."""
from __future__ import annotations

import os
from typing import Any

import mlflow
import mlflow.catboost
import mlflow.sklearn
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from mlflow import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

CAT_COLS = ["Gender", "Country", "City", "Signup_Quarter"]
RANDOM_SEED = 42

MODEL_NAME_CHURN = os.getenv("MODEL_NAME_CHURN", "churn_model")
MODEL_NAME_LTV = os.getenv("MODEL_NAME_LTV", "ltv_model")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
FEATURES_PATH = os.getenv(
    "RETRAIN_FEATURES_PATH", "data/processed/features_with_synthetic.parquet"
)


def _ensure_mlflow():
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError("MLFLOW_TRACKING_URI is required for retraining.")
    mlflow.set_tracking_uri(uri)


def _feature_lists(df: pd.DataFrame, label: str) -> tuple[list[str], list[str]]:
    ignore = set(CAT_COLS + ["Customer_ID", "event_timestamp", "data_source", label])
    num_cols = [c for c in df.columns if c not in ignore]
    return CAT_COLS + num_cols, num_cols


def _make_preprocessor(num_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
            ("num", StandardScaler(), num_cols),
        ]
    )


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in CAT_COLS:
        out[c] = out[c].astype(str)
    return out


def _split_train_test(combined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Train on real non-Q4 + synthetic; hold out real Q4 for evaluation
    real = combined[combined["data_source"] == "real"].copy()
    synth = combined[combined["data_source"] == "synthetic"].copy()
    test = real[real["Signup_Quarter"].astype(str) == "Q4"].copy()
    train = pd.concat(
        [real[real["Signup_Quarter"].astype(str) != "Q4"], synth],
        ignore_index=True,
    )
    return train, test


def _train_churn_models(
    train_df: pd.DataFrame,
) -> tuple[str, Any, dict[str, float], dict[str, Any]]:
    features, num_cols = _feature_lists(train_df, "Churned")
    X = train_df[features]
    y = train_df["Churned"].astype(int)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    prep = _make_preprocessor(num_cols)
    results: dict[str, float] = {}
    models: dict[str, Any] = {}

    cb = CatBoostClassifier(
        verbose=0,
        random_state=RANDOM_SEED,
        auto_class_weights="Balanced",
        iterations=400,
        depth=6,
        learning_rate=0.05,
    )
    cb.fit(X_tr, y_tr, cat_features=CAT_COLS)
    models["CatBoost"] = cb
    results["CatBoost"] = float(roc_auc_score(y_va, cb.predict_proba(X_va)[:, 1]))

    candidates = {
        "RandomForest": Pipeline(
            [
                ("prep", prep),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=RANDOM_SEED,
                        class_weight="balanced",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "GradientBoosting": Pipeline(
            [
                ("prep", prep),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        random_state=RANDOM_SEED, class_weight="balanced"
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("prep", prep),
                (
                    "clf",
                    XGBClassifier(
                        random_state=RANDOM_SEED,
                        eval_metric="logloss",
                        n_estimators=300,
                        verbosity=0,
                        scale_pos_weight=float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1)),
                    ),
                ),
            ]
        ),
        "LightGBM": Pipeline(
            [
                ("prep", prep),
                (
                    "clf",
                    LGBMClassifier(
                        random_state=RANDOM_SEED,
                        class_weight="balanced",
                        n_estimators=300,
                        verbose=-1,
                    ),
                ),
            ]
        ),
    }
    for name, pipe in candidates.items():
        pipe.fit(X_tr, y_tr)
        models[name] = pipe
        results[name] = float(roc_auc_score(y_va, pipe.predict_proba(X_va)[:, 1]))

    winner = max(results, key=results.get)
    print("Churn val ROC-AUC:", {k: round(v, 4) for k, v in results.items()})
    print("Churn winner:", winner)
    return winner, models[winner], results, {"features": features}


def _train_ltv_models(
    train_df: pd.DataFrame,
) -> tuple[str, Any, dict[str, float], dict[str, Any]]:
    features, num_cols = _feature_lists(train_df, "Lifetime_Value")
    X = train_df[features]
    y_raw = train_df["Lifetime_Value"].astype(float)
    y_log = np.log1p(y_raw)
    X_tr, X_va, y_tr, y_va, y_tr_raw, y_va_raw = train_test_split(
        X, y_log, y_raw, test_size=0.2, random_state=RANDOM_SEED
    )
    prep = _make_preprocessor(num_cols)
    results: dict[str, float] = {}
    models: dict[str, Any] = {}

    def _rmse(pred_log, y_true_raw):
        pred = np.expm1(pred_log)
        return float(mean_squared_error(y_true_raw, pred) ** 0.5)

    cb = CatBoostRegressor(
        verbose=0,
        random_state=RANDOM_SEED,
        iterations=500,
        depth=6,
        learning_rate=0.05,
    )
    cb.fit(X_tr, y_tr, cat_features=CAT_COLS)
    models["CatBoost"] = cb
    results["CatBoost"] = _rmse(cb.predict(X_va), y_va_raw)

    candidates = {
        "RandomForest": Pipeline(
            [
                ("prep", prep),
                (
                    "reg",
                    RandomForestRegressor(
                        n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1
                    ),
                ),
            ]
        ),
        "GradientBoosting": Pipeline(
            [
                ("prep", prep),
                ("reg", HistGradientBoostingRegressor(random_state=RANDOM_SEED)),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("prep", prep),
                (
                    "reg",
                    XGBRegressor(
                        random_state=RANDOM_SEED, n_estimators=400, verbosity=0
                    ),
                ),
            ]
        ),
        "LightGBM": Pipeline(
            [
                ("prep", prep),
                (
                    "reg",
                    LGBMRegressor(
                        random_state=RANDOM_SEED, n_estimators=400, verbose=-1
                    ),
                ),
            ]
        ),
    }
    for name, pipe in candidates.items():
        pipe.fit(X_tr, y_tr)
        models[name] = pipe
        results[name] = _rmse(pipe.predict(X_va), y_va_raw)

    winner = min(results, key=results.get)
    print("LTV val RMSE:", {k: round(v, 2) for k, v in results.items()})
    print("LTV winner:", winner)
    return winner, models[winner], results, {"features": features}


def _predict_churn_proba(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def _predict_ltv_raw(model, X: pd.DataFrame) -> np.ndarray:
    return np.expm1(model.predict(X))


def _eval_churn_q4(model, test_df: pd.DataFrame, features: list[str]) -> dict[str, float]:
    X = test_df[features]
    y = test_df["Churned"].astype(int)
    proba = _predict_churn_proba(model, X)
    preds = (proba >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "f1": float(f1_score(y, preds)),
    }


def _eval_ltv_q4(model, test_df: pd.DataFrame, features: list[str]) -> dict[str, float]:
    X = test_df[features]
    y = test_df["Lifetime_Value"].astype(float)
    pred = _predict_ltv_raw(model, X)
    rmse = float(mean_squared_error(y, pred) ** 0.5)
    mae = float(mean_absolute_error(y, pred))
    return {"rmse": rmse, "mae": mae, "rmse_pct_of_mean": float(rmse / y.mean() * 100)}


def _load_registered_model(model_name: str):
    uri = f"models:/{model_name}@{MODEL_ALIAS}"
    try:
        return mlflow.catboost.load_model(uri)
    except Exception:
        return mlflow.sklearn.load_model(uri)


def _score_existing_champion_churn(test_df: pd.DataFrame, features: list[str]):
    client = MlflowClient()
    try:
        client.get_model_version_by_alias(MODEL_NAME_CHURN, MODEL_ALIAS)
    except Exception:
        return None
    model = _load_registered_model(MODEL_NAME_CHURN)
    return _eval_churn_q4(model, test_df, features)


def _score_existing_champion_ltv(test_df: pd.DataFrame, features: list[str]):
    client = MlflowClient()
    try:
        client.get_model_version_by_alias(MODEL_NAME_LTV, MODEL_ALIAS)
    except Exception:
        return None
    model = _load_registered_model(MODEL_NAME_LTV)
    return _eval_ltv_q4(model, test_df, features)


def _log_and_maybe_promote_churn(
    winner_name: str,
    model: Any,
    q4_metrics: dict[str, float],
    val_results: dict[str, float],
    champion_metrics: dict[str, float] | None,
) -> dict[str, Any]:
    promote = champion_metrics is None or q4_metrics["roc_auc"] > champion_metrics["roc_auc"]
    with mlflow.start_run(run_name=f"retrain_churn_{winner_name}") as run:
        mlflow.set_tags(
            {
                "pipeline": "continuous_retrain",
                "data_mix": "real_plus_synthetic",
                "eval_holdout": "real_Q4_only",
                "winner_model": winner_name,
            }
        )
        mlflow.log_params({f"val_auc_{k}": v for k, v in val_results.items()})
        mlflow.log_metrics({f"q4_{k}": v for k, v in q4_metrics.items()})
        if champion_metrics:
            mlflow.log_metrics({f"champion_q4_{k}": v for k, v in champion_metrics.items()})
        mlflow.log_param("promoted_to_champion", promote)

        if winner_name == "CatBoost":
            mlflow.catboost.log_model(model, "model", registered_model_name=MODEL_NAME_CHURN)
        else:
            mlflow.sklearn.log_model(model, "model", registered_model_name=MODEL_NAME_CHURN)

        run_id = run.info.run_id

    client = MlflowClient()
    latest = client.get_latest_versions(MODEL_NAME_CHURN)[0].version
    client.set_registered_model_alias(MODEL_NAME_CHURN, "candidate", latest)
    if promote:
        client.set_registered_model_alias(MODEL_NAME_CHURN, MODEL_ALIAS, latest)
        print(f"Promoted churn_model v{latest} -> @{MODEL_ALIAS}")
    else:
        print(
            f"Kept existing churn @{MODEL_ALIAS} "
            f"(candidate q4 AUC {q4_metrics['roc_auc']:.4f} "
            f"<= champion {champion_metrics['roc_auc']:.4f})"
        )
    return {"run_id": run_id, "version": latest, "promoted": promote, "q4": q4_metrics}


def _log_and_maybe_promote_ltv(
    winner_name: str,
    model: Any,
    q4_metrics: dict[str, float],
    val_results: dict[str, float],
    champion_metrics: dict[str, float] | None,
) -> dict[str, Any]:
    promote = champion_metrics is None or q4_metrics["rmse"] < champion_metrics["rmse"]
    with mlflow.start_run(run_name=f"retrain_ltv_{winner_name}") as run:
        mlflow.set_tags(
            {
                "pipeline": "continuous_retrain",
                "data_mix": "real_plus_synthetic",
                "eval_holdout": "real_Q4_only",
                "winner_model": winner_name,
            }
        )
        mlflow.log_params({f"val_rmse_{k}": v for k, v in val_results.items()})
        mlflow.log_metrics({f"q4_{k}": v for k, v in q4_metrics.items()})
        if champion_metrics:
            mlflow.log_metrics({f"champion_q4_{k}": v for k, v in champion_metrics.items()})
        mlflow.log_param("promoted_to_champion", promote)

        if winner_name == "CatBoost":
            mlflow.catboost.log_model(model, "model", registered_model_name=MODEL_NAME_LTV)
        else:
            mlflow.sklearn.log_model(model, "model", registered_model_name=MODEL_NAME_LTV)

        run_id = run.info.run_id

    client = MlflowClient()
    latest = client.get_latest_versions(MODEL_NAME_LTV)[0].version
    client.set_registered_model_alias(MODEL_NAME_LTV, "candidate", latest)
    if promote:
        client.set_registered_model_alias(MODEL_NAME_LTV, MODEL_ALIAS, latest)
        print(f"Promoted ltv_model v{latest} -> @{MODEL_ALIAS}")
    else:
        print(
            f"Kept existing ltv @{MODEL_ALIAS} "
            f"(candidate q4 RMSE {q4_metrics['rmse']:.2f} "
            f">= champion {champion_metrics['rmse']:.2f})"
        )
    return {"run_id": run_id, "version": latest, "promoted": promote, "q4": q4_metrics}


def run_retrain(features_path: str = FEATURES_PATH) -> dict[str, Any]:
    _ensure_mlflow()
    combined = pd.read_parquet(features_path)
    if "data_source" not in combined.columns:
        combined["data_source"] = "real"
    combined = _prepare_frame(combined)

    train_df, test_df = _split_train_test(combined)
    print(f"Train rows={len(train_df)}  Real-Q4 test rows={len(test_df)}")

    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_CHURN", "churn_classification"))
    churn_winner, churn_model, churn_val, churn_meta = _train_churn_models(train_df)
    churn_q4 = _eval_churn_q4(churn_model, test_df, churn_meta["features"])
    print("Churn real-Q4:", churn_q4)
    churn_champ = _score_existing_champion_churn(test_df, churn_meta["features"])
    churn_out = _log_and_maybe_promote_churn(
        churn_winner, churn_model, churn_q4, churn_val, churn_champ
    )

    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_LTV", "ltv_regression"))
    ltv_winner, ltv_model, ltv_val, ltv_meta = _train_ltv_models(train_df)
    ltv_q4 = _eval_ltv_q4(ltv_model, test_df, ltv_meta["features"])
    print("LTV real-Q4:", ltv_q4)
    ltv_champ = _score_existing_champion_ltv(test_df, ltv_meta["features"])
    ltv_out = _log_and_maybe_promote_ltv(
        ltv_winner, ltv_model, ltv_q4, ltv_val, ltv_champ
    )

    return {
        "churn": {"winner": churn_winner, **churn_out},
        "ltv": {"winner": ltv_winner, **ltv_out},
    }


if __name__ == "__main__":
    summary = run_retrain()
    print(summary)
