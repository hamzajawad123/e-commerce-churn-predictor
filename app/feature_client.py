"""Fetch online features from Feast for churn and LTV scoring."""
import os

from feast import FeatureStore

FEAST_REPO_PATH = os.getenv("FEAST_REPO_PATH", "./feature_repo")

_store = None


def get_store() -> FeatureStore:
    global _store
    if _store is None:
        _store = FeatureStore(repo_path=FEAST_REPO_PATH)
    return _store


def get_churn_features(customer_id: int) -> dict:
    store = get_store()
    result = store.get_online_features(
        features=store.get_feature_service("churn_ltv_feature_service"),
        entity_rows=[{"Customer_ID": customer_id}],
    ).to_dict()
    # Unwrap single-row Feast response {field: [value]} -> {field: value}
    return {k: v[0] for k, v in result.items() if k != "Customer_ID"}


def get_ltv_features(customer_id: int) -> dict:
    store = get_store()
    result = store.get_online_features(
        features=store.get_feature_service("ltv_feature_service"),
        entity_rows=[{"Customer_ID": customer_id}],
    ).to_dict()
    return {k: v[0] for k, v in result.items() if k != "Customer_ID"}


def customer_exists(features: dict) -> bool:
    # Missing entity keys come back as all-None feature values
    return any(v is not None for v in features.values())
