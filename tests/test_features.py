"""Feast online-store smoke tests for churn and LTV feature services."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("FEAST_REPO_PATH", str(Path(__file__).resolve().parent.parent / "feature_repo"))

import app.feature_client as feature_client  # noqa: E402


def test_churn_features_shape():
    features = feature_client.get_churn_features(0)
    assert "Lifetime_Value" in features, "Lifetime_Value must be present as an input for churn"
    assert "Churned" not in features, "Churned must never appear as a feature"


def test_ltv_features_exclude_target():
    features = feature_client.get_ltv_features(0)
    assert "Lifetime_Value" not in features, "Lifetime_Value must be excluded from LTV inputs"


def test_customer_exists_true_for_real_customer():
    features = feature_client.get_churn_features(0)
    assert feature_client.customer_exists(features) is True


def test_customer_exists_false_for_missing_customer():
    features = feature_client.get_churn_features(999999999)
    assert feature_client.customer_exists(features) is False
