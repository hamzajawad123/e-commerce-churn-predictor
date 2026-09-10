import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import retention_policy as rp  # noqa: E402


def test_risk_tier_thresholds():
    assert rp.risk_tier(0.9) == "High"
    assert rp.risk_tier(0.5) == "Medium"
    assert rp.risk_tier(0.1) == "Low"
    assert rp.risk_tier(rp.CHURN_RISK_HIGH_THRESHOLD) == "High"  # inclusive boundary


def test_ltv_tier_uses_real_mean_threshold():
    assert rp.ltv_tier(2000) == "High"
    assert rp.ltv_tier(500) == "Low"


def test_q4_gets_low_confidence_flag():
    assert rp.ltv_confidence("Q4") == "low"
    for q in ["Q1", "Q2", "Q3"]:
        assert rp.ltv_confidence(q) == "normal"


def test_full_decision_matches_rule_table():
    result = rp.full_decision(0.85, 2000, "Q4")
    assert result["risk_tier"] == "High"
    assert result["ltv_tier"] == "High"
    assert result["ltv_confidence"] == "low"
    assert result["retention_strategy"] == "Personal agent callback + high-value discount"

    result_low_risk = rp.full_decision(0.1, 2000, "Q1")
    assert result_low_risk["retention_strategy"] == "No action"
