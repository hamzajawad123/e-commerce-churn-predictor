"""Rule-based retention actions from churn risk and LTV tiers."""
import os

CHURN_RISK_HIGH_THRESHOLD = float(os.getenv("CHURN_RISK_HIGH_THRESHOLD", "0.7"))
CHURN_RISK_MEDIUM_THRESHOLD = float(os.getenv("CHURN_RISK_MEDIUM_THRESHOLD", "0.4"))

# High LTV cutoff = dataset mean Lifetime_Value from EDA
LTV_HIGH_THRESHOLD = float(os.getenv("LTV_HIGH_THRESHOLD", "1440.63"))

RETENTION_TABLE = {
    ("High", "High"): "Personal agent callback + high-value discount",
    ("High", "Low"): "Automated small coupon",
    ("Medium", "High"): "Proactive email + moderate cashback offer",
    ("Medium", "Low"): "Automated email nudge",
    ("Low", "High"): "No action",
    ("Low", "Low"): "No action",
}


def risk_tier(churn_probability: float) -> str:
    if churn_probability >= CHURN_RISK_HIGH_THRESHOLD:
        return "High"
    if churn_probability >= CHURN_RISK_MEDIUM_THRESHOLD:
        return "Medium"
    return "Low"


def ltv_tier(predicted_ltv: float) -> str:
    return "High" if predicted_ltv >= LTV_HIGH_THRESHOLD else "Low"


def ltv_confidence(signup_quarter: str) -> str:
    # Q4 LTV predictions are less reliable due to purchase-shift in that cohort
    return "low" if signup_quarter == "Q4" else "normal"


def decide(risk: str, ltv: str) -> str:
    return RETENTION_TABLE[(risk, ltv)]


def full_decision(churn_probability: float, predicted_ltv: float, signup_quarter: str) -> dict:
    risk = risk_tier(churn_probability)
    ltv = ltv_tier(predicted_ltv)
    return {
        "risk_tier": risk,
        "ltv_tier": ltv,
        "ltv_confidence": ltv_confidence(signup_quarter),
        "retention_strategy": decide(risk, ltv),
    }
