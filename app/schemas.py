from pydantic import BaseModel


class ChurnResponse(BaseModel):
    customer_id: int
    churn_probability: float
    risk_tier: str


class LTVResponse(BaseModel):
    customer_id: int
    predicted_ltv: float
    ltv_tier: str
    ltv_confidence: str


class FullPredictionResponse(BaseModel):
    customer_id: int
    churn_probability: float
    risk_tier: str
    predicted_ltv: float
    ltv_tier: str
    ltv_confidence: str
    retention_strategy: str


class HighRiskCustomer(BaseModel):
    customer_id: int
    churn_probability: float
    risk_tier: str
    predicted_ltv: float
    ltv_tier: str
    retention_strategy: str
    predicted_at: str


class HealthResponse(BaseModel):
    status: str
    churn_model_loaded: bool
    ltv_model_loaded: bool


class ErrorResponse(BaseModel):
    detail: str
