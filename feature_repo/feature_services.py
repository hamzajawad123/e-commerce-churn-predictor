from feast import FeatureService

from features import (
    customer_engagement_features,
    customer_profile_features,
    customer_purchase_features,
    customer_service_features,
)

# Churn service: includes Lifetime_Value as an input feature
churn_ltv_feature_service = FeatureService(
    name="churn_ltv_feature_service",
    features=[
        customer_profile_features,
        customer_engagement_features,
        customer_purchase_features,
        customer_service_features,
    ],
)

# LTV service: drops Lifetime_Value so the target is not used as a feature
ltv_feature_service = FeatureService(
    name="ltv_feature_service",
    features=[
        customer_profile_features,
        customer_engagement_features,
        customer_purchase_features,
        customer_service_features[
            ["Customer_Service_Calls", "Product_Reviews_Written", "Credit_Balance", "Dissatisfaction_Composite"]
        ],
    ],
)
