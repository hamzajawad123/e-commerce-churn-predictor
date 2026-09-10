from datetime import timedelta

from feast import FeatureView, Field
from feast.types import Float32, String

from data_sources import customer_source
from entities import customer

# Long TTL: one snapshot per customer, not a live event stream
SNAPSHOT_TTL = timedelta(days=3650)

# Churned is the label — keep it out of every feature view

customer_profile_features = FeatureView(
    name="customer_profile_features",
    entities=[customer],
    ttl=SNAPSHOT_TTL,
    schema=[
        Field(name="Age", dtype=Float32),
        Field(name="Gender", dtype=String),
        Field(name="Country", dtype=String),
        Field(name="City", dtype=String),
        Field(name="Membership_Years", dtype=Float32),
        Field(name="Signup_Quarter", dtype=String),
    ],
    online=True,
    source=customer_source,
    tags={"group": "demographics"},
)

customer_engagement_features = FeatureView(
    name="customer_engagement_features",
    entities=[customer],
    ttl=SNAPSHOT_TTL,
    schema=[
        Field(name="Login_Frequency", dtype=Float32),
        Field(name="Session_Duration_Avg", dtype=Float32),
        Field(name="Pages_Per_Session", dtype=Float32),
        Field(name="Cart_Abandonment_Rate", dtype=Float32),
        Field(name="Wishlist_Items", dtype=Float32),
        Field(name="Email_Open_Rate", dtype=Float32),
        Field(name="Mobile_App_Usage", dtype=Float32),
        Field(name="Social_Media_Engagement_Score", dtype=Float32),
        Field(name="Engagement_Composite", dtype=Float32),
    ],
    online=True,
    source=customer_source,
    tags={"group": "engagement"},
)

customer_purchase_features = FeatureView(
    name="customer_purchase_features",
    entities=[customer],
    ttl=SNAPSHOT_TTL,
    schema=[
        Field(name="Total_Purchases", dtype=Float32),
        Field(name="Average_Order_Value", dtype=Float32),
        Field(name="Days_Since_Last_Purchase", dtype=Float32),
        Field(name="Discount_Usage_Rate", dtype=Float32),
        Field(name="Returns_Rate", dtype=Float32),
        Field(name="Payment_Method_Diversity", dtype=Float32),
        Field(name="Recency", dtype=Float32),
        Field(name="Frequency", dtype=Float32),
        Field(name="Monetary_Proxy", dtype=Float32),
        Field(name="Tenure_Normalized_Activity", dtype=Float32),
    ],
    online=True,
    source=customer_source,
    tags={"group": "purchase_behavior"},
)

customer_service_features = FeatureView(
    name="customer_service_features",
    entities=[customer],
    ttl=SNAPSHOT_TTL,
    schema=[
        Field(name="Customer_Service_Calls", dtype=Float32),
        Field(name="Product_Reviews_Written", dtype=Float32),
        Field(name="Lifetime_Value", dtype=Float32),
        Field(name="Credit_Balance", dtype=Float32),
        Field(name="Dissatisfaction_Composite", dtype=Float32),
    ],
    online=True,
    source=customer_source,
    tags={"group": "service_and_financial"},
)
