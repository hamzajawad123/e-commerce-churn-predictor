"""Shared cleaning and feature engineering used by real and synthetic pipelines."""
import pandas as pd

COLUMNS_WITH_MISSING = [
    "Age", "Session_Duration_Avg", "Pages_Per_Session", "Wishlist_Items",
    "Days_Since_Last_Purchase", "Discount_Usage_Rate", "Returns_Rate", "Email_Open_Rate",
    "Customer_Service_Calls", "Product_Reviews_Written", "Social_Media_Engagement_Score",
    "Mobile_App_Usage", "Payment_Method_Diversity", "Credit_Balance",
]

ENGAGEMENT_CLUSTER = [
    "Session_Duration_Avg", "Pages_Per_Session", "Mobile_App_Usage", "Login_Frequency",
    "Wishlist_Items", "Email_Open_Rate", "Social_Media_Engagement_Score",
]

DISSATISFACTION_COLS = ["Customer_Service_Calls", "Cart_Abandonment_Rate"]


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in COLUMNS_WITH_MISSING:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # RFM-style and composite columns — same formulas as the Colab notebook
    df = df.copy()

    df["Recency"] = df["Days_Since_Last_Purchase"]
    df["Frequency"] = df["Total_Purchases"]
    df["Monetary_Proxy"] = df["Average_Order_Value"] * df["Total_Purchases"]
    df["Tenure_Normalized_Activity"] = df["Total_Purchases"] / (df["Membership_Years"] + 1)

    z = (df[ENGAGEMENT_CLUSTER] - df[ENGAGEMENT_CLUSTER].mean()) / df[ENGAGEMENT_CLUSTER].std()
    df["Engagement_Composite"] = z.mean(axis=1)

    z2 = (df[DISSATISFACTION_COLS] - df[DISSATISFACTION_COLS].mean()) / df[DISSATISFACTION_COLS].std()
    df["Dissatisfaction_Composite"] = z2.mean(axis=1)

    return df


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Total_Purchases"] = df["Total_Purchases"].clip(lower=0)  # no negative purchases
    df = impute_missing(df)
    df = engineer_features(df)
    return df
