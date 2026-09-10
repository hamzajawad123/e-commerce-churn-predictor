"""Fit and save the SDV GaussianCopula synthesizer on real features."""
from pathlib import Path

import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer

RAW_FEATURES_PATH = "data/processed/features.parquet"
GENERATOR_OUTPUT_PATH = "data/synthetic_generator.pkl"
METADATA_OUTPUT_PATH = "data/synthetic_generator_metadata.json"

EXCLUDE_FROM_FIT = [
    "Customer_ID",
    "event_timestamp",
    "data_source",
    "Recency",
    "Frequency",
    "Monetary_Proxy",
    "Tenure_Normalized_Activity",
    "Engagement_Composite",
    "Dissatisfaction_Composite",
]

SDTYPE_OVERRIDES = {
    "City": "categorical",
    "Email_Open_Rate": "numerical",
    "Payment_Method_Diversity": "numerical",
    "Churned": "categorical",
}


def fit_generator(
    features_path: str = RAW_FEATURES_PATH,
    output_path: str = GENERATOR_OUTPUT_PATH,
) -> str:
    df = pd.read_parquet(features_path)
    train_df = df[[c for c in df.columns if c not in EXCLUDE_FROM_FIT]]

    metadata = Metadata.detect_from_dataframe(data=train_df, table_name="customers")
    for column, sdtype in SDTYPE_OVERRIDES.items():
        if column in train_df.columns:
            metadata.update_column(column, table_name="customers", sdtype=sdtype)

    print(f"Fitting GaussianCopulaSynthesizer on {train_df.shape[0]} real rows...")
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(train_df)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    synthesizer.save(output_path)
    metadata.save_to_json(METADATA_OUTPUT_PATH, mode="overwrite")
    print(f"Saved generator to {output_path}")
    return output_path


def main():
    fit_generator()


if __name__ == "__main__":
    main()
