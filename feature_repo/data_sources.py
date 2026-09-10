from pathlib import Path

from feast import FileSource

# Relative path so feast apply works across machines (avoid absolute paths)
FEATURES_PATH = str(Path("..") / "data" / "processed" / "features.parquet")

customer_source = FileSource(
    name="customer_features_source",
    path=FEATURES_PATH,
    timestamp_field="event_timestamp",
)
