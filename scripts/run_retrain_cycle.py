"""CI helper: generate synthetic batch, refresh Feast features, then retrain."""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from generate_synthetic_batch import generate_batch  # noqa: E402
from retrain_pipeline import run_retrain  # noqa: E402

FEAST_FEATURES_PATH = "data/processed/features.parquet"
COMBINED_PATH = "data/processed/features_with_synthetic.parquet"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-rows", type=int, default=5000)
    args = parser.parse_args()

    print("=== 1/3 Generate synthetic batch ===")
    combined = generate_batch(args.num_rows)
    Path(COMBINED_PATH).parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(COMBINED_PATH, index=False)
    # Refresh Feast source parquet (includes new synthetic customer IDs)
    feast_df = combined.drop(columns=["data_source"], errors="ignore")
    feast_df.to_parquet(FEAST_FEATURES_PATH, index=False)
    print(
        f"Wrote {COMBINED_PATH} and refreshed {FEAST_FEATURES_PATH} "
        f"({len(combined)} rows)"
    )

    print("=== 2/3 Retrain + compare + promote ===")
    if not os.getenv("MLFLOW_TRACKING_URI"):
        raise RuntimeError("MLFLOW_TRACKING_URI must be set for retrain cycle")
    summary = run_retrain(COMBINED_PATH)
    print("Retrain summary:", summary)

    print("=== 3/3 Done (caller should feast apply/materialize next) ===")


if __name__ == "__main__":
    main()
