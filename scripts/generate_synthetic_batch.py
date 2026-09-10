"""Sample synthetic rows, merge with real data, and write a combined parquet."""
import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
from sdv.utils import load_synthesizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from feature_engineering import clean_and_engineer  # noqa: E402
from fit_synthetic_generator import (  # noqa: E402
    GENERATOR_OUTPUT_PATH,
    fit_generator,
)

FEAST_FEATURES_PATH = "data/processed/features.parquet"
REAL_ONLY_PATH = "data/processed/features_real.parquet"


def ensure_real_snapshot() -> Path:
    # Snapshot real DVC data once so Feast overwrites don't pollute the next run
    real_path = Path(REAL_ONLY_PATH)
    feast_path = Path(FEAST_FEATURES_PATH)
    if not real_path.exists():
        if not feast_path.exists():
            raise FileNotFoundError(
                f"Missing {feast_path}. Run `dvc pull` before the retrain cycle."
            )
        real_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(feast_path, real_path)
        print(f"Snapshotted real features -> {real_path}")
    return real_path


def ensure_generator(generator_path: str = GENERATOR_OUTPUT_PATH) -> None:
    real_path = ensure_real_snapshot()
    if not Path(generator_path).exists():
        print(f"{generator_path} missing — fitting generator from real data...")
        fit_generator(features_path=str(real_path), output_path=generator_path)


def generate_batch(num_rows: int, generator_path: str = GENERATOR_OUTPUT_PATH) -> pd.DataFrame:
    ensure_generator(generator_path)

    real_df = pd.read_parquet(ensure_real_snapshot())
    real_df = real_df.copy()
    real_df["data_source"] = "real"

    synthesizer = load_synthesizer(generator_path)
    synthetic = synthesizer.sample(num_rows=num_rows)
    synthetic["Churned"] = synthetic["Churned"].astype(int)

    start_id = int(real_df["Customer_ID"].max()) + 1
    synthetic["Customer_ID"] = range(start_id, start_id + len(synthetic))
    synthetic["event_timestamp"] = pd.Timestamp.now(tz="UTC").normalize()
    synthetic["data_source"] = "synthetic"

    synthetic = clean_and_engineer(synthetic)
    synthetic = synthetic[real_df.columns]
    return pd.concat([real_df, synthetic], ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-rows", type=int, default=5000)
    parser.add_argument("--output", default="data/processed/features_with_synthetic.parquet")
    parser.add_argument(
        "--also-write-features",
        action="store_true",
        help="Also overwrite data/processed/features.parquet (for Feast materialize).",
    )
    args = parser.parse_args()

    combined = generate_batch(args.num_rows)
    real_count = int((combined["data_source"] == "real").sum())
    synth_count = int((combined["data_source"] == "synthetic").sum())
    print(
        f"Combined dataset: {real_count} real + {synth_count} synthetic = {len(combined)} rows"
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.output, index=False)
    print(f"Saved: {args.output}")

    if args.also_write_features:
        # Drop data_source before writing the Feast features file
        feast_df = combined.drop(columns=["data_source"], errors="ignore")
        feast_df.to_parquet(FEAST_FEATURES_PATH, index=False)
        print(f"Also wrote Feast source: {FEAST_FEATURES_PATH}")


if __name__ == "__main__":
    main()
