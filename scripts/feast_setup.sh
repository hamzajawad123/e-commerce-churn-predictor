#!/usr/bin/env bash
# Apply Feast definitions and materialize the online store
set -e

cd "$(dirname "$0")/../feature_repo"

echo "Running feast apply..."
feast apply

echo "Running feast materialize-incremental..."
feast materialize-incremental "$(date -u +"%Y-%m-%dT%H:%M:%S")"

echo "Done. Registry + online store are in feature_repo/data/."
