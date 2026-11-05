#!/usr/bin/env bash
cd "$(dirname "$0")/.." && uv run mlflow ui --backend-store-uri "sqlite:///$PWD/mlruns/mlflow.db" --port 5000
