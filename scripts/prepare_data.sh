#!/usr/bin/env bash
# Normalize existing CSVs. Config sections: dataset and files (paths and column mappings).
# Default config: src/configs/data_preparation.json
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

CONFIG="${CONFIG:-src/configs/data_preparation.json}"
RUN_NAME="${RUN_NAME:-data_preparation}"
args=(--config "$CONFIG" --run-dir "runs/$RUN_NAME")
exec python -m src.datasets.prepare "${args[@]}"
