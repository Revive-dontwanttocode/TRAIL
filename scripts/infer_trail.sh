#!/usr/bin/env bash
# Infer and evaluate. Config sections: dataset, model/adapter/data paths, inference_params and evaluation.
# Default config: src/configs/batch_inference/douban_batch.json
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

CONFIG="${CONFIG:-src/configs/batch_inference/douban_batch.json}"
RUN_NAME="${RUN_NAME:-douban_inference}"
args=(--config "$CONFIG" --run-dir "runs/$RUN_NAME")
if [[ -n "${DEVICE:-}" ]]; then
    args+=(--device "$DEVICE")
fi
exec python -m src.run_inference_batch "${args[@]}"
