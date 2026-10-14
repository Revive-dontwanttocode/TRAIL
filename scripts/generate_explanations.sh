#!/usr/bin/env bash
# Generate explanations. Config sections: dataset, data, model, inference_params and vllm.
# Default config: src/configs/explanation_generation/douban_data_generation.json
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

CONFIG="${CONFIG:-src/configs/explanation_generation/douban_data_generation.json}"
RUN_NAME="${RUN_NAME:-douban_explanations}"
args=(--config "$CONFIG" --run-dir "runs/$RUN_NAME")
if [[ -n "${DEVICE:-}" ]]; then
    args+=(--device "$DEVICE")
fi
exec python -m src.run_data_generation "${args[@]}"
