#!/usr/bin/env bash
# Train TRAIL. Config sections: dataset, data/model paths, LoRA, contrastive loss and training_config.
# Default config: src/configs/trail_training/douban_train.json
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

CONFIG="${CONFIG:-src/configs/trail_training/douban_train.json}"
RUN_NAME="${RUN_NAME:-douban_train}"
args=(--config "$CONFIG" --run-dir "runs/$RUN_NAME")
if [[ -n "${DEVICE:-}" ]]; then
    args+=(--device "$DEVICE")
fi
exec python -m src.run_fine_tuning "${args[@]}"
