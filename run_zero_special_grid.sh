#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_TEMPLATE="${ROOT_DIR}/config.yaml"
EVAL_SCRIPT="${ROOT_DIR}/evaluate.py"
LOG_DIR="${ROOT_DIR}/logs_zero_special"

mkdir -p "${LOG_DIR}"

declare -a SKIP_VALUES=(0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50)
declare -a TARGET_VALUES=(0.85 0.90 0.95 0.99)
declare -a GPU_SLOTS=(0 1 2)
declare -A ACTIVE_PIDS=()

create_config() {
    local skip_value="$1"
    local target_value="$2"
    local gpu_id="$3"
    local output_path="$4"

    python - "$skip_value" "$target_value" "$gpu_id" "$CONFIG_TEMPLATE" "$output_path" <<'PY'
import sys
from pathlib import Path

import yaml

skip = float(sys.argv[1])
target = float(sys.argv[2])
gpu = sys.argv[3]
template_path = Path(sys.argv[4])
output_path = Path(sys.argv[5])

with template_path.open("r", encoding="utf-8") as fh:
    config = yaml.safe_load(fh)

model_key = "llama-2-7b-attention"
models = config.get("models", {})
if model_key not in models:
    raise SystemExit(f"model '{model_key}' not found in {template_path}")

model_cfg = models[model_key]
model_cfg["model_name_or_path"] = "/mnt/public/models/Llama-2-7b-hf"
model_cfg["use_which_plan"] = "vanilla"
model_cfg["output_layer"] = 27
model_cfg["tp_starting_index"] = 0
model_cfg["tp_exiting_index"] = 0
model_cfg["batch_size"] = 16
model_cfg["mode"] = "test"
model_cfg["task_set"] = "sts"
model_cfg["prompt_method"] = "prompteol"

enhance_cfg = model_cfg.setdefault("attention_enhance", {})
enhance_cfg["enabled"] = True
enhance_cfg["enable_attention_override"] = True
enhance_cfg["head_order"] = "score"
enhance_cfg["override_mode"] = "zero_special"
enhance_cfg["zero_special_skip_threshold"] = float(f"{skip:.2f}")
enhance_cfg["zero_special_target_threshold"] = float(f"{target:.2f}")
enhance_cfg["score_file"] = "./head_score/llama-2-7b-80k.json"
enhance_cfg["top_k"] = 25
enhance_cfg["gamma"] = 20
enhance_cfg["target_phrase"] = 'means in one word:"'
enhance_cfg["analysis_samples"] = 1
enhance_cfg["analysis_dir"] = "attention_analysis"
enhance_cfg["average_last_token_attention"] = False
enhance_cfg["average_last_token_start_layer"] = 29

config.setdefault("gpu_config", {})["cuda_visible_devices"] = str(gpu)
config["default_config"] = model_key

with output_path.open("w", encoding="utf-8") as fh:
    yaml.safe_dump(config, fh, allow_unicode=True, sort_keys=False)
PY
}

run_job() {
    local skip_value="$1"
    local target_value="$2"
    local gpu_id="$3"

    local temp_dir
    temp_dir="$(mktemp -d)"
    local temp_config="${temp_dir}/config.yaml"

    create_config "$skip_value" "$target_value" "$gpu_id" "$temp_config"

    local skip_tag="${skip_value/./p}"
    local target_tag="${target_value/./p}"
    local log_file="${LOG_DIR}/skip_${skip_tag}_target_${target_tag}.log"

    (
        set +e
        CUDA_VISIBLE_DEVICES="${gpu_id}" python "${EVAL_SCRIPT}" \
            --config llama-2-7b-attention \
            --config_file "${temp_config}" \
            >"${log_file}" 2>&1
        status=$?
        rm -rf "${temp_dir}"
        exit "${status}"
    ) &

    ACTIVE_PIDS["${gpu_id}"]=$!
}

job_index=0
for skip in "${SKIP_VALUES[@]}"; do
    for target in "${TARGET_VALUES[@]}"; do
        slot_index=$((job_index % ${#GPU_SLOTS[@]}))
        gpu_id="${GPU_SLOTS[$slot_index]}"

        if [[ -n "${ACTIVE_PIDS[$gpu_id]:-}" ]]; then
            wait "${ACTIVE_PIDS[$gpu_id]}" || true
        fi

        echo "[INFO] Launching skip=${skip}, target=${target} on GPU ${gpu_id}"
        run_job "${skip}" "${target}" "${gpu_id}"

        job_index=$((job_index + 1))
    done
done

for pid in "${ACTIVE_PIDS[@]}"; do
    wait "${pid}" || true
done

echo "[INFO] All jobs completed."
