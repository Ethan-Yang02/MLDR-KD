#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_experiment.sh CONFIG.yaml [train.py arguments...]" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="$1"
shift
if [[ "${CONFIG_PATH}" != /* ]]; then
  CONFIG_PATH="${PROJECT_ROOT}/${CONFIG_PATH}"
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Configuration not found: ${CONFIG_PATH}" >&2
  exit 2
fi

EXTRA_ARGS=()
[[ -n "${DATA_DIR:-}" ]] && EXTRA_ARGS+=(--data-dir "${DATA_DIR}")
[[ -n "${OUTPUT_DIR:-}" ]] && EXTRA_ARGS+=(--output-dir "${OUTPUT_DIR}")
[[ -n "${TEACHER_CKPT:-}" ]] && EXTRA_ARGS+=(--teacher-checkpoint "${TEACHER_CKPT}")
[[ -n "${VIM_MODEL_DIR:-}" ]] && EXTRA_ARGS+=(--vim-model-dir "${VIM_MODEL_DIR}")

DEVICE="${DEVICE:-cuda}"
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}"
if [[ ! "${GPU_IDS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "GPU_IDS must be a comma-separated list of GPU indices, for example 0 or 0,2." >&2
  exit 2
fi

IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"
declare -A SEEN_GPUS=()
for gpu_id in "${GPU_LIST[@]}"; do
  if [[ -n "${SEEN_GPUS[${gpu_id}]:-}" ]]; then
    echo "GPU_IDS contains a duplicate GPU index: ${gpu_id}" >&2
    exit 2
  fi
  SEEN_GPUS["${gpu_id}"]=1
done
NUM_PROCESSES="${#GPU_LIST[@]}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

if [[ -z "${MASTER_PORT:-}" ]]; then
  MASTER_PORT="$(python - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("", 0))
    print(sock.getsockname()[1])
PY
)"
elif [[ ! "${MASTER_PORT}" =~ ^[0-9]+$ ]] \
    || (( 10#${MASTER_PORT} < 1024 || 10#${MASTER_PORT} > 65535 )); then
  echo "MASTER_PORT must be an integer between 1024 and 65535." >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}; processes: ${NUM_PROCESSES}; master port: ${MASTER_PORT}"
python -m torch.distributed.launch \
  --nproc_per_node="${NUM_PROCESSES}" \
  --master_port="${MASTER_PORT}" \
  --use-env \
  train.py \
  --config "${CONFIG_PATH}" \
  --device "${DEVICE}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
