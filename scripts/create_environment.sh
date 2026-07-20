#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_NAME="${MLDR_KD_ENV_NAME:-mldr-kd}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu118}"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda is not available on PATH." >&2
  echo "Initialize Conda first, then rerun this script." >&2
  exit 1
fi

if conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -Fxq -- "${ENV_NAME}"; then
  echo "Error: Conda environment '${ENV_NAME}' already exists." >&2
  echo "To keep it, follow the manual install steps in README.md." >&2
  echo "To recreate it, run: conda env remove -n ${ENV_NAME} -y" >&2
  exit 2
fi

echo "[1/5] Creating the lightweight Conda environment '${ENV_NAME}'..."
conda env create --name "${ENV_NAME}" --file "${PROJECT_ROOT}/environment.yml"

echo "[2/5] Installing PyTorch 2.1.1 and TorchVision 0.16.1 (CUDA 11.8 wheels)..."
conda run --name "${ENV_NAME}" python -m pip install \
  "torch==2.1.1+cu118" \
  "torchvision==0.16.1+cu118" \
  --index-url "${PYTORCH_INDEX_URL}"

echo "[3/5] Installing the minimal runtime dependencies..."
conda run --name "${ENV_NAME}" python -m pip install \
  --requirement "${PROJECT_ROOT}/requirements.txt"

echo "[4/5] Installing MLDR-KD in editable mode..."
conda run --name "${ENV_NAME}" python -m pip install \
  --editable "${PROJECT_ROOT}" --no-deps

echo "[5/5] Checking the installation..."
conda run --name "${ENV_NAME}" python "${PROJECT_ROOT}/scripts/check_environment.py"

echo
echo "Environment '${ENV_NAME}' is ready."
echo "Activate it with: conda activate ${ENV_NAME}"
