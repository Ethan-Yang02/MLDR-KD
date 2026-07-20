#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WHEEL_DIR="${VIM_WHEEL_DIR:-${PROJECT_ROOT}/wheels}"
VIM_MODEL_DIR="${VIM_MODEL_DIR:-${PROJECT_ROOT}/third_party/Vim/vim}"

SETUPTOOLS_VERSION="80.9.0"
TRANSFORMERS_VERSION="4.43.3"
CAUSAL_WHEEL="causal_conv1d-1.1.1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
MAMBA_WHEEL="mamba_ssm-1.2.0.post1+vimcu118torch2.1cxx11abifalse-cp310-cp310-linux_x86_64.whl"
CAUSAL_SHA256="f572458de281ed15fbd6f9262d44d66591416c7a903c4da699ed4d70b3cf5a83"
MAMBA_SHA256="2906d8cdd3f8f84c91f876f9ddd9337ffbc9161a9e1e66897313438d125b392a"

fail() {
  echo "Error: $*" >&2
  exit 2
}

verify_wheel() {
  local path="$1"
  local expected="$2"
  local actual
  [[ -f "${path}" ]] || fail "Missing wheel: ${path}. Download artifacts as shown in README.md."
  read -r actual _ < <(sha256sum "${path}")
  if [[ "${actual}" != "${expected}" ]]; then
    fail "SHA-256 mismatch for ${path}. Expected ${expected}, received ${actual}."
  fi
}

echo "[1/4] Checking the runtime and vendored Vim source..."
[[ -f "${VIM_MODEL_DIR}/models_mamba.py" ]] || fail "Missing ${VIM_MODEL_DIR}/models_mamba.py."
[[ -f "${PROJECT_ROOT}/third_party/Vim/mamba-1p1p1/mamba_ssm/modules/mamba_simple.py" ]] \
  || fail "The vendored Vim checkout is incomplete."

python - <<PY
import platform
import sys

try:
    import torch
except ImportError as error:
    raise SystemExit("PyTorch is missing. Run scripts/create_environment.sh first.") from error

actual = {
    "Python": f"{sys.version_info.major}.{sys.version_info.minor}",
    "PyTorch": torch.__version__.split("+", 1)[0],
    "PyTorch CUDA": torch.version.cuda,
    "CXX11 ABI": str(bool(torch._C._GLIBCXX_USE_CXX11_ABI)),
    "platform": f"{platform.system()} {platform.machine()}",
}
expected = {
    "Python": "3.10",
    "PyTorch": "2.1.1",
    "PyTorch CUDA": "11.8",
    "CXX11 ABI": "False",
    "platform": "Linux x86_64",
}
problems = [
    f"{name}: {actual[name]} (expected {value})"
    for name, value in expected.items()
    if actual[name].lower() != value.lower()
]
if problems:
    raise SystemExit(
        "The supplied wheels do not match this environment:\n  - "
        + "\n  - ".join(problems)
        + "\nRun scripts/create_environment.sh."
    )
for name, value in actual.items():
    print(f"{name}: {value}")
PY

echo "[2/4] Installing Python-side ViM dependencies..."
python -m pip install \
  "setuptools==${SETUPTOOLS_VERSION}" \
  "einops==0.8.0" \
  "transformers==${TRANSFORMERS_VERSION}"

python -W ignore::UserWarning - <<PY
import pkg_resources  # noqa: F401
from torch.utils import cpp_extension  # noqa: F401
print("PyTorch C++ extension helper: OK")
PY

echo "[3/4] Verifying and installing local CUDA wheels..."
CAUSAL_PATH="${WHEEL_DIR%/}/${CAUSAL_WHEEL}"
MAMBA_PATH="${WHEEL_DIR%/}/${MAMBA_WHEEL}"
verify_wheel "${CAUSAL_PATH}" "${CAUSAL_SHA256}"
verify_wheel "${MAMBA_PATH}" "${MAMBA_SHA256}"
python -m pip install --no-deps --force-reinstall "${CAUSAL_PATH}" "${MAMBA_PATH}"

echo "[4/4] Verifying compiled extensions and constructing ViM-S..."
python "${PROJECT_ROOT}/scripts/check_environment.py" \
  --check-vim-dependencies \
  --vim-model-dir "${VIM_MODEL_DIR}"

echo
echo "ViM dependencies are ready."
