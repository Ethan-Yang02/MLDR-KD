#!/usr/bin/env python3
"""Verify the minimal MLDR-KD runtime and the visible GPUs."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import inspect
import sys
import warnings
from pathlib import Path

import timm
import torch
import torchvision

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mldr_kd import register_vision_mamba


EXPECTED = {
    "torch": "2.1.1",
    "torchvision": "0.16.1",
    "timm": "0.6.5",
    "numpy": "1.26.3",
    "PyYAML": "6.0.3",
    "tqdm": "4.65.2",
    "huggingface-hub": "0.24.3",
}
VIM_EXPECTED = {
    "einops": "0.8.0",
    "transformers": "4.43.3",
    "causal-conv1d": "1.1.1",
    "mamba-ssm": "1.2.0.post1",
}
VIM_MODULES = (
    "pkg_resources",
    "causal_conv1d",
    "mamba_ssm",
    "causal_conv1d_cuda",
    "selective_scan_cuda",
)
EXPECTED_CUDA = "11.8"


def base_version(version: str) -> str:
    return version.split("+", 1)[0]


def check_distribution(distribution: str, expected: str) -> bool:
    try:
        installed = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        print(f"{distribution}: MISSING [expected {expected}]")
        return True

    status = "OK" if base_version(installed) == expected else "MISMATCH"
    print(f"{distribution}: {installed} [{status}; expected {expected}]")
    return status != "OK"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-vim-dependencies",
        action="store_true",
        help="also verify the optional ViM packages and compiled CUDA extensions",
    )
    parser.add_argument("--vim-model-dir", type=Path)
    args = parser.parse_args()
    check_vim = args.check_vim_dependencies or args.vim_model_dir is not None

    print(f"Python: {sys.version.split()[0]}")
    failed = False
    for distribution, expected in EXPECTED.items():
        failed |= check_distribution(distribution, expected)

    cuda_runtime = torch.version.cuda
    cuda_status = "OK" if cuda_runtime == EXPECTED_CUDA else "MISMATCH"
    print(
        f"CUDA runtime in PyTorch: {cuda_runtime} "
        f"[{cuda_status}; expected {EXPECTED_CUDA}]"
    )
    failed |= cuda_status != "OK"
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Visible GPUs: {torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        print(
            f"  GPU {index}: {props.name}, "
            f"{props.total_memory / 1024 ** 3:.1f} GiB, sm_{props.major}{props.minor}"
        )

    for model_name in ("resnet18", "resnet101", "vit_small_patch16_224", "swin_tiny_patch4_window7_224", "mixer_b16_224", "vit_large_patch16_224"):
        timm.create_model(model_name, pretrained=False, num_classes=10)
        print(f"timm model {model_name}: OK")

    if check_vim:
        print("ViM optional dependencies:")
        for distribution, expected in VIM_EXPECTED.items():
            failed |= check_distribution(distribution, expected)

        cxx11_abi = bool(torch._C._GLIBCXX_USE_CXX11_ABI)
        abi_status = "OK" if not cxx11_abi else "MISMATCH"
        print(f"PyTorch CXX11 ABI: {cxx11_abi} [{abi_status}; expected False]")
        failed |= cxx11_abi

        for module_name in VIM_MODULES:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    importlib.import_module(module_name)
            except Exception as exc:
                print(f"import {module_name}: FAILED [{exc}]")
                failed = True
            else:
                print(f"import {module_name}: OK")

        try:
            from mamba_ssm.modules.mamba_simple import Mamba

            supports_bimamba = "bimamba_type" in inspect.signature(Mamba.__init__).parameters
        except Exception as exc:
            print(f"ViM Mamba API: FAILED [{exc}]")
            failed = True
        else:
            if supports_bimamba:
                print("ViM Mamba API: OK")
            else:
                print("ViM Mamba API: FAILED [Mamba.__init__ lacks bimamba_type; use the ViM-compatible +vim wheel]")
                failed = True

    if args.vim_model_dir is not None:
        model_dir = args.vim_model_dir.expanduser().resolve()
        if not (model_dir / "models_mamba.py").is_file():
            raise FileNotFoundError(model_dir / "models_mamba.py")
        try:
            register_vision_mamba(model_dir)
            timm.create_model(
                "vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2",
                pretrained=False,
                num_classes=100,
            )
        except Exception as exc:
            print(f"Vision Mamba construction: FAILED [{exc}]")
            failed = True
        else:
            print("Vision Mamba construction: OK")

    if failed:
        message = (
            "Environment check failed. Base-version mismatches should be fixed "
            "with scripts/create_environment.sh."
        )
        if check_vim:
            message += (
                " Missing or incompatible ViM packages should be fixed with "
                "scripts/install_vim_dependencies.sh."
            )
        raise SystemExit(message)
    print("Environment check passed.")


if __name__ == "__main__":
    main()
