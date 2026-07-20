"""ResNet students and heterogeneous teacher construction."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from timm import create_model


_RESNET_STAGE_SHAPES: dict[str, tuple[tuple[int, int, int], ...]] = {
    "resnet18": (
        (64, 56, 56),
        (128, 28, 28),
        (256, 14, 14),
        (512, 7, 7),
    ),
    "resnet101": (
        (256, 56, 56),
        (512, 28, 28),
        (1024, 14, 14),
        (2048, 7, 7),
    ),
}


class StageFeatureStudent(nn.Module):
    """Wrap a ResNet and capture its four residual stages during training.

    Hooks are enabled only inside forward_with_features. Normal validation and
    deployment call forward, so none of the MLDR-KD training-only feature paths
    participate in inference.
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        stage_paths: tuple[str, str, str, str],
        stage_shapes: tuple[tuple[int, int, int], ...],
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.stage_kind = "cnn"
        self.stage_shapes = stage_shapes
        self._capture_enabled = False
        self._captured: list[torch.Tensor | None] = [None] * len(stage_paths)
        self._hook_handles = []

        for index, path in enumerate(stage_paths):
            module = _resolve_module(backbone, path)
            self._hook_handles.append(
                module.register_forward_hook(self._make_hook(index))
            )

    def _make_hook(self, index: int):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if self._capture_enabled:
                self._captured[index] = _first_tensor(output)

        return hook

    def forward_with_features(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        self._captured = [None] * len(self._captured)
        self._capture_enabled = True
        try:
            logits = self.backbone(inputs)
        finally:
            self._capture_enabled = False
        if any(feature is None for feature in self._captured):
            missing = [
                str(index + 1)
                for index, feature in enumerate(self._captured)
                if feature is None
            ]
            raise RuntimeError(f"Failed to capture ResNet stages: {', '.join(missing)}")
        features = [feature for feature in self._captured if feature is not None]
        self._captured = [None] * len(self._captured)
        return logits, features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone(inputs)


def build_student(name: str, num_classes: int) -> StageFeatureStudent:
    """Build a paper ResNet student from scratch."""

    if name not in _RESNET_STAGE_SHAPES:
        supported = ", ".join(sorted(_RESNET_STAGE_SHAPES))
        raise NotImplementedError(
            f"This ResNet-only release supports {supported}; received {name!r}."
        )
    backbone = create_model(name, pretrained=False, num_classes=num_classes)
    return StageFeatureStudent(
        backbone,
        stage_paths=("layer1", "layer2", "layer3", "layer4"),
        stage_shapes=_RESNET_STAGE_SHAPES[name],
    )


def build_teacher(
    name: str,
    num_classes: int,
    checkpoint: str | Path,
    *,
    vim_model_dir: str | Path | None = None,
) -> nn.Module:
    """Build and strictly load one heterogeneous paper teacher.

    Standard teachers are provided by timm. Vision Mamba teachers are
    registered from the official hustvl/Vim source supplied by the user.
    """

    checkpoint_path = Path(checkpoint).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {checkpoint_path}. "
            "See weights/README.md for the expected filenames."
        )

    if name.startswith("vim_"):
        register_vision_mamba(vim_model_dir)
    teacher = create_model(name, pretrained=False, num_classes=num_classes)

    payload: Any = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_state_dict(payload)
    state_dict = {_strip_prefixes(key): value for key, value in state_dict.items()}
    try:
        teacher.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            "The teacher checkpoint is incompatible with "
            f"{name} configured for {num_classes} classes. Original error:\n{error}"
        ) from error

    teacher.requires_grad_(False)
    teacher.eval()
    return teacher


def teacher_feature_dim(teacher: nn.Module) -> int:
    """Return the final representation width used by the original projector."""

    for attribute in ("num_features", "embed_dim"):
        value = getattr(teacher, attribute, None)
        if isinstance(value, int) and value > 0:
            return value
    head = getattr(teacher, "head", None)
    if isinstance(head, nn.Linear):
        return int(head.in_features)
    raise AttributeError(
        f"Cannot infer the final feature dimension of {type(teacher).__name__}; "
        "set project_dim explicitly in the experiment YAML."
    )


def register_vision_mamba(vim_model_dir: str | Path | None) -> None:
    """Register official ViM models with Mamba layer-norm compatibility."""

    if vim_model_dir is None:
        raise ValueError(
            "A ViM teacher requires --vim-model-dir (or vim_model_dir in YAML) "
            "pointing to the official hustvl/Vim/vim directory."
        )
    directory = Path(vim_model_dir).expanduser().resolve()
    model_file = directory / "models_mamba.py"
    if not model_file.is_file():
        raise FileNotFoundError(
            f"models_mamba.py not found under {directory}. Clone hustvl/Vim and "
            "point VIM_MODEL_DIR to its vim/ subdirectory."
        )
    directory_text = str(directory)
    if directory_text not in sys.path:
        sys.path.insert(0, directory_text)

    _install_mamba_layer_norm_compatibility()
    module = importlib.import_module("models_mamba")
    if getattr(module, "RMSNorm", None) is None:
        raise ImportError(
            "Vision Mamba imported without RMSNorm. The installed mamba-ssm "
            "layer-norm module is incompatible with this Vim checkout."
        )


def _install_mamba_layer_norm_compatibility() -> None:
    current_name = "mamba_ssm.ops.triton.layer_norm"
    legacy_name = "mamba_ssm.ops.triton.layernorm"
    try:
        importlib.import_module(current_name)
    except ModuleNotFoundError as error:
        if error.name != current_name:
            raise
        legacy_module = importlib.import_module(legacy_name)
        sys.modules[current_name] = legacy_module


def _resolve_module(root: nn.Module, path: str) -> nn.Module:
    module: Any = root
    for component in path.split("."):
        module = module[int(component)] if component.isdigit() else getattr(module, component)
    if not isinstance(module, nn.Module):
        raise TypeError(f"Resolved stage {path!r} is not an nn.Module")
    return module


def _first_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            try:
                return _first_tensor(item)
            except TypeError:
                continue
    raise TypeError(f"Stage output does not contain a tensor: {type(output)!r}")


def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint must contain a state-dict-like mapping")
    for key in ("state_dict_ema", "model_ema", "state_dict", "model"):
        candidate = payload.get(key)
        if isinstance(candidate, dict):
            return candidate
    if payload and all(torch.is_tensor(value) for value in payload.values()):
        return payload
    raise KeyError(
        "Checkpoint must be a raw state dict or contain one of: "
        "state_dict_ema, model_ema, state_dict, model"
    )


def _strip_prefixes(key: str) -> str:
    prefixes = ("module.", "teacher.", "model.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    return key
