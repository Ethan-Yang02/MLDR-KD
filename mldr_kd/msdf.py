"""Multi-Scale Dynamic Fusion (MSDF) for ResNet students."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn


class SepConv(nn.Module):
    """Depthwise separable convolution used by the ResNet projectors."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class CNNStageProjector(nn.Module):
    """Project one ResNet stage to a descriptor and class logits."""

    def __init__(
        self,
        stage_index: int,
        in_channels: int,
        project_dim: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        if stage_index not in (1, 2, 3, 4):
            raise ValueError("stage_index must be in [1, 4]")

        layers: list[nn.Module] = []
        current_channels = in_channels
        downsample_count = 4 - stage_index
        if downsample_count == 0:
            layers.append(
                nn.Conv2d(current_channels, project_dim, kernel_size=1, bias=False)
            )
        else:
            for block_index in range(downsample_count):
                is_last = block_index == downsample_count - 1
                out_channels = project_dim if is_last else current_channels * 2
                layers.append(SepConv(current_channels, out_channels))
                current_channels = out_channels

        self.project = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(project_dim, num_classes)
        self.apply(_initialize_weights)

    def forward(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.project(feature)
        descriptor = self.pool(projected).flatten(1)
        logits = self.classifier(descriptor)
        return descriptor, logits


class MLPStageGate(nn.Module):
    """Two-layer gate used by every main experiment."""

    def __init__(
        self,
        descriptor_dim: int,
        hidden_dim: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        return self.network(descriptors)


class MSDF(nn.Module):
    """Generate per-sample, per-stage, and per-class fusion weights."""

    def __init__(
        self,
        descriptor_dim: int,
        hidden_dim: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.gate = MLPStageGate(
            descriptor_dim,
            hidden_dim,
            num_classes,
        )
        self.apply(_initialize_weights)

    def forward(
        self,
        descriptors: Sequence[torch.Tensor],
        stage_logits: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not descriptors or len(descriptors) != len(stage_logits):
            raise ValueError("descriptors and stage_logits must have equal non-zero length")
        descriptor_tensor = torch.stack(tuple(descriptors), dim=1)
        logit_tensor = torch.stack(tuple(stage_logits), dim=1)
        weights = torch.softmax(self.gate(descriptor_tensor), dim=1)
        fused_logits = torch.sum(weights * logit_tensor, dim=1)
        return fused_logits, weights


def build_stage_projectors(
    student: Any,
    *,
    project_dim: int,
    num_classes: int,
) -> tuple[nn.ModuleList, int]:
    """Build projectors for all four ResNet stages."""

    stage_ids = (1, 2, 3, 4)
    projectors = nn.ModuleList(
        CNNStageProjector(
            stage_index=stage,
            in_channels=int(student.stage_shapes[stage - 1][0]),
            project_dim=project_dim,
            num_classes=num_classes,
        )
        for stage in stage_ids
    )
    return projectors, project_dim


def _initialize_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
