"""Training-only MLDR-KD wrapper for ResNet students."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dfra import DFRALoss
from .models import StageFeatureStudent
from .msdf import MSDF, build_stage_projectors


@dataclass
class MLDRKDOutput:
    student_logits: torch.Tensor
    total_loss: torch.Tensor
    losses: dict[str, torch.Tensor]
    fusion_weights: torch.Tensor


class MLDRKD(nn.Module):
    """MLDR-KD with a ResNet student and training-only auxiliary modules."""

    def __init__(
        self,
        student: StageFeatureStudent,
        teacher: nn.Module,
        *,
        num_classes: int,
        project_dim: int,
        gate_hidden_dim: int,
        temperature: float = 1.0,
        alpha: float = 2.0,
        ce_loss_weight: float = 1.0,
        logit_distill_weight: float = 1.0,
        stage_distill_weight: float = 1.0,
        fused_distill_weight: float = 1.0,
        label_smoothing: float = 0.1,
    ) -> None:
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.teacher.requires_grad_(False)

        self.projectors, descriptor_dim = build_stage_projectors(
            student,
            project_dim=project_dim,
            num_classes=num_classes,
        )
        self.msdf = MSDF(
            descriptor_dim=descriptor_dim,
            hidden_dim=gate_hidden_dim,
            num_classes=num_classes,
        )
        self.distillation_loss = DFRALoss(
            temperature=temperature,
            alpha=alpha,
        )
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.ce_loss_weight = float(ce_loss_weight)
        self.logit_distill_weight = float(logit_distill_weight)
        self.stage_distill_weight = float(stage_distill_weight)
        self.fused_distill_weight = float(fused_distill_weight)
        if min(
            self.ce_loss_weight,
            self.logit_distill_weight,
            self.stage_distill_weight,
            self.fused_distill_weight,
        ) <= 0:
            raise ValueError("main experiment loss weights must be positive")

    def train(self, mode: bool = True) -> "MLDRKD":
        super().train(mode)
        self.teacher.eval()
        return self

    def _distill(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        return self.distillation_loss(student_logits, teacher_logits).total

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> MLDRKDOutput:
        with torch.no_grad():
            teacher_logits = self.teacher(images)

        student_logits, stage_features = self.student.forward_with_features(images)
        descriptors: list[torch.Tensor] = []
        stage_logits: list[torch.Tensor] = []
        for projector, feature in zip(self.projectors, stage_features, strict=True):
            descriptor, logits = projector(feature)
            descriptors.append(descriptor)
            stage_logits.append(logits)

        fused_logits, fusion_weights = self.msdf(descriptors, stage_logits)

        if labels.ndim == 2:
            ce_loss = torch.sum(
                -labels * F.log_softmax(student_logits, dim=-1), dim=-1
            ).mean()
        else:
            ce_loss = self.ce(student_logits, labels)

        logit_distill = self._distill(student_logits, teacher_logits)
        stage_distill = torch.stack(
            [self._distill(logits, teacher_logits) for logits in stage_logits]
        ).sum()
        fused_distill = self._distill(fused_logits, teacher_logits)

        weighted_losses = {
            "ce": self.ce_loss_weight * ce_loss,
            "logit_distill": self.logit_distill_weight * logit_distill,
            "stage_distill": self.stage_distill_weight * stage_distill,
            "fused_distill": self.fused_distill_weight * fused_distill,
        }
        total_loss = torch.stack(tuple(weighted_losses.values())).sum()
        return MLDRKDOutput(
            student_logits=student_logits,
            total_loss=total_loss,
            losses=weighted_losses,
            fusion_weights=fusion_weights,
        )

    def parameter_counts(self) -> dict[str, int]:
        student_parameters = sum(
            parameter.numel() for parameter in self.student.parameters()
        )
        auxiliary_parameters = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if not name.startswith("student.") and not name.startswith("teacher.")
        )
        return {
            "student": student_parameters,
            "training_auxiliary": auxiliary_parameters,
            "inference": student_parameters,
        }
