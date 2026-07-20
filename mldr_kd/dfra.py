"""Decoupled Fine-grained Relation Alignment (DFRA)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DFRAOutput:
    """Loss values produced by one DFRA comparison."""

    total: torch.Tensor
    class_relation: torch.Tensor
    sample_relation: torch.Tensor
    probability: torch.Tensor


def probability_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """KL(teacher || student) on the softened class probabilities."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "student and teacher logits must have the same shape; got "
            f"{tuple(student_logits.shape)} and {tuple(teacher_logits.shape)}"
        )
    student = student_logits / temperature
    teacher = teacher_logits.detach().to(student_logits.device) / temperature
    return F.kl_div(
        F.log_softmax(student, dim=-1),
        F.softmax(teacher, dim=-1),
        reduction="batchmean",
    )


def _relation_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    normalization_dim: int,
) -> torch.Tensor:
    """KL(teacher || student) for a relation tensor."""

    log_student = F.log_softmax(student_logits, dim=normalization_dim)
    teacher = F.softmax(teacher_logits, dim=normalization_dim)
    elementwise = F.kl_div(log_student, teacher, reduction="none")
    return elementwise.sum(dim=-1).mean()


class DFRALoss(nn.Module):
    """Align class-wise, sample-wise, and marginal teacher distributions."""

    def __init__(self, temperature: float = 1.0, alpha: float = 2.0) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.temperature = float(temperature)
        self.alpha = float(alpha)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> DFRAOutput:
        if student_logits.ndim != 2 or teacher_logits.ndim != 2:
            raise ValueError("DFRA expects [batch, classes] logit tensors")
        if student_logits.shape != teacher_logits.shape:
            raise ValueError(
                "student and teacher logits must have the same shape; got "
                f"{tuple(student_logits.shape)} and {tuple(teacher_logits.shape)}"
            )

        teacher_logits = teacher_logits.detach().to(student_logits.device)
        student = student_logits / self.temperature
        teacher = teacher_logits / self.temperature
        num_classes = student.shape[-1]
        scale = num_classes ** -0.5

        student_class = torch.matmul(student.unsqueeze(-1), student.unsqueeze(-2)) * scale
        teacher_class = torch.matmul(teacher.unsqueeze(-1), teacher.unsqueeze(-2)) * scale

        student_by_class = student.transpose(0, 1)
        teacher_by_class = teacher.transpose(0, 1)
        student_sample = torch.matmul(
            student_by_class.unsqueeze(-1), student_by_class.unsqueeze(-2)
        ) * scale
        teacher_sample = torch.matmul(
            teacher_by_class.unsqueeze(-1), teacher_by_class.unsqueeze(-2)
        ) * scale

        class_loss = _relation_kl(
            student_class, teacher_class, normalization_dim=-2
        )
        sample_loss = _relation_kl(
            student_sample, teacher_sample, normalization_dim=-2
        )
        probability_loss = probability_kl(
            student_logits,
            teacher_logits,
            temperature=self.temperature,
        )

        total = class_loss + sample_loss + self.alpha * probability_loss
        return DFRAOutput(
            total=total,
            class_relation=class_loss,
            sample_relation=sample_loss,
            probability=probability_loss,
        )
