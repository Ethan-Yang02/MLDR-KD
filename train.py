#!/usr/bin/env python3
"""Train the ResNet-student MLDR-KD paper experiments."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml
from timm.data import Mixup, create_transform
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from torchvision import datasets
from tqdm import tqdm

from mldr_kd import MLDRKD, build_student, build_teacher, teacher_feature_dim


CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)
SUPPORTED_DATASETS = {"cifar100"}
EXPECTED_CLASSES = {"cifar100": 100}


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


class DistributedEvalSampler(Sampler[int]):
    """Shard evaluation data without padding or duplicate samples."""

    def __init__(self, dataset: Dataset[Any], rank: int, world_size: int) -> None:
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        return max((len(self.dataset) - self.rank + self.world_size - 1) // self.world_size, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MLDR-KD paper experiments with ResNet-18/101 students"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--teacher-checkpoint", type=Path, default=None)
    parser.add_argument("--student-checkpoint", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--vim-model-dir",
        type=Path,
        default=None,
        help="Official hustvl/Vim/vim directory for a Vision Mamba teacher",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a flat YAML key; VALUE is parsed with YAML (repeatable)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dist-backend",
        default=None,
        choices=("nccl", "gloo"),
        help="DDP backend (default: nccl for CUDA, gloo for CPU)",
    )
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args()


def setup_distributed(args: argparse.Namespace) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistributedContext(False, 0, 1, 0, torch.device(args.device))

    if "RANK" not in os.environ:
        raise RuntimeError("WORLD_SIZE > 1 but RANK is missing; launch with torchrun")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    requested_device = torch.device(args.device)
    if requested_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed CUDA training requested but CUDA is unavailable")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = args.dist_backend or "nccl"
    else:
        device = requested_device
        backend = args.dist_backend or "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    return DistributedContext(True, rank, world_size, local_rank, device)


def _normalize_dataset_name(name: Any) -> str:
    normalized = str(name).lower().replace("-", "_")
    aliases = {"cifar_100": "cifar100"}
    return aliases.get(normalized, normalized)


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("Configuration must be a YAML mapping")

    path_overrides = {
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
        "teacher_checkpoint": args.teacher_checkpoint,
        "vim_model_dir": args.vim_model_dir,
    }
    for key, value in path_overrides.items():
        if value is not None:
            config[key] = str(value)

    for entry in args.override:
        if "=" not in entry:
            raise ValueError(f"Invalid --override {entry!r}; expected KEY=VALUE")
        key, raw_value = entry.split("=", 1)
        key = key.strip()
        if key not in config:
            raise KeyError(
                f"Unknown override key {key!r}; add it to the YAML before overriding it"
            )
        config[key] = yaml.safe_load(raw_value)
    if args.no_amp:
        config["amp"] = False

    required = (
        "data_dir",
        "output_dir",
        "student",
        "teacher",
        "teacher_checkpoint",
        "dataset",
        "num_classes",
        "epochs",
        "batch_size",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise KeyError(f"Missing required config keys: {', '.join(missing)}")

    config["dataset"] = _normalize_dataset_name(config["dataset"])
    if config["dataset"] not in SUPPORTED_DATASETS:
        raise NotImplementedError(
            f"dataset must be one of {sorted(SUPPORTED_DATASETS)}"
        )
    expected_classes = EXPECTED_CLASSES[config["dataset"]]
    if int(config["num_classes"]) != expected_classes:
        raise ValueError(
            f"{config['dataset']} requires num_classes={expected_classes}, "
            f"received {config['num_classes']}"
        )
    if config["student"] not in {"resnet18", "resnet101"}:
        raise ValueError("This release keeps only resnet18 and resnet101 students")

    if int(config["epochs"]) <= 0 or int(config["batch_size"]) <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if float(config.get("temperature", 1.0)) <= 0:
        raise ValueError("temperature must be positive")
    return config


def seed_everything(seed: int, rank: int = 0) -> None:
    seed += rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_transforms(config: dict[str, Any]) -> tuple[Any, Any]:
    image_size = int(config.get("image_size", 224))
    common = {
        "input_size": (3, image_size, image_size),
        "mean": CIFAR100_MEAN,
        "std": CIFAR100_STD,
    }
    train_transform = create_transform(
        **common,
        is_training=True,
        scale=config.get("scale"),
        ratio=config.get("ratio"),
        hflip=float(config.get("horizontal_flip", 0.5)),
        vflip=float(config.get("vertical_flip", 0.0)),
        color_jitter=float(config.get("color_jitter", 0.0)),
        auto_augment=config.get("auto_augment"),
        interpolation=str(config.get("train_interpolation", "random")),
        re_prob=float(config.get("random_erasing", 0.0)),
        re_mode=str(config.get("random_erasing_mode", "pixel")),
        re_count=int(config.get("random_erasing_count", 1)),
    )
    validation_transform = create_transform(
        **common,
        is_training=False,
        interpolation=str(config.get("validation_interpolation", "bilinear")),
        crop_pct=float(config.get("crop_pct", 0.875)),
    )
    return train_transform, validation_transform


def make_loaders(
    config: dict[str, Any],
    context: DistributedContext,
) -> tuple[DataLoader[Any], DataLoader[Any], DistributedSampler[Any] | None]:
    train_transform, validation_transform = make_transforms(config)
    train_dataset: Dataset[Any] | None = None
    validation_dataset: Dataset[Any] | None = None
    root = Path(config["data_dir"]).expanduser()
    if not context.enabled or context.is_main:
        train_dataset = datasets.CIFAR100(
            root=root,
            train=True,
            download=True,
            transform=train_transform,
        )
        validation_dataset = datasets.CIFAR100(
            root=root,
            train=False,
            download=True,
            transform=validation_transform,
        )
    if context.enabled:
        dist.barrier()
        if not context.is_main:
            train_dataset = datasets.CIFAR100(
                root=root,
                train=True,
                download=False,
                transform=train_transform,
            )
            validation_dataset = datasets.CIFAR100(
                root=root,
                train=False,
                download=False,
                transform=validation_transform,
            )

    assert train_dataset is not None and validation_dataset is not None
    workers = int(config.get("workers", 4))
    common: dict[str, Any] = {
        "batch_size": int(config["batch_size"]),
        "num_workers": workers,
        "pin_memory": context.device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        common["prefetch_factor"] = int(config.get("prefetch_factor", 2))

    train_sampler = None
    validation_sampler = None
    if context.enabled:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=int(config.get("seed", 42)),
            drop_last=True,
        )
        validation_sampler = DistributedEvalSampler(
            validation_dataset,
            context.rank,
            context.world_size,
        )
    train_loader = DataLoader(
        train_dataset,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
        **common,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        sampler=validation_sampler,
        drop_last=False,
        **common,
    )
    return train_loader, validation_loader, train_sampler


def _resolve_dimension(value: Any, automatic: int, name: str) -> int:
    if value is None or str(value).lower() == "auto":
        return automatic
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{name} must be positive or 'auto'")
    return resolved


def make_distiller(config: dict[str, Any], device: torch.device) -> MLDRKD:
    student = build_student(config["student"], int(config["num_classes"]))
    teacher = build_teacher(
        config["teacher"],
        int(config["num_classes"]),
        config["teacher_checkpoint"],
        vim_model_dir=config.get("vim_model_dir"),
    )
    student_dim = int(student.stage_shapes[-1][0])
    project_dim = _resolve_dimension(
        config.get("project_dim", "auto"),
        max(student_dim, teacher_feature_dim(teacher)),
        "project_dim",
    )
    gate_hidden_dim = _resolve_dimension(
        config.get("gate_hidden_dim", "auto"),
        4 * int(config["num_classes"]),
        "gate_hidden_dim",
    )
    config["resolved_project_dim"] = project_dim
    config["resolved_gate_hidden_dim"] = gate_hidden_dim

    distiller = MLDRKD(
        student,
        teacher,
        num_classes=int(config["num_classes"]),
        project_dim=project_dim,
        gate_hidden_dim=gate_hidden_dim,
        temperature=float(config.get("temperature", 1.0)),
        alpha=float(config.get("alpha", 2.0)),
        ce_loss_weight=float(config.get("ce_loss_weight", 1.0)),
        logit_distill_weight=float(config.get("logit_distill_weight", 1.0)),
        stage_distill_weight=float(config.get("stage_distill_weight", 1.0)),
        fused_distill_weight=float(config.get("fused_distill_weight", 1.0)),
        label_smoothing=float(config.get("label_smoothing", 0.1)),
    )
    return distiller.to(device)


def make_optimizer(
    model: torch.nn.Module,
    config: dict[str, Any],
) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer_name = str(config.get("optimizer", "sgd")).lower()
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=float(config.get("learning_rate", 0.05)),
            momentum=float(config.get("momentum", 0.9)),
            weight_decay=float(config.get("weight_decay", 0.002)),
            nesterov=bool(config.get("nesterov", True)),
        )
    raise NotImplementedError(f"Unsupported optimizer: {optimizer_name}")


def set_learning_rate(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
) -> float:
    base_lr = float(config.get("learning_rate", 0.05))
    warmup_lr = float(config.get("warmup_learning_rate", 0.0001))
    warmup_epochs = int(config.get("warmup_epochs", 3))
    total_epochs = int(config["epochs"])

    if warmup_epochs > 0 and epoch < warmup_epochs:
        learning_rate = warmup_lr + (base_lr - warmup_lr) * epoch / warmup_epochs
    else:
        scheduler = str(config.get("scheduler", "cosine")).lower()
        if scheduler == "cosine":
            min_lr = float(config.get("min_learning_rate", 0.001))
            progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs - 1, 1)
            progress = min(max(progress, 0.0), 1.0)
            learning_rate = min_lr + 0.5 * (base_lr - min_lr) * (
                1.0 + math.cos(math.pi * progress)
            )
        elif scheduler == "step":
            decay_epochs = int(config.get("decay_epochs", 30))
            decay_rate = float(config.get("decay_rate", 0.1))
            learning_rate = base_lr * decay_rate ** (epoch // decay_epochs)
            learning_rate = max(
                learning_rate,
                float(config.get("min_learning_rate", 0.0)),
            )
        else:
            raise ValueError("scheduler must be 'cosine' or 'step'")
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return learning_rate


def resolve_accumulation_steps(
    config: dict[str, Any],
    context: DistributedContext,
) -> int:
    del context
    steps = int(config.get("accumulation_steps", 1))
    if steps <= 0:
        raise ValueError("accumulation_steps must be positive")
    return steps


def train_one_epoch(
    distiller: torch.nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    epoch: int,
    mixup_fn: Mixup | None,
    context: DistributedContext,
    accumulation_steps: int,
) -> dict[str, float]:
    distiller.train()
    totals = {
        "loss": 0.0,
        "ce": 0.0,
        "logit_distill": 0.0,
        "stage_distill": 0.0,
        "fused_distill": 0.0,
    }
    samples = 0
    total_batches = len(loader)
    optimizer.zero_grad(set_to_none=True)

    progress = tqdm(
        loader,
        desc=f"train {epoch + 1}",
        dynamic_ncols=True,
        disable=not context.is_main,
    )
    for batch_index, (images, labels) in enumerate(progress):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)

        group_start = (batch_index // accumulation_steps) * accumulation_steps
        group_size = min(accumulation_steps, total_batches - group_start)
        should_step = (
            (batch_index + 1) % accumulation_steps == 0
            or batch_index + 1 == total_batches
        )
        sync_context: Any = contextlib.nullcontext()
        if isinstance(distiller, DDP) and not should_step:
            sync_context = distiller.no_sync()

        with sync_context:
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                output = distiller(images, labels)
                scaled_loss = output.total_loss / group_size
            scaler.scale(scaled_loss).backward()

        if should_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        batch_size = images.shape[0]
        samples += batch_size
        totals["loss"] += output.total_loss.detach().item() * batch_size
        for name, value in output.losses.items():
            totals[name] += value.detach().item() * batch_size
        progress.set_postfix(loss=f"{totals['loss'] / samples:.4f}")

    if context.enabled:
        packed = torch.tensor(
            [*(totals[name] for name in totals), float(samples)],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        samples = int(packed[-1].item())
        totals = {name: packed[index].item() for index, name in enumerate(totals)}
    return {name: value / samples for name, value in totals.items()}


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    amp_enabled: bool,
    description: str,
    context: DistributedContext,
) -> dict[str, float]:
    """Evaluate the teacher or the original student network only."""

    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    samples = 0
    for images, labels in tqdm(
        loader,
        desc=description,
        dynamic_ncols=True,
        disable=not context.is_main,
    ):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(images)
        top_k = min(5, logits.shape[1])
        top = logits.topk(k=top_k, dim=1).indices
        correct = top.eq(labels.unsqueeze(1))
        correct_top1 += correct[:, :1].sum().item()
        correct_top5 += correct.sum().item()
        samples += labels.numel()

    if context.enabled:
        packed = torch.tensor(
            [correct_top1, correct_top5, samples],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        correct_top1, correct_top5, samples = packed.tolist()
    return {
        "top1": 100.0 * correct_top1 / samples,
        "top5": 100.0 * correct_top5 / samples,
    }


def save_student(
    path: Path,
    distiller: MLDRKD,
    config: dict[str, Any],
    epoch: int,
    accuracy: float,
) -> None:
    torch.save(
        {
            "model": distiller.student.backbone.state_dict(),
            "student": config["student"],
            "num_classes": int(config["num_classes"]),
            "epoch": epoch,
            "top1": accuracy,
        },
        path,
    )


def save_training_state(
    path: Path,
    distiller: MLDRKD,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    best_top1: float,
) -> None:
    state_without_teacher = {
        key: value
        for key, value in distiller.state_dict().items()
        if not key.startswith("teacher.")
    }
    torch.save(
        {
            "distiller": state_without_teacher,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_top1": best_top1,
        },
        path,
    )


def _move_optimizer_state(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def load_training_state(
    path: Path,
    distiller: MLDRKD,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> tuple[int, float]:
    payload = torch.load(path, map_location="cpu")
    incompatible = distiller.load_state_dict(payload["distiller"], strict=False)
    unexpected = list(incompatible.unexpected_keys)
    invalid_missing = [
        key for key in incompatible.missing_keys if not key.startswith("teacher.")
    ]
    if unexpected or invalid_missing:
        raise RuntimeError(
            f"Invalid resume checkpoint; missing={invalid_missing}, unexpected={unexpected}"
        )
    optimizer.load_state_dict(payload["optimizer"])
    _move_optimizer_state(optimizer, device)
    scaler.load_state_dict(payload.get("scaler", {}))
    return int(payload["epoch"]) + 1, float(payload.get("best_top1", 0.0))


def load_student_checkpoint(path: Path, student: torch.nn.Module) -> None:
    payload = torch.load(path, map_location="cpu")
    state_dict = payload.get("model", payload)
    student.backbone.load_state_dict(state_dict, strict=True)


def append_metrics(path: Path, metrics: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metrics, sort_keys=True) + "\n")


def unwrap_distiller(model: torch.nn.Module) -> MLDRKD:
    if isinstance(model, DDP):
        return model.module
    if not isinstance(model, MLDRKD):
        raise TypeError(f"Expected MLDRKD or DDP, got {type(model).__name__}")
    return model


def broadcast_buffers(module: torch.nn.Module) -> None:
    if not dist.is_available() or not dist.is_initialized():
        return
    for buffer in module.buffers():
        dist.broadcast(buffer, src=0)


def make_mixup(config: dict[str, Any]) -> Mixup | None:
    mixup_alpha = float(config.get("mixup", 0.0))
    cutmix_alpha = float(config.get("cutmix", 0.0))
    if mixup_alpha <= 0 and cutmix_alpha <= 0:
        return None
    return Mixup(
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        prob=1.0,
        switch_prob=0.5,
        mode="batch",
        label_smoothing=float(config.get("label_smoothing", 0.1)),
        num_classes=int(config["num_classes"]),
    )


def write_resolved_config(path: Path, config: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)


def main() -> None:
    args = parse_args()
    config = load_config(args)
    if args.print_config:
        print(yaml.safe_dump(config, sort_keys=True))
        return

    context = setup_distributed(args)
    seed_everything(int(config.get("seed", 42)), context.rank)
    if context.device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = context.device
    amp_enabled = bool(config.get("amp", False)) and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(config.get("cudnn_benchmark", True))

    output_dir = Path(config["output_dir"]).expanduser()
    if context.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    if context.enabled:
        dist.barrier()

    if args.eval_only:
        if args.student_checkpoint is None:
            raise ValueError("--eval-only requires --student-checkpoint")
        if context.is_main:
            write_resolved_config(output_dir / "resolved_config.yaml", config)
        _, validation_loader, _ = make_loaders(config, context)
        student = build_student(
            config["student"],
            int(config["num_classes"]),
        ).to(device)
        load_student_checkpoint(args.student_checkpoint, student)
        metrics = evaluate(
            student,
            validation_loader,
            device,
            amp_enabled,
            "student-only eval",
            context,
        )
        if context.is_main:
            print(json.dumps(metrics, indent=2))
        return

    distiller = make_distiller(config, device)
    accumulation_steps = resolve_accumulation_steps(config, context)
    config["resolved_accumulation_steps"] = accumulation_steps
    config["resolved_world_size"] = context.world_size
    config["resolved_effective_batch_size"] = (
        int(config["batch_size"]) * context.world_size * accumulation_steps
    )
    if context.is_main:
        write_resolved_config(output_dir / "resolved_config.yaml", config)

    train_loader, validation_loader, train_sampler = make_loaders(config, context)
    counts = distiller.parameter_counts()
    if context.is_main:
        print(
            "Parameters: "
            f"student/inference={counts['inference'] / 1e6:.3f}M, "
            f"training auxiliary={counts['training_auxiliary'] / 1e6:.3f}M"
        )
        print("Validation always uses the original ResNet student alone.")
        print(
            f"world_size={context.world_size}, per-GPU batch={config['batch_size']}, "
            f"accumulation={accumulation_steps}, "
            f"effective batch={config['resolved_effective_batch_size']}"
        )

    if args.student_checkpoint is not None:
        load_student_checkpoint(args.student_checkpoint, distiller.student)

    optimizer = make_optimizer(distiller, config)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    start_epoch = 0
    best_top1 = 0.0
    if args.resume is not None:
        start_epoch, best_top1 = load_training_state(
            args.resume,
            distiller,
            optimizer,
            scaler,
            device,
        )

    training_model: torch.nn.Module = distiller
    if context.enabled:
        ddp_options: dict[str, Any] = {
            "broadcast_buffers": True,
            "find_unused_parameters": False,
        }
        if device.type == "cuda":
            ddp_options.update(
                device_ids=[context.local_rank],
                output_device=context.local_rank,
            )
        training_model = DDP(distiller, **ddp_options)

    teacher_metrics = evaluate(
        distiller.teacher,
        validation_loader,
        device,
        amp_enabled,
        "teacher verification",
        context,
    )
    if context.is_main:
        print(f"Teacher Top-1: {teacher_metrics['top1']:.2f}%")

    metrics_path = output_dir / "metrics.jsonl"
    total_epochs = int(config["epochs"])
    save_every = int(config.get("save_every", 25))
    mixup_fn = make_mixup(config)
    for epoch in range(start_epoch, total_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        epoch_start = time.time()
        learning_rate = set_learning_rate(optimizer, epoch, config)
        train_metrics = train_one_epoch(
            training_model,
            train_loader,
            optimizer,
            scaler,
            device,
            amp_enabled,
            epoch,
            mixup_fn,
            context,
            accumulation_steps,
        )

        core_distiller = unwrap_distiller(training_model)
        broadcast_buffers(core_distiller.student)
        validation_metrics = evaluate(
            core_distiller.student,
            validation_loader,
            device,
            amp_enabled,
            "student validation",
            context,
        )
        record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "seconds": time.time() - epoch_start,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        if context.is_main:
            append_metrics(metrics_path, record)
            print(json.dumps(record, indent=2))

        if context.is_main and validation_metrics["top1"] > best_top1:
            best_top1 = validation_metrics["top1"]
            save_student(
                output_dir / "best_student.pth",
                core_distiller,
                config,
                epoch,
                best_top1,
            )
        if context.is_main and (
            (epoch + 1) % save_every == 0 or epoch + 1 == total_epochs
        ):
            save_training_state(
                output_dir / "last_training.pth",
                core_distiller,
                optimizer,
                scaler,
                epoch,
                best_top1,
            )

    if context.is_main:
        print(f"Best student-only Top-1: {best_top1:.2f}%")


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
