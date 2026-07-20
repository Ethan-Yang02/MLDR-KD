# MLDR-KD

## Installation

The repository includes the complete official Vim source at commit
`dd0358ad1e42701f22afbefa0717cc8825cf9f45`.

The paper results were reproduced with 4 x NVIDIA RTX 3090 (24 GiB), driver
550.78, Python 3.10.13, PyTorch 2.1.1 + CUDA 11.8, TorchVision 0.16.1, and
timm 0.6.5. The setup does not depend on a local environment name, project
path, system CUDA toolkit, or `nvcc`.

```bash
bash scripts/create_environment.sh
conda activate mldr-kd

python scripts/download_artifacts.py \
  --repo-id YYaoXin/MLDR-KD

bash scripts/install_vim_dependencies.sh
```

Keep the `weights/` and `wheels/` directory layout documented in
their README files. CIFAR-100 is downloaded automatically.

## Running Experiments

```bash
GPU_IDS=0 bash scripts/run_experiment.sh configs/cifar100/resnet18_vit_small.yaml
GPU_IDS=0 bash scripts/run_experiment.sh configs/cifar100/resnet18_swin_tiny.yaml
GPU_IDS=0 bash scripts/run_experiment.sh configs/cifar100/resnet18_mixer_b16.yaml
GPU_IDS=0 bash scripts/run_experiment.sh configs/cifar100/resnet18_vim_small.yaml
GPU_IDS=0 bash scripts/run_experiment.sh configs/cifar100/resnet101_vit_large.yaml
```

Use `GPU_IDS=0,2` for multiple GPUs. The process count and an available
`MASTER_PORT` are selected automatically. Override paths without editing YAML:

```bash
DATA_DIR=/path/to/data \
TEACHER_CKPT=/path/to/teacher.pth \
OUTPUT_DIR=/path/to/output \
GPU_IDS=0 \
bash scripts/run_experiment.sh configs/cifar100/resnet18_vit_small.yaml
```

The launcher intentionally uses `torch.distributed.launch`; its deprecation
warning can be ignored.
