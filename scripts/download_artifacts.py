#!/usr/bin/env python3
"""Download MLDR-KD teacher checkpoints and verified ViM wheels."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER = "YOUR_HF_USERNAME/MLDR-KD"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()

    if args.repo_id == PLACEHOLDER or "/" not in args.repo_id:
        raise SystemExit(
            "Replace YOUR_HF_USERNAME/MLDR-KD with the public Hugging Face repo ID."
        )

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        allow_patterns=["weights/*", "wheels/*"],
        local_dir=PROJECT_ROOT,
    )
    print(f"Artifacts downloaded under {PROJECT_ROOT}")


if __name__ == "__main__":
    main()
