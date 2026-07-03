"""Trains the YOLOv8 lesion detector, then flattens the checkpoint to where everything
downstream expects it.

There's no Ultralytics Python training API used here beyond what `yolo detect train` already
does; this is a thin wrapper around that CLI command, kept as a script so training the
detector looks like every other training step in this repo (`python scripts/...`) instead of
a multi-line command you have to type and then fix up by hand afterwards.

Usage: python scripts/no_seg/train_detection.py [--epochs 40] [--imgsz 640] [--batch 16] [--seed 0]
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.prepare_detection_lists import main as prepare_detection_lists
from src.utils.config import load_config

_RUN_NAME    = "yolov8n_lesion"
_BASE_MODEL  = "yolov8n.pt"


def _yolo_executable() -> str:
    """Resolves the yolo CLI next to the current interpreter so this works whether or
    not the venv happens to be on PATH."""
    candidate = Path(sys.executable).parent / ("yolo.exe" if os.name == "nt" else "yolo")
    return str(candidate) if candidate.exists() else "yolo"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the YOLOv8 lesion detector.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz",  type=int, default=640)
    parser.add_argument("--batch",  type=int, default=16)
    parser.add_argument("--seed",   type=int, default=0)
    parser.add_argument("--model",  default=_BASE_MODEL, help="Base checkpoint to fine-tune from")
    args = parser.parse_args()

    base_cfg     = load_config()
    det_dir      = Path(base_cfg.paths.outputs) / "detection"
    ckpt_dir     = det_dir / "checkpoints"
    dataset_yaml = det_dir / "dataset.yaml"

    print("  Refreshing train_resolved.txt / val_resolved.txt for this machine ...\n")
    prepare_detection_lists()

    print(f"\n  Training {args.model} -> {_RUN_NAME}  "
          f"({args.epochs} epochs, imgsz={args.imgsz}, batch={args.batch})\n")

    cmd = [
        _yolo_executable(), "detect", "train",
        f"model={args.model}",
        f"data={dataset_yaml}",
        f"epochs={args.epochs}",
        f"imgsz={args.imgsz}",
        f"batch={args.batch}",
        f"seed={args.seed}",
        f"project={ckpt_dir}",
        f"name={_RUN_NAME}",
        "exist_ok=True",  # reuse checkpoints/yolov8n_lesion/ on rerun instead of
                          # incrementing to yolov8n_lesion2/, which would break the
                          # hardcoded copy step below
    ]
    subprocess.run(cmd, check=True)

    # Ultralytics writes <project>/<name>/weights/best.pt; flatten it to <project>/best.pt,
    # which is where convert_to_onnx.py, the dashboard, and JETSON.md all expect to find it.
    trained = ckpt_dir / _RUN_NAME / "weights" / "best.pt"
    flat    = ckpt_dir / "best.pt"
    if trained.exists():
        shutil.copy(trained, flat)
        print(f"\n  Copied {trained.relative_to(ckpt_dir)} -> {flat.name}")
    else:
        print(f"\n  [WARN] expected checkpoint not found: {trained}")


if __name__ == "__main__":
    main()
