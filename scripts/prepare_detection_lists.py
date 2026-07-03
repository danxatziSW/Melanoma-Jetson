"""Materializes outputs/detection/train.txt / val.txt (relative, tracked in git) into
absolute-path files Ultralytics can train against on this machine.

Ultralytics reads these list files itself, so paths in them have to be real absolute
paths (or resolvable relative to the list file's own location) — our usual
resolve_dataset_paths() helper only works on pandas DataFrames we control. This script
is the detection-specific equivalent: run it once per machine (or whenever
paths.melanoma_data changes) before training the detector.

Usage: python scripts/prepare_detection_lists.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config


def main() -> None:
    cfg = load_config()
    root = Path(cfg.paths.melanoma_data)
    det_dir = Path(cfg.paths.outputs) / "detection"

    for name in ("train", "val"):
        src = det_dir / f"{name}.txt"
        dst = det_dir / f"{name}_resolved.txt"
        lines = src.read_text().splitlines()
        resolved = [str(root / line) for line in lines if line.strip()]
        dst.write_text("\n".join(resolved) + "\n")
        print(f"  {src.name} ({len(lines)} images) -> {dst.name}")

    print(f"\n  Data root : {root}")
    print(f"  Output    : {det_dir}")


if __name__ == "__main__":
    main()
