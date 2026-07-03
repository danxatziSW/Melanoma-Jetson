"""Exports trained classifier checkpoints, and the YOLOv8 detector, to ONNX.

Needed because ONNX exports aren't committed to the repo (too large, easy to regenerate),
so anyone who trains or re-trains a model has to produce their own ONNX before running
scripts/deployedTensorrt/convert.py or anything else downstream that expects one.

Usage:
    python scripts/no_seg/convert_to_onnx.py --models resnet50 --datasets ham10000 --aug none
    python scripts/no_seg/convert_to_onnx.py --models all --datasets all --aug none light none_sens
    python scripts/no_seg/convert_to_onnx.py --detection --skip-classifiers
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.models.registry import build_model, uses_metadata
from src.utils.config import load_config

ALL_MODELS = [
    "resnet50", "efficientnet_b2", "mobilenetv3_large",
    "convnext_tiny_se", "medfusionnet", "yolov8_cls",
]
ALL_DATASETS = ["ham10000", "isic2019", "isic2020"]
ALL_AUGS     = ["none", "light", "none_sens"]

_ONNX_OPSET   = 17
_METADATA_DIM = 8


def _build_matching_model(model_name: str, state_dict: dict, config, num_classes: int):
    """yolov8_cls can be saved as the ultralytics wrapper ("inner.*") or as the
    MobileNetV3-RW timm fallback ("conv_stem.*"), depending on whether ultralytics
    was installed at training time."""
    if model_name != "yolov8_cls":
        return build_model(model_name, config, num_classes=num_classes)
    first_key = next(iter(state_dict))
    if first_key.startswith(("conv_stem", "blocks", "classifier")):
        import timm
        return timm.create_model(
            "mobilenetv3_rw", pretrained=False,
            num_classes=num_classes, drop_rate=getattr(config, "dropout", 0.2),
        )
    return build_model(model_name, config, num_classes=num_classes)


def convert_one(dataset: str, model_name: str, aug_mode: str, noseg_dir: Path) -> Path | None:
    run_id    = f"{model_name}_{aug_mode}"
    ckpt_file = noseg_dir / dataset / run_id / "checkpoints" / f"{run_id}.pt"
    if not ckpt_file.exists():
        print(f"  [SKIP] {dataset}/{run_id}: checkpoint not found: {ckpt_file}")
        return None

    config     = load_config(model_name)
    state_dict = torch.load(ckpt_file, map_location="cpu")
    model      = _build_matching_model(model_name, state_dict, config, num_classes=2)
    model.load_state_dict(state_dict)
    model.eval()

    with_meta = uses_metadata(model_name)
    input_size = getattr(config, "input_size", 224)
    onnx_path  = noseg_dir / dataset / run_id / "onnx" / f"{run_id}.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_img = torch.zeros(1, 3, input_size, input_size)
    kwargs = dict(opset_version=_ONNX_OPSET, do_constant_folding=True, dynamo=False)
    if with_meta:
        dummy_meta = torch.zeros(1, _METADATA_DIM)
        torch.onnx.export(model, (dummy_img, dummy_meta), str(onnx_path),
                          input_names=["image", "metadata"], output_names=["logits"], **kwargs)
    else:
        torch.onnx.export(model, dummy_img, str(onnx_path),
                          input_names=["image"], output_names=["logits"], **kwargs)

    size_mb = onnx_path.stat().st_size / 1e6
    print(f"  [OK]   {dataset}/{run_id} -> {onnx_path.name}  ({size_mb:.1f} MB)")
    return onnx_path


def convert_detection(det_dir: Path, imgsz: int = 640) -> Path | None:
    """Exports outputs/detection/checkpoints/best.pt via Ultralytics' own exporter
    (not torch.onnx.export directly: YOLO models need its export-time graph surgery,
    e.g. folding the detection head, that raw tracing wouldn't reproduce)."""
    ckpt = det_dir / "checkpoints" / "best.pt"
    if not ckpt.exists():
        print(f"  [SKIP] detection: checkpoint not found: {ckpt}")
        return None

    from ultralytics import YOLO
    model = YOLO(str(ckpt))
    onnx_path = Path(model.export(format="onnx", imgsz=imgsz))

    size_mb = onnx_path.stat().st_size / 1e6
    print(f"  [OK]   detection -> {onnx_path.name}  ({size_mb:.1f} MB)")
    return onnx_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export trained classifier checkpoints, and/or the YOLOv8 detector, to ONNX."
    )
    parser.add_argument("--models",   nargs="+", default=["all"], choices=ALL_MODELS + ["all"])
    parser.add_argument("--datasets", nargs="+", default=["all"], choices=ALL_DATASETS + ["all"])
    parser.add_argument("--aug",      nargs="+", default=["none"], choices=ALL_AUGS + ["all"])
    parser.add_argument("--detection", action="store_true",
                        help="Also export outputs/detection/checkpoints/best.pt")
    parser.add_argument("--skip-classifiers", action="store_true",
                        help="Skip the --models/--datasets/--aug classifier export entirely")
    args = parser.parse_args()

    models   = ALL_MODELS   if "all" in args.models   else args.models
    datasets = ALL_DATASETS if "all" in args.datasets else args.datasets
    augs     = ALL_AUGS     if "all" in args.aug      else args.aug

    base_cfg  = load_config()
    noseg_dir = Path(base_cfg.paths.outputs) / "ablation_noseg"
    det_dir   = Path(base_cfg.paths.outputs) / "detection"

    converted, skipped = 0, 0

    if not args.skip_classifiers:
        print(f"\n  Converting {len(models)} models x {len(datasets)} datasets x {len(augs)} aug modes\n")
        for dataset in datasets:
            for aug_mode in augs:
                for model_name in models:
                    try:
                        result = convert_one(dataset, model_name, aug_mode, noseg_dir)
                        converted += result is not None
                        skipped   += result is None
                    except Exception as exc:
                        print(f"  [ERROR] {dataset}/{model_name}_{aug_mode}: {exc}")
                        skipped += 1

    if args.detection:
        print("\n  Converting detector\n")
        try:
            result = convert_detection(det_dir)
            converted += result is not None
            skipped   += result is None
        except Exception as exc:
            print(f"  [ERROR] detection: {exc}")
            skipped += 1

    print(f"\n  Done: {converted} exported, {skipped} skipped\n")


if __name__ == "__main__":
    main()
