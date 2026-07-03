import types
from pathlib import Path

import torch
import segmentation_models_pytorch as smp
import torch.nn as nn

from src.models.ddrnet import DDRNet23Slim

_DDR_ENCODERS = {"ddrnet23s"}


def build_unet(config: types.SimpleNamespace) -> nn.Module:
    encoder_name = getattr(config, "encoder_name", "efficientnet-b2")
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=getattr(config, "encoder_weights", "imagenet"),
        in_channels=getattr(config, "in_channels", 3),
        classes=getattr(config, "classes", 1),
        activation=None,
    )


def get_segmentation_model(encoder_name: str, config: types.SimpleNamespace) -> nn.Module:
    if encoder_name in _DDR_ENCODERS:
        num_classes = getattr(config, "classes", 1)
        return DDRNet23Slim(num_classes=num_classes, deep_supervision=True)
    cfg_copy = types.SimpleNamespace(**vars(config))
    cfg_copy.encoder_name = encoder_name
    return build_unet(cfg_copy)


def load_seg_checkpoint(
    ckpt_path: str | Path,
    encoder_name: str,
    device: torch.device,
) -> nn.Module:
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Segmentation checkpoint not found: {ckpt_path}")
    model = get_segmentation_model(encoder_name, types.SimpleNamespace(classes=1))
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    model.to(device)
    return model
