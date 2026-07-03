import types

import torch.nn as nn

from src.models.resnet import build_resnet
from src.models.efficientnet import build_efficientnet
from src.models.mobilenetv3 import build_mobilenetv3
from src.models.convnext_se import build_convnext_se
from src.models.medfusionnet import build_medfusionnet
from src.models.yolov8_cls import build_yolov8_cls

_REGISTRY = {
    "resnet50": build_resnet,
    "efficientnet_b2": build_efficientnet,
    "mobilenetv3_large": build_mobilenetv3,
    "convnext_tiny_se": build_convnext_se,
    "medfusionnet": build_medfusionnet,
    "yolov8_cls": build_yolov8_cls,
}


def build_model(model_name: str, config: types.SimpleNamespace, num_classes: int) -> nn.Module:
    if model_name not in _REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[model_name](config, num_classes)


def uses_metadata(model_name: str) -> bool:
    return model_name == "medfusionnet"
