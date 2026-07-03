import types

import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


def build_resnet(config: types.SimpleNamespace, num_classes: int) -> nn.Module:
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if getattr(config, "pretrained", True) else None)
    dropout = getattr(config, "dropout", 0.3)
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(2048, num_classes),
    )
    return model
