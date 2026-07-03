import types

import torch.nn as nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


def build_mobilenetv3(config: types.SimpleNamespace, num_classes: int) -> nn.Module:
    pretrained = getattr(config, "pretrained", True)
    dropout = getattr(config, "dropout", 0.2)
    model = mobilenet_v3_large(
        weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
    )
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model
