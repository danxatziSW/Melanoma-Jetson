import timm
import types

import torch.nn as nn


def build_efficientnet(config: types.SimpleNamespace, num_classes: int) -> nn.Module:
    backbone = getattr(config, "backbone", "efficientnet_b2")
    pretrained = getattr(config, "pretrained", True)
    dropout = getattr(config, "dropout", 0.3)
    model = timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout,
    )
    return model
