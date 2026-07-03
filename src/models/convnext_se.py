import timm
import types

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale


class ConvNeXtStageWithSE(nn.Module):
    def __init__(self, stage_block: nn.Module, channels: int, reduction: int = 16):
        super().__init__()
        self.block = stage_block
        self.se = SEBlock(channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.block(x))


# output channels per convnext_tiny stage
_CONVNEXT_TINY_CHANNELS = [96, 192, 384, 768]


def build_convnext_se(config: types.SimpleNamespace, num_classes: int) -> nn.Module:
    pretrained = getattr(config, "pretrained", True)
    dropout = getattr(config, "dropout", 0.3)
    reduction = getattr(config, "se_reduction", 16)

    model = timm.create_model("convnext_tiny", pretrained=pretrained, num_classes=0)

    stages = model.stages
    for i, (stage, ch) in enumerate(zip(stages, _CONVNEXT_TINY_CHANNELS)):
        stages[i] = ConvNeXtStageWithSE(stage, channels=ch, reduction=reduction)

    head_dim = _CONVNEXT_TINY_CHANNELS[-1]
    model.head = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.LayerNorm(head_dim),
        nn.Dropout(p=dropout),
        nn.Linear(head_dim, num_classes),
    )
    return model
