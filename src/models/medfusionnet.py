import timm
import types

import torch
import torch.nn as nn


class MedFusionNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        metadata_dim: int = 8,
        fusion_hidden: int = 256,
        dropout: float = 0.4,
        pretrained: bool = True,
    ):
        super().__init__()

        # EfficientNet-B0 image branch, global-pooled to 1280-d
        self.image_branch = timm.create_model(
            "efficientnet_b0", pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        img_feat_dim = self.image_branch.num_features  # 1280

        self.meta_branch = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
        )

        fused_dim = img_feat_dim + 128
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(fusion_hidden, num_classes),
        )

    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        img_feat = self.image_branch(image)
        meta_feat = self.meta_branch(metadata)
        fused = torch.cat([img_feat, meta_feat], dim=1)
        return self.fusion_head(fused)


def build_medfusionnet(config: types.SimpleNamespace, num_classes: int) -> MedFusionNet:
    return MedFusionNet(
        num_classes=num_classes,
        metadata_dim=getattr(config, "metadata_dim", 8),
        fusion_hidden=getattr(config, "fusion_hidden", 256),
        dropout=getattr(config, "dropout", 0.4),
        pretrained=getattr(config, "pretrained", True),
    )
