import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal = (1.0 - p_t) ** self.gamma * ce_loss
        if self.alpha is not None:
            focal = self.alpha[targets] * focal
        return focal.mean()


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets, weight=self.weight)


def build_loss(loss_name: str, class_weights: torch.Tensor | None = None, gamma: float = 2.0) -> nn.Module:
    if loss_name == "focal":
        return FocalLoss(alpha=class_weights, gamma=gamma)
    if loss_name == "weighted_ce":
        return WeightedCrossEntropyLoss(weight=class_weights)
    raise ValueError(f"Unknown loss '{loss_name}'. Choose 'focal' or 'weighted_ce'.")
