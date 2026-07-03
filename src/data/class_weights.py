import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def compute_class_weights(labels: list[int] | np.ndarray, num_classes: int) -> torch.Tensor:
    labels = np.array(labels)
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def build_weighted_sampler(labels: list[int] | np.ndarray, num_classes: int) -> WeightedRandomSampler:
    labels = np.array(labels)
    class_weights = compute_class_weights(labels, num_classes)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(labels),
        replacement=True,
    )
