import numpy as np
import torch
import torch.nn as nn


def segment_image(model: nn.Module, image_tensor: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        device = next(model.parameters()).device
        image_tensor = image_tensor.to(device)
        logits = model(image_tensor)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()

    binary = (prob > threshold).astype(np.uint8) * 255
    return binary
