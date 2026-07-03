from pathlib import Path

import cv2
import numpy as np


def apply_mask_crop(image: np.ndarray, mask: np.ndarray, padding_frac: float = 0.05) -> np.ndarray:
    h, w = image.shape[:2]
    binary = (mask > 127).astype(np.uint8)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n_labels <= 1:
        return image  # no lesion found — return full image

    # label 0 is background; find largest foreground component
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    component_mask = (labels == largest_label).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    component_mask = cv2.morphologyEx(component_mask, cv2.MORPH_CLOSE, kernel)

    x, y, bw, bh, _ = stats[largest_label]
    pad_x = int(bw * padding_frac)
    pad_y = int(bh * padding_frac)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x)
    y2 = min(h, y + bh + pad_y)

    cropped_img = image[y1:y2, x1:x2].copy()
    cropped_mask = component_mask[y1:y2, x1:x2]

    # background dimmed to 30%, smooth transition via Gaussian blur
    blurred = cv2.GaussianBlur(cropped_mask.astype(np.float32) / 255.0, (21, 21), 0)
    focus = 0.3 + 0.7 * blurred
    cropped_img = np.clip(cropped_img.astype(np.float32) * focus[:, :, np.newaxis], 0, 255).astype(np.uint8)

    return cropped_img


def apply_mask_crop_from_path(image: np.ndarray, mask_path: str, padding_frac: float = 0.05) -> np.ndarray:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return image
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return apply_mask_crop(image, mask, padding_frac)
