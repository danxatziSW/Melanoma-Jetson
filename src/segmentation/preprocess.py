from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from src.segmentation.infer import segment_image
from src.segmentation.crop import apply_mask_crop

_SEG_NORM = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


def cache_seg_crops(
    df: pd.DataFrame,
    seg_model: nn.Module,
    device: torch.device,
    cache_dir: Path,
    image_col: str = "image_path",
) -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    crop_paths = [
        str(cache_dir / f"{Path(str(p)).stem}.jpg")
        for p in df[image_col]
    ]

    pending = [
        (i, row)
        for i, (_, row) in enumerate(df.iterrows())
        if not Path(crop_paths[i]).exists()
    ]

    if pending:
        print(f"  Pre-caching {len(pending)}/{len(df)} seg crops → {cache_dir}")
        seg_model.eval()
        with torch.no_grad():
            for i, row in tqdm(pending, desc="  seg-crop", unit="img"):
                _save_one_crop(str(row[image_col]), crop_paths[i], seg_model)
    else:
        print(f"  All {len(df)} seg crops already cached.")

    out = df.copy()
    out["cached_crop_path"] = crop_paths
    return out


def _save_one_crop(image_path: str, crop_path: str, seg_model: nn.Module) -> None:
    image = cv2.imread(image_path)
    if image is None:
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    aug = _SEG_NORM(image=image)
    tensor = aug["image"].unsqueeze(0)   # segment_image handles device placement

    mask = segment_image(seg_model, tensor)   # (256, 256) uint8 0/255

    h, w = image.shape[:2]
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    crop = apply_mask_crop(image, mask)
    cv2.imwrite(crop_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
