from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.io import resolve_dataset_paths


class SegmentationDataset(Dataset):
    def __init__(self, csv_path: str | Path, transform: Optional[Callable] = None,
                 data_root: str | Path | None = None):
        self.df = pd.read_csv(csv_path)
        if data_root is not None:
            self.df = resolve_dataset_paths(self.df, data_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = cv2.imread(str(row["image_path"]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(row["mask_path"]), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
            # albumentations drops the channel dim from masks
            if isinstance(mask, torch.Tensor) and mask.dim() == 2:
                mask = mask.unsqueeze(0)
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask


class MelanomaDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        transform: Optional[Callable] = None,
        use_metadata: bool = False,
        seg_model=None,
        seg_transform: Optional[Callable] = None,
        data_root: str | Path | None = None,
    ):
        self.df = pd.read_csv(csv_path)
        if data_root is not None:
            self.df = resolve_dataset_paths(self.df, data_root)
        self.transform = transform
        self.use_metadata = use_metadata
        self.seg_model = seg_model
        self.seg_transform = seg_transform

        if use_metadata:
            self._encode_metadata()

    def _encode_metadata(self) -> None:
        df = self.df
        age_col = "age_approx" if "age_approx" in df.columns else None
        self.age = (
            (df[age_col].fillna(df[age_col].median()) / 100.0).values.astype(np.float32)
            if age_col
            else np.zeros(len(df), dtype=np.float32)
        )
        sex_map = {"male": 1.0, "female": 0.0}
        sex_col = "sex" if "sex" in df.columns else None
        self.sex = (
            df[sex_col].map(sex_map).fillna(0.5).values.astype(np.float32)
            if sex_col
            else np.full(len(df), 0.5, dtype=np.float32)
        )
        site_col = "anatom_site_general_challenge" if "anatom_site_general_challenge" in df.columns else None
        site_categories = ["head/neck", "upper extremity", "lower extremity", "torso", "palms/soles", "oral/genital"]
        if site_col:
            site_series = df[site_col].fillna("unknown")
            self.site_ohe = np.zeros((len(df), len(site_categories)), dtype=np.float32)
            for i, cat in enumerate(site_categories):
                self.site_ohe[:, i] = (site_series == cat).astype(np.float32)
        else:
            self.site_ohe = np.zeros((len(df), len(site_categories)), dtype=np.float32)

    def _get_metadata_tensor(self, idx: int) -> torch.Tensor:
        meta = np.concatenate([[self.age[idx], self.sex[idx]], self.site_ohe[idx]])
        return torch.from_numpy(meta)

    def _apply_seg_crop(self, image: np.ndarray) -> np.ndarray:
        from src.segmentation.infer import segment_image
        from src.segmentation.crop import apply_mask_crop
        import torch

        h, w = image.shape[:2]
        img_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        if self.seg_transform is not None:
            aug = self.seg_transform(image=image)
            img_tensor = aug["image"]

        mask = segment_image(self.seg_model, img_tensor.unsqueeze(0))
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return apply_mask_crop(image, mask)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = cv2.imread(str(row["image_path"]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = int(row["label_int"])

        # prefer cached crop, then seg model, then mask file
        cached = row.get("cached_crop_path", None)
        if cached and pd.notna(cached) and Path(str(cached)).exists():
            loaded = cv2.imread(str(cached))
            if loaded is not None:
                image = cv2.cvtColor(loaded, cv2.COLOR_BGR2RGB)
        elif self.seg_model is not None:
            image = self._apply_seg_crop(image)
        elif "has_mask" in row and row["has_mask"] and "mask_path" in row and pd.notna(row["mask_path"]):
            from src.segmentation.crop import apply_mask_crop_from_path
            image = apply_mask_crop_from_path(image, str(row["mask_path"]))

        if self.transform is not None:
            augmented = self.transform(image=image)
            image_tensor = augmented["image"]
        else:
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        if self.use_metadata:
            return image_tensor, self._get_metadata_tensor(idx), label
        return image_tensor, label
