from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.data.augmentation import build_seg_train_transform, build_seg_val_transform
from src.data.dataset import SegmentationDataset
from src.segmentation.model import get_segmentation_model
from src.utils.io import append_excel_row
from src.utils.reproducibility import seed_everything

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _save_sample_grid(
    model: torch.nn.Module,
    val_loader: DataLoader,
    out_path: Path,
    device: torch.device,
    n: int = 4,
) -> None:
    try:
        import cv2
    except ImportError:
        return

    model.eval()
    dataset = val_loader.dataset
    indices = torch.randperm(len(dataset))[:n].tolist()
    sample_loader = DataLoader(
        Subset(dataset, indices), batch_size=n, shuffle=False,
        num_workers=val_loader.num_workers,
    )
    imgs_saved, masks_saved, preds_saved = [], [], []
    with torch.no_grad():
        for imgs, masks in sample_loader:
            logits = model(imgs.to(device))
            preds  = (torch.sigmoid(logits) > 0.5).float().cpu()
            for i in range(imgs.size(0)):
                img = imgs[i].permute(1, 2, 0).numpy()
                img = np.clip(img * _IMAGENET_STD + _IMAGENET_MEAN, 0, 1)
                img = (img * 255).astype(np.uint8)
                imgs_saved.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                m = (masks[i].squeeze().numpy() * 255).astype(np.uint8)
                p = (preds[i].squeeze().numpy() * 255).astype(np.uint8)
                masks_saved.append(cv2.cvtColor(m, cv2.COLOR_GRAY2BGR))
                preds_saved.append(cv2.cvtColor(p, cv2.COLOR_GRAY2BGR))

    rows = []
    for img, gt, pred in zip(imgs_saved, masks_saved, preds_saved):
        h = img.shape[0]
        divider = np.full((h, 4, 3), 128, dtype=np.uint8)
        rows.append(np.concatenate([img, divider, gt, divider, pred], axis=1))
    grid = np.concatenate(rows, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> float:
    pred = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred * target).sum()
    return ((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)).item()


def iou_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> float:
    pred = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return ((intersection + smooth) / (union + smooth)).item()


def train_segmentation(config: types.SimpleNamespace, encoder_name: str | None = None) -> None:
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = encoder_name or getattr(config, "encoder_name", "efficientnet-b2")

    gpu_label = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"\n  Encoder : {enc}", flush=True)
    print(f"  Device  : {gpu_label}", flush=True)

    input_size = getattr(config, "input_size", 256)
    train_ds = SegmentationDataset(
        Path(config.paths.data_splits) / "seg_train.csv",
        transform=build_seg_train_transform(input_size),
    )
    val_ds = SegmentationDataset(
        Path(config.paths.data_splits) / "seg_val.csv",
        transform=build_seg_val_transform(input_size),
    )
    print(f"  Dataset : {len(train_ds)} train / {len(val_ds)} val\n", flush=True)

    num_workers = getattr(config, "num_workers", 0)
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    model = get_segmentation_model(enc, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay,
    )

    warmup_epochs = getattr(config, "warmup_epochs", 0)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config.epochs - warmup_epochs, 1), eta_min=config.scheduler.eta_min,
    )
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs],
        )
    else:
        scheduler = cosine

    dice_loss   = smp.losses.DiceLoss(mode="binary")
    lovasz_loss = smp.losses.LovaszLoss(mode="binary")
    dw = getattr(config, "dice_weight", 0.7)
    bw = getattr(config, "bce_weight", 0.3)

    out_dir = Path(config.paths.outputs) / "segmentation"
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_xlsx = out_dir / "metrics" / "segmentation_metrics.xlsx"
    metrics_xlsx.parent.mkdir(parents=True, exist_ok=True)

    best_dice = 0.0
    patience   = getattr(config, "early_stopping_patience", 8)
    no_improve = 0

    epoch_bar = tqdm(
        range(1, config.epochs + 1),
        desc="Epochs",
        unit="ep",
        dynamic_ncols=True,
        file=sys.stdout,
    )

    for epoch in epoch_bar:
        model.train()
        train_loss = 0.0
        batch_bar = tqdm(
            train_loader,
            desc="  train",
            leave=False,
            unit="batch",
            dynamic_ncols=True,
            file=sys.stdout,
        )
        for imgs, masks in batch_bar:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            if isinstance(logits, tuple):
                logits = logits[0]   # DDRNet returns (main, aux4, aux3) during training
            loss = dw * dice_loss(logits, masks) + bw * lovasz_loss(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss /= len(train_loader)

        model.eval()
        val_loss = val_dice = val_iou = 0.0
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc="  val  ", leave=False, unit="batch",
                                    dynamic_ncols=True, file=sys.stdout):
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                loss = dw * dice_loss(logits, masks) + bw * lovasz_loss(logits, masks)
                val_loss += loss.item()
                val_dice += dice_coefficient(logits, masks)
                val_iou  += iou_score(logits, masks)
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_iou  /= len(val_loader)

        scheduler.step()

        append_excel_row(
            metrics_xlsx, "TrainLog",
            {
                "epoch": epoch, "encoder": enc,
                "train_loss": round(train_loss, 6),
                "val_loss":   round(val_loss,   6),
                "val_dice":   round(val_dice,   6),
                "val_iou":    round(val_iou,    6),
            },
        )

        if val_dice > best_dice:
            best_dice  = val_dice
            no_improve = 0
            ckpt_path  = ckpt_dir / f"best_unet_{enc.replace('-', '_')}.pt"
            torch.save(model.state_dict(), ckpt_path)
            tqdm.write(f"  ep {epoch:02d}  dice={val_dice:.4f}  iou={val_iou:.4f}  ★ saved")
        else:
            no_improve += 1
            tqdm.write(f"  ep {epoch:02d}  dice={val_dice:.4f}  iou={val_iou:.4f}  ({no_improve}/{patience})")

        epoch_bar.set_postfix(
            tr_loss=f"{train_loss:.4f}",
            dice=f"{val_dice:.4f}",
            iou=f"{val_iou:.4f}",
            best=f"{best_dice:.4f}",
        )
        if no_improve >= patience:
            tqdm.write(f"  Early stop — no improvement for {patience} epochs")
            break

    safe_enc = enc.replace("-", "_").replace("/", "_")
    sample_path = out_dir / "samples" / f"sample_{safe_enc}.png"
    _save_sample_grid(model, val_loader, sample_path, device, n=4)
    print(f"  Sample  : {sample_path}", flush=True)
    print(f"\n  Done.  Best val_dice = {best_dice:.4f}", flush=True)
