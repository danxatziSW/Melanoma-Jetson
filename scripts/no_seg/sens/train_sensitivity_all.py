from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from src.models.registry import build_model, uses_metadata
from src.utils.config import load_config
from src.utils.io import append_excel_row, resolve_dataset_paths

ALL_MODELS   = ["resnet50", "efficientnet_b2", "mobilenetv3_large", "convnext_tiny_se", "medfusionnet", "yolov8_cls"]
ALL_DATASETS = ["ham10000", "isic2019", "isic2020"]

AUG_MODE        = "none"
SENS_SUFFIX     = "none_sens"
FINETUNE_LR     = 2e-5
FINETUNE_EPOCHS = 30
FOCAL_GAMMA     = 3.0
MEL_WEIGHT_MULT = 2.0
EARLY_STOP_PAT  = 10
BATCH_MULTIPLIER = 2   # doubles batch size vs config — safe with AMP on 5070 (12 GB)

_MEAN      = (0.485, 0.456, 0.406)
_STD       = (0.229, 0.224, 0.225)
_SITE_CATS = [
    "head/neck", "upper extremity", "lower extremity",
    "torso", "palms/soles", "oral/genital",
]


class SensDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: A.Compose, with_meta: bool = False):
        self.df        = df.reset_index(drop=True)
        self.transform = transform
        self.with_meta = with_meta
        if with_meta:
            self._encode_metadata()

    def _encode_metadata(self) -> None:
        df      = self.df
        age_col = "age_approx" if "age_approx" in df.columns else None
        self.age = (
            (df[age_col].fillna(df[age_col].median()) / 100.0).values.astype(np.float32)
            if age_col else np.zeros(len(df), dtype=np.float32)
        )
        sex_col = "sex" if "sex" in df.columns else None
        self.sex = (
            df[sex_col].map({"male": 1.0, "female": 0.0}).fillna(0.5).values.astype(np.float32)
            if sex_col else np.full(len(df), 0.5, dtype=np.float32)
        )
        site_col = "anatom_site_general_challenge" if "anatom_site_general_challenge" in df.columns else None
        self.site_ohe = np.zeros((len(df), len(_SITE_CATS)), dtype=np.float32)
        if site_col:
            site_s = df[site_col].fillna("unknown")
            for i, cat in enumerate(_SITE_CATS):
                self.site_ohe[:, i] = (site_s == cat).astype(np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        image = cv2.imread(str(row["image_path"]))
        image = (
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if image is not None
            else np.zeros((224, 224, 3), dtype=np.uint8)
        )
        img_t = self.transform(image=image)["image"]
        label = int(row["binary_label"])
        if self.with_meta:
            meta = np.concatenate([[self.age[idx], self.sex[idx]], self.site_ohe[idx]])
            return img_t, torch.from_numpy(meta), label
        return img_t, label


def _train_transform(size: int) -> A.Compose:
    return A.Compose([
        A.RandomResizedCrop(size=(size, size), scale=(0.7, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])


def _val_transform(size: int) -> A.Compose:
    return A.Compose([
        A.Resize(height=int(size * 1.1), width=int(size * 1.1)),
        A.CenterCrop(height=size, width=size),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])


def _load_split(splits_dir: Path, split: str, dataset: str, melanoma_root: Path) -> pd.DataFrame:
    df = pd.read_csv(splits_dir / f"{split}.csv")
    df = df[df["dataset_source"] == dataset].copy()
    df["binary_label"] = (df["label_str"] == "mel").astype(int)
    df = resolve_dataset_paths(df, melanoma_root)
    return df.reset_index(drop=True)


def _weighted_sampler(labels: np.ndarray, mel_mult: float) -> WeightedRandomSampler:
    n_mel = max(int((labels == 1).sum()), 1)
    n_non = max(int((labels == 0).sum()), 1)
    w = np.where(labels == 1, mel_mult / n_mel, 1.0 / n_non)
    return WeightedRandomSampler(w.tolist(), num_samples=len(labels), replacement=True)


def _focal_loss(logits: torch.Tensor, targets: torch.Tensor,
                mel_weight: float, gamma: float) -> torch.Tensor:
    w   = torch.tensor([1.0, mel_weight], dtype=torch.float32, device=logits.device)
    ce  = nn.functional.cross_entropy(logits, targets, weight=w, reduction="none")
    p_t = torch.exp(-ce)
    return ((1.0 - p_t) ** gamma * ce).mean()


def _f2(tp: int, fp: int, fn: int) -> float:
    prec = tp / max(tp + fp, 1)
    sens = tp / max(tp + fn, 1)
    return (5 * prec * sens) / max(4 * prec + sens, 1e-9)


def _val_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    best_thr, best_f2 = 0.50, -1.0
    for thr in np.round(np.arange(0.25, 0.86, 0.01), 2):
        preds = (probs >= thr).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        f2 = _f2(tp, fp, fn)
        if f2 > best_f2:
            best_f2, best_thr = f2, thr
    preds = (probs >= best_thr).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1   = (2 * prec * sens) / max(prec + sens, 1e-9)
    try:
        auc = float(roc_auc_score(labels, probs))
    except Exception:
        auc = float("nan")
    return dict(
        best_threshold=best_thr, val_f2=round(best_f2, 4),
        val_f1=round(f1, 4),    val_sensitivity=round(sens, 4),
        val_specificity=round(spec, 4), val_auc=round(auc, 4),
    )


def train_one(model_name: str, dataset: str,
              base_cfg, device: torch.device) -> bool:
    """returns True if completed successfully, False if skipped."""
    cfg       = load_config(model_name)
    inp_size  = getattr(cfg, "input_size", 224)
    nw        = getattr(cfg, "num_workers", 4)
    meta      = uses_metadata(model_name)
    batch_size = cfg.batch_size * BATCH_MULTIPLIER

    splits_dir   = Path(base_cfg.paths.data_splits)
    melanoma_root = Path(base_cfg.paths.melanoma_data)
    ablation_dir = Path(base_cfg.paths.outputs) / "ablation_noseg"
    out_dir      = Path(base_cfg.paths.outputs) / "ablation_noseg" / dataset / f"{model_name}_{SENS_SUFFIX}"
    ckpt_dir     = out_dir / "checkpoints"
    log_xlsx     = out_dir / f"{model_name}_{SENS_SUFFIX}_log.xlsx"
    best_ckpt    = ckpt_dir / f"{model_name}_{SENS_SUFFIX}.pt"

    src_ckpt = ablation_dir / dataset / f"{model_name}_{AUG_MODE}" / "checkpoints" / f"{model_name}_{AUG_MODE}.pt"
    if not src_ckpt.exists():
        print(f"  [SKIP] {model_name}/{dataset} — checkpoint not found: {src_ckpt}")
        return False

    if best_ckpt.exists():
        print(f"  [SKIP] already fine-tuned: {best_ckpt.name}")
        return True

    ckpt_dir.mkdir(parents=True, exist_ok=True)

    sep = "=" * 68
    print(f"\n{sep}")
    print(f"  Model   : {model_name}  |  Dataset : {dataset.upper()}  |  Meta: {meta}")
    print(f"  From    : {src_ckpt.name}")
    print(f"  To      : {best_ckpt}")
    print(f"{sep}\n")

    train_df = _load_split(splits_dir, "cls_train", dataset, melanoma_root)
    val_df   = _load_split(splits_dir, "cls_val",   dataset, melanoma_root)
    n_mel_tr = int((train_df["binary_label"] == 1).sum())
    n_non_tr = int((train_df["binary_label"] == 0).sum())
    mel_weight = MEL_WEIGHT_MULT * (n_non_tr / max(n_mel_tr, 1))

    print(f"  Train : {len(train_df)}  mel={n_mel_tr}  non-mel={n_non_tr}  "
          f"ratio={n_non_tr/max(n_mel_tr,1):.1f}:1  mel_weight={mel_weight:.1f}")
    print(f"  Val   : {len(val_df)}  mel={int((val_df['binary_label']==1).sum())}\n")

    train_ds     = SensDataset(train_df, _train_transform(inp_size), with_meta=meta)
    val_ds       = SensDataset(val_df,   _val_transform(inp_size),   with_meta=meta)
    sampler      = _weighted_sampler(train_df["binary_label"].values, MEL_WEIGHT_MULT)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=nw, pin_memory=True, prefetch_factor=2 if nw > 0 else None,
                              persistent_workers=(nw > 0))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=nw, pin_memory=True, prefetch_factor=2 if nw > 0 else None,
                              persistent_workers=(nw > 0))

    model = build_model(model_name, cfg, num_classes=2)
    model.load_state_dict(torch.load(src_ckpt, map_location=device, weights_only=True))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=FINETUNE_LR,
                                  weight_decay=cfg.optimizer.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=FINETUNE_EPOCHS, eta_min=1e-7)

    best_f2, no_improve = -1.0, 0
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    print(f"  {'Ep':>3}  {'TrainLoss':>10}  {'ValF2':>7}  {'ValF1':>7}  "
          f"{'Sens':>7}  {'Spec':>7}  {'AUC':>7}  {'Thr':>5}")

    for epoch in range(1, FINETUNE_EPOCHS + 1):
        model.train()
        t_loss = 0.0
        bar = tqdm(train_loader, desc=f"  ep{epoch:02d} train", leave=False,
                   unit="batch", dynamic_ncols=True, file=sys.stdout)
        for batch in bar:
            if meta:
                imgs, mdata, labels = batch
                imgs, mdata, labels = imgs.to(device), mdata.to(device), labels.to(device)
            else:
                imgs, labels = batch
                imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(imgs, mdata) if meta else model(imgs)
                loss   = _focal_loss(logits, labels, mel_weight, FOCAL_GAMMA)

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            t_loss += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss = t_loss / len(train_loader)

        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            for batch in tqdm(val_loader, desc=f"  ep{epoch:02d} val  ",
                              leave=False, unit="batch",
                              dynamic_ncols=True, file=sys.stdout):
                if meta:
                    imgs, mdata, labels = batch
                    probs = torch.softmax(model(imgs.to(device), mdata.to(device)), dim=1)[:, 1].cpu().numpy()
                else:
                    imgs, labels = batch
                    probs = torch.softmax(model(imgs.to(device)), dim=1)[:, 1].cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.numpy())

        m = _val_metrics(np.array(all_probs), np.array(all_labels))
        scheduler.step()

        improved = m["val_f2"] > best_f2
        if improved:
            best_f2    = m["val_f2"]
            no_improve = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            no_improve += 1

        star = " ★" if improved else ""
        print(f"  {epoch:>3}  {train_loss:>10.4f}  "
              f"{m['val_f2']:>7.4f}  {m['val_f1']:>7.4f}  "
              f"{m['val_sensitivity']:>7.4f}  {m['val_specificity']:>7.4f}  "
              f"{m['val_auc']:>7.4f}  {m['best_threshold']:>5.2f}{star}")

        append_excel_row(log_xlsx, "TrainLog", {"epoch": epoch,
                         "train_loss": round(train_loss, 6), **m})

        if no_improve >= EARLY_STOP_PAT:
            print(f"\n  Early stop — no F2 improvement for {EARLY_STOP_PAT} epochs.")
            break

    print(f"\n  Best val F2 : {best_f2:.4f}  →  {best_ckpt}\n")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return True


def main() -> None:
    base_cfg = load_config()
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True   # auto-tune kernels for fixed input size
    gpu_label = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"

    total = len(ALL_MODELS) * len(ALL_DATASETS)
    print(f"\n{'='*68}")
    print(f"  Sensitivity fine-tuning — all no-seg models")
    print(f"  Models   : {ALL_MODELS}")
    print(f"  Datasets : {ALL_DATASETS}")
    print(f"  Runs     : {total}  (skips missing/already-done checkpoints)")
    print(f"  Device   : {gpu_label}")
    print(f"  Gamma    : {FOCAL_GAMMA}   MelMult: {MEL_WEIGHT_MULT}x   LR: {FINETUNE_LR}")
    print(f"  Monitor  : F2 (β=2)   EarlyStop: {EARLY_STOP_PAT} epochs")
    print(f"{'='*68}\n")

    done, skipped = 0, 0
    idx = 0
    for dataset in ALL_DATASETS:
        for model_name in ALL_MODELS:
            idx += 1
            print(f"[{idx}/{total}] {model_name} / {dataset}")
            ok = train_one(model_name, dataset, base_cfg, device)
            if ok:
                done += 1
            else:
                skipped += 1

    print(f"\n{'='*68}")
    print(f"  Training complete — {done} models fine-tuned, {skipped} skipped.")
    print(f"  Checkpoints: outputs/ablation_noseg/<dataset>/<model>_none_sens/")
    print(f"\n  Running ensemble evaluation ...")
    print(f"{'='*68}\n")

    project_root  = Path(__file__).resolve().parents[3]
    eval_script   = Path(__file__).resolve().parent / "evaluate_2models_mean_sens.py"
    tflite_script = project_root / "scripts" / "convert_to_tflite.py"

    print("\n  Step 1 / 2 — TFLite conversion ...")
    tflite_models = [m for m in ALL_MODELS if m != "convnext_tiny_se"]
    result = subprocess.run(
        [sys.executable, str(tflite_script),
         "--sens", "--skip-existing",
         "--models"] + tflite_models,
        cwd=str(project_root),
    )
    if result.returncode != 0:
        print(f"  [WARNING] TFLite conversion exited with code {result.returncode}.")
        print(f"  Run manually: python scripts/convert_to_tflite.py --sens --skip-existing")

    print("\n  Step 2 / 2 — Ensemble evaluation ...")
    result = subprocess.run(
        [sys.executable, str(eval_script)],
        cwd=str(project_root),
    )
    if result.returncode != 0:
        print(f"\n  [WARNING] Evaluation script exited with code {result.returncode}.")
        print(f"  Run manually: python scripts/no_seg/evaluate_2models_mean_sens.py")


if __name__ == "__main__":
    main()
