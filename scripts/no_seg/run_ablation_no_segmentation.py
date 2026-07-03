"""ablation study: 6 models × 3 datasets — no segmentation."""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from src.models.registry import build_model, uses_metadata
from src.training.callbacks import CSVEpochLogger, EarlyStopping, ModelCheckpoint
from src.utils.config import load_config
from src.utils.io import append_excel_row, resolve_dataset_paths, write_excel_sheet
from src.utils.reproducibility import seed_everything

ALL_DATASETS = ["ham10000", "isic2019", "isic2020"]
ALL_MODELS = [
    "resnet50", "efficientnet_b2", "mobilenetv3_large",
    "convnext_tiny_se", "medfusionnet", "yolov8_cls",
]
AUG_MODE = "none"

_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)
_SITE_CATS = [
    "head/neck", "upper extremity", "lower extremity",
    "torso", "palms/soles", "oral/genital",
]


def _train_transform(input_size: int) -> A.Compose:
    return A.Compose([
        A.RandomResizedCrop(size=(input_size, input_size), scale=(0.7, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])


def _val_transform(input_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(height=int(input_size * 1.1), width=int(input_size * 1.1)),
        A.CenterCrop(height=input_size, width=input_size),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])


class NoSegDataset(Dataset):
    """mel vs non-mel dataset — reads raw full-resolution images, no segmentation."""

    def __init__(self, df: pd.DataFrame, transform: A.Compose, with_meta: bool = False):
        self.df        = df.reset_index(drop=True)
        self.transform = transform
        self.with_meta = with_meta
        if with_meta:
            self._encode_metadata()

    def _encode_metadata(self) -> None:
        df = self.df
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
        if image is None:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img_tensor = self.transform(image=image)["image"]
        label      = int(row["binary_label"])

        if self.with_meta:
            meta = np.concatenate([[self.age[idx], self.sex[idx]], self.site_ohe[idx]])
            return img_tensor, torch.from_numpy(meta), label
        return img_tensor, label


def load_dataset_binary(splits_dir: Path, dataset_source: str, melanoma_root: Path) -> pd.DataFrame:
    frames = []
    for name in ("cls_train.csv", "cls_val.csv", "cls_test.csv"):
        p = splits_dir / name
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError(f"No split CSVs found in {splits_dir}")
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["image_path"])
    df = df[df["dataset_source"] == dataset_source].copy()
    if df.empty:
        raise ValueError(
            f"No rows for dataset_source='{dataset_source}'. "
            "data_splits/*.csv needs to be rebuilt for this dataset first (see the README)."
        )
    df["binary_label"] = (df["label_str"] == "mel").astype(int)
    df = resolve_dataset_paths(df, melanoma_root)
    return df


def intra_dataset_split(
    df: pd.DataFrame,
    val_fraction: float = 0.20,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, val_df = train_test_split(
        df, test_size=val_fraction, stratify=df["binary_label"], random_state=seed,
    )
    mel     = val_df[val_df["binary_label"] == 1]
    non_mel = val_df[val_df["binary_label"] == 0]
    n = min(len(mel), len(non_mel))
    val_balanced = pd.concat([
        mel.sample(n=n, random_state=seed),
        non_mel.sample(n=n, random_state=seed),
    ]).reset_index(drop=True)
    return train_df.reset_index(drop=True), val_balanced


def _build_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    n_mel = max(int((labels == 1).sum()), 1)
    n_non = max(int((labels == 0).sum()), 1)
    w = np.where(labels == 1, 1.0 / n_mel, 1.0 / n_non)
    return WeightedRandomSampler(w.tolist(), num_samples=len(labels), replacement=True)


def run_one(
    dataset_name: str,
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    device: torch.device,
    config,
    out_root: Path,
    num_workers: int,
) -> dict:
    run_id   = f"{model_name}_{AUG_MODE}"
    out_dir  = out_root / dataset_name / run_id
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = out_dir / f"{run_id}_metrics.xlsx"

    sep = "=" * 68
    print(f"\n{sep}")
    print(f"  Dataset  : {dataset_name.upper()}")
    print(f"  Model    : {model_name}")
    print(f"  Epochs   : {config.epochs}")
    print(f"  Workers  : {num_workers}")
    print(f"  Output   : {out_dir}")
    print(sep + "\n")

    meta     = uses_metadata(model_name)
    inp_size = getattr(config, "input_size", 224)

    train_ds = NoSegDataset(train_df, _train_transform(inp_size), with_meta=meta)
    val_ds   = NoSegDataset(val_df,   _val_transform(inp_size),   with_meta=meta)

    sampler      = _build_sampler(train_df["binary_label"].values)
    persistent   = num_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=persistent,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=persistent,
        persistent_workers=persistent,
    )

    cls_model = build_model(model_name, config, num_classes=2)
    cls_model.to(device)

    optimizer = torch.optim.AdamW(
        cls_model.parameters(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.scheduler.eta_min,
    )
    loss_fn    = nn.CrossEntropyLoss()
    logger     = CSVEpochLogger(xlsx_path)
    checkpoint = ModelCheckpoint(ckpt_dir, monitor="val_auc", mode="max", filename=run_id)
    early_stop = EarlyStopping(
        patience=getattr(config, "early_stopping_patience", 10),
        monitor="val_auc", mode="max",
    )

    gpu_label = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"  Device   : {gpu_label}")
    print(
        f"  Train    : {len(train_df)} images  "
        f"(mel={int(train_df['binary_label'].sum())}, "
        f"non-mel={int((train_df['binary_label']==0).sum())})"
    )
    print(
        f"  Val      : {len(val_df)} images  "
        f"(mel={int(val_df['binary_label'].sum())}, "
        f"non-mel={int((val_df['binary_label']==0).sum())})"
    )
    print(f"  Batches  : train={len(train_loader)}  val={len(val_loader)}\n")

    best_auc, best_probs, best_labels = 0.0, None, None

    epoch_bar = tqdm(
        range(1, config.epochs + 1),
        desc="Epochs", unit="ep", dynamic_ncols=True, file=sys.stdout,
    )
    for epoch in epoch_bar:
        cls_model.train()
        t_loss, t_preds, t_labels = 0.0, [], []
        bar = tqdm(train_loader, desc="  train", leave=False,
                   unit="batch", dynamic_ncols=True, file=sys.stdout)
        for batch in bar:
            if meta:
                imgs, mdata, labels = batch
                imgs, mdata, labels = imgs.to(device), mdata.to(device), labels.to(device)
            else:
                imgs, labels = batch
                mdata = None
                imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = cls_model(imgs, mdata) if meta else cls_model(imgs)
            loss   = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            t_loss += loss.item()
            t_preds.extend(logits.argmax(dim=1).cpu().numpy())
            t_labels.extend(labels.cpu().numpy())
            bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = t_loss / len(train_loader)
        train_acc  = float(accuracy_score(t_labels, t_preds))

        cls_model.eval()
        v_loss, v_preds, v_labels, v_probs = 0.0, [], [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="  val  ", leave=False,
                              unit="batch", dynamic_ncols=True, file=sys.stdout):
                if meta:
                    imgs, mdata, labels = batch
                    imgs, mdata, labels = imgs.to(device), mdata.to(device), labels.to(device)
                else:
                    imgs, labels = batch
                    mdata = None
                    imgs, labels = imgs.to(device), labels.to(device)

                logits = cls_model(imgs, mdata) if meta else cls_model(imgs)
                loss   = loss_fn(logits, labels)
                v_loss += loss.item()
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                v_probs.extend(probs)
                v_preds.extend((probs >= 0.5).astype(int))
                v_labels.extend(labels.cpu().numpy())

        val_loss = v_loss / len(val_loader)
        val_acc  = float(accuracy_score(v_labels, v_preds))
        try:
            val_auc = float(roc_auc_score(v_labels, v_probs))
        except Exception:
            val_auc = float("nan")

        if val_auc > best_auc:
            best_auc    = val_auc
            best_probs  = np.array(v_probs)
            best_labels = np.array(v_labels)

        scheduler.step()
        metrics = dict(
            train_loss=train_loss, val_loss=val_loss,
            train_acc=train_acc,   val_acc=val_acc,
            val_auc=val_auc,
        )
        logger.on_epoch_end(epoch, **metrics)
        improved = checkpoint.on_epoch_end(cls_model, **metrics)

        epoch_bar.set_postfix(
            loss=f"{train_loss:.4f}/{val_loss:.4f}",
            acc=f"{val_acc:.3f}",
            auc=f"{val_auc:.4f}",
        )
        star = " ★ saved" if improved else ""
        tqdm.write(
            f"  ep {epoch:02d}/{config.epochs}  "
            f"loss={train_loss:.4f}/{val_loss:.4f}  "
            f"acc={train_acc:.3f}/{val_acc:.3f}  "
            f"auc={val_auc:.4f}{star}"
        )

        if early_stop.on_epoch_end(**metrics):
            tqdm.write(f"  Early stop at epoch {epoch}")
            break

    if best_probs is not None and len(best_probs) > 0:
        prob_df = pd.DataFrame({
            "image_path":      val_df["image_path"].values,
            "true_label":      best_labels,
            "mel_probability": best_probs,
            "predicted_label": (best_probs >= 0.5).astype(int),
        })
        write_excel_sheet(xlsx_path, "Probabilities", prob_df)

    summary = {
        "dataset":       dataset_name,
        "model":         model_name,
        "aug_mode":      AUG_MODE,
        "segmentation":  "none",
        "epochs_run":    config.epochs,
        "best_val_auc":  round(best_auc, 4),
        "val_samples":   len(val_df),
        "mel_count":     int(val_df["binary_label"].sum()),
        "non_mel_count": int((val_df["binary_label"] == 0).sum()),
        "output_dir":    str(out_dir),
    }
    print(f"\n  Best val AUC : {best_auc:.4f}")
    print(f"  Results      : {xlsx_path}\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "No-segmentation ablation: 6 models × 3 datasets.\n"
            "Images are pre-resized to 256px and cached — no seg crops.\n"
            "Summary written to outputs/ablation/ablation_noSegmentation.xlsx."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--datasets", nargs="+", default=["all"], choices=ALL_DATASETS + ["all"],
        metavar="DATASET",
    )
    parser.add_argument(
        "--models", nargs="+", default=["all"], choices=ALL_MODELS + ["all"],
        metavar="MODEL",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="DataLoader worker processes (default: 4). Set 0 to disable.",
    )
    args = parser.parse_args()

    datasets  = ALL_DATASETS if "all" in args.datasets else args.datasets
    models    = ALL_MODELS   if "all" in args.models   else args.models

    total = len(datasets) * len(models)
    print(
        f"\n  Grid  : {len(datasets)} datasets "
        f"× {len(models)} models "
        f"= {total} experiments  [NO SEGMENTATION]"
    )
    print(f"  Epochs per run : {args.epochs}")
    print(f"  Workers        : {args.num_workers}")
    print(f"  Skip existing  : {args.skip_existing}\n")

    base_cfg = load_config()
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_root     = Path(base_cfg.paths.outputs) / "ablation_noseg"
    out_root.mkdir(parents=True, exist_ok=True)
    summary_xlsx = Path(base_cfg.paths.outputs) / "ablation" / "ablation_noSegmentation.xlsx"
    splits_dir   = Path(base_cfg.paths.data_splits)
    melanoma_root = Path(base_cfg.paths.melanoma_data)

    seed_everything(base_cfg.seed)

    print("  Loading dataset splits ...")
    dataset_splits: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for ds in datasets:
        try:
            df = load_dataset_binary(splits_dir, ds, melanoma_root)
            train_df, val_df = intra_dataset_split(df, val_fraction=0.20, seed=base_cfg.seed)
            dataset_splits[ds] = (train_df, val_df)
            print(
                f"    {ds.upper():<12} total={len(df)}  "
                f"train={len(train_df)}  val={len(val_df)}"
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"    [SKIP] {ds}: {exc}")

    run_idx, all_summaries = 0, []
    ds_summaries: dict[str, list[dict]] = {ds: [] for ds in datasets}

    for ds in datasets:
        if ds not in dataset_splits:
            continue
        train_df, val_df = dataset_splits[ds]

        for model_name in models:
            run_idx += 1
            run_id    = f"{model_name}_{AUG_MODE}"
            ckpt_file = out_root / ds / run_id / "checkpoints" / f"{run_id}.pt"

            print(f"\n  [{run_idx}/{total}]  {ds} / {model_name}")

            if args.skip_existing and ckpt_file.exists():
                print(f"  [SKIP] checkpoint already exists: {ckpt_file}")
                continue

            config        = load_config(model_name)
            config.epochs = args.epochs

            try:
                summary = run_one(
                    ds, model_name,
                    train_df, val_df,
                    device, config,
                    out_root,
                    args.num_workers,
                )
                all_summaries.append(summary)
                ds_summaries[ds].append(summary)
                append_excel_row(summary_xlsx, "Results", summary)

            except Exception as exc:
                print(f"\n  [ERROR] {ds}/{run_id}: {exc}")
                traceback.print_exc()

        if ds_summaries[ds]:
            ds_xlsx = Path(base_cfg.paths.outputs) / "ablation" / f"ablation_noSegmentation_{ds}.xlsx"
            write_excel_sheet(ds_xlsx, "Results", pd.DataFrame(ds_summaries[ds]))
            print(f"\n  Per-dataset summary written → {ds_xlsx}")

    if all_summaries:
        sep = "=" * 68
        print(f"\n{sep}")
        print(f"  Completed : {len(all_summaries)} / {total} experiments")
        print(f"  Combined  : {summary_xlsx}")
        for ds in datasets:
            if ds_summaries.get(ds):
                ds_xlsx = Path(base_cfg.paths.outputs) / "ablation" / f"ablation_noSegmentation_{ds}.xlsx"
                print(f"  {ds.upper():<12}: {ds_xlsx}")
        print(f"{sep}\n")
        df_out = pd.DataFrame(all_summaries)[
            ["dataset", "model", "aug_mode", "segmentation", "epochs_run", "best_val_auc", "val_samples"]
        ]
        print(df_out.to_string(index=False))
        print()
    else:
        print("\n  No experiments completed.\n")


if __name__ == "__main__":
    main()
