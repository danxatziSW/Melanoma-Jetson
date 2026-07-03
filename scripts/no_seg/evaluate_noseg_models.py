"""Evaluate all no-segmentation ablation checkpoints cross-dataset.

Loads raw images directly (no seg crop), matches the structure of
evaluate_models.py but points at ablation_noseg/ checkpoints.

Output: outputs/ablation/evaluation_nosegmentation.xlsx
  Sheet "Summary"          — one row per (train_ds, eval_ds, model, aug)
  Sheet "Threshold_Sweep"  — all threshold steps
  Sheets "HAM10000", "ISIC2019", "ISIC2020" — pivot by aug mode
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.models.registry import build_model, uses_metadata
from src.utils.config import load_config
from src.utils.io import resolve_dataset_paths, write_excel_sheet

ALL_DATASETS = ["ham10000", "isic2019", "isic2020"]
ALL_MODELS   = [
    "resnet50", "efficientnet_b2", "mobilenetv3_large",
    "convnext_tiny_se", "medfusionnet", "yolov8_cls",
]
ALL_AUGS     = ["none", "light"]

_MEAN      = (0.485, 0.456, 0.406)
_STD       = (0.229, 0.224, 0.225)
_SITE_CATS = [
    "head/neck", "upper extremity", "lower extremity",
    "torso", "palms/soles", "oral/genital",
]
THRESHOLDS = np.round(np.arange(0.35, 0.66, 0.01), 2)


class NoSegEvalDataset(Dataset):
    """Loads raw images (no seg crop) with a clean eval transform."""

    def __init__(self, df: pd.DataFrame, input_size: int, with_meta: bool = False):
        self.df        = df.reset_index(drop=True)
        self.with_meta = with_meta
        self.transform = A.Compose([
            A.Resize(height=int(input_size * 1.1), width=int(input_size * 1.1)),
            A.CenterCrop(height=input_size, width=input_size),
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ])
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


def _load_test_df(splits_dir: Path, dataset_source: str, melanoma_root: Path) -> pd.DataFrame:
    df = pd.read_csv(splits_dir / "cls_test.csv")
    df = df[df["dataset_source"] == dataset_source].copy()
    if df.empty:
        raise ValueError(f"No test rows for dataset_source='{dataset_source}'.")
    df["binary_label"] = (df["label_str"] == "mel").astype(int)
    df = resolve_dataset_paths(df, melanoma_root)
    missing = df["image_path"].apply(lambda p: not Path(str(p)).exists()).sum()
    if missing:
        print(f"  WARNING: {missing} test images not found on disk.")
    return df


def _build_matching_model(model_name: str, state_dict: dict, config, num_classes: int):
    """Build the architecture that matches the saved state_dict keys.

    yolov8_cls can be saved as _YOLOClsWrapper ("inner.*") or as the
    MobileNetV3-RW timm fallback ("conv_stem.*") depending on whether
    ultralytics was installed at training time.
    """
    if model_name != "yolov8_cls":
        return build_model(model_name, config, num_classes=num_classes)
    first_key = next(iter(state_dict))
    if first_key.startswith(("conv_stem", "blocks", "classifier")):
        import timm
        return timm.create_model(
            "mobilenetv3_rw", pretrained=False,
            num_classes=num_classes, drop_rate=getattr(config, "dropout", 0.2),
        )
    return build_model(model_name, config, num_classes=num_classes)


def _load_checkpoint(
    train_dataset: str,
    model_name: str,
    aug_mode: str,
    device: torch.device,
    config,
    noseg_dir: Path,
) -> nn.Module | None:
    run_id    = f"{model_name}_{aug_mode}"
    ckpt_file = noseg_dir / train_dataset / run_id / "checkpoints" / f"{run_id}.pt"
    if not ckpt_file.exists():
        print(f"  [SKIP] checkpoint not found: {ckpt_file.relative_to(noseg_dir.parent)}")
        return None
    state_dict = torch.load(ckpt_file, map_location=device)
    model = _build_matching_model(model_name, state_dict, config, num_classes=2)
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def _run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    with_meta: bool,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="    infer", leave=False,
                          unit="batch", dynamic_ncols=True, file=sys.stdout):
            if with_meta:
                imgs, mdata, labels = batch
                logits = model(imgs.to(device), mdata.to(device))
            else:
                imgs, labels = batch
                logits = model(imgs.to(device))
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    return np.array(all_probs), np.array(all_labels)


def _metrics_at_threshold(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    accuracy    = (tp + tn) / max(tp + tn + fp + fn, 1)
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision   = tp / max(tp + fp, 1)
    f1          = (2 * precision * sensitivity) / max(precision + sensitivity, 1e-9)
    return dict(
        threshold=round(float(threshold), 2),
        accuracy=round(accuracy, 4), sensitivity=round(sensitivity, 4),
        specificity=round(specificity, 4), precision=round(precision, 4),
        f1=round(f1, 4), tp=tp, tn=tn, fp=fp, fn=fn,
    )


def evaluate_one(
    train_dataset: str,
    eval_dataset: str,
    model_name: str,
    aug_mode: str,
    test_df: pd.DataFrame,
    device: torch.device,
    config,
    model: nn.Module,
) -> tuple[dict, list[dict]]:
    meta     = uses_metadata(model_name)
    inp_size = getattr(config, "input_size", 224)
    nw       = 4

    ds     = NoSegEvalDataset(test_df, inp_size, with_meta=meta)
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=False,
                        num_workers=nw, pin_memory=True, persistent_workers=True)

    probs, labels = _run_inference(model, loader, device, meta)

    try:
        auc = float(roc_auc_score(labels, probs))
    except Exception:
        auc = float("nan")

    sweep_rows = []
    for thr in THRESHOLDS:
        row = _metrics_at_threshold(probs, labels, thr)
        row.update(train_dataset=train_dataset, eval_dataset=eval_dataset,
                   model=model_name, aug_mode=aug_mode, auc=round(auc, 4))
        sweep_rows.append(row)

    best = max(sweep_rows, key=lambda r: r["f1"])
    summary = dict(
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        model=model_name, aug_mode=aug_mode, auc=round(auc, 4),
        best_threshold=best["threshold"],
        accuracy=best["accuracy"], sensitivity=best["sensitivity"],
        specificity=best["specificity"], precision=best["precision"],
        f1=best["f1"], tp=best["tp"], tn=best["tn"], fp=best["fp"], fn=best["fn"],
        test_samples=len(test_df),
        mel_count=int(labels.sum()),
        non_mel_count=int((labels == 0).sum()),
    )

    print(
        f"      eval={eval_dataset:<12}  AUC={auc:.4f}  thr={best['threshold']:.2f}  "
        f"F1={best['f1']:.4f}  sens={best['sensitivity']:.4f}  spec={best['specificity']:.4f}"
    )
    return summary, sweep_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate no-segmentation ablation checkpoints on all test sets."
    )
    parser.add_argument("--datasets", nargs="+", default=["all"],
                        choices=ALL_DATASETS + ["all"], metavar="DATASET")
    parser.add_argument("--models", nargs="+", default=["all"],
                        choices=ALL_MODELS + ["all"], metavar="MODEL")
    parser.add_argument("--aug", nargs="+", default=["all"],
                        choices=ALL_AUGS + ["all"], metavar="AUG")
    args = parser.parse_args()

    datasets  = ALL_DATASETS if "all" in args.datasets else args.datasets
    models    = ALL_MODELS   if "all" in args.models   else args.models
    aug_modes = ALL_AUGS     if "all" in args.aug      else args.aug

    n_models     = len(datasets) * len(models) * len(aug_modes)
    n_eval_pairs = n_models * len(datasets)
    print(f"\n  Checkpoints     : {n_models}  (train_ds × model × aug)")
    print(f"  Eval pairs      : {n_eval_pairs}  (each checkpoint on every test set)")
    print(f"  Threshold sweep : {THRESHOLDS[0]:.2f} → {THRESHOLDS[-1]:.2f}  ({len(THRESHOLDS)} steps)\n")

    base_cfg   = load_config()
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits_dir = Path(base_cfg.paths.data_splits)
    melanoma_root = Path(base_cfg.paths.melanoma_data)
    noseg_dir  = Path(base_cfg.paths.outputs) / "ablation_noseg"
    out_xlsx   = Path(base_cfg.paths.outputs) / "ablation" / "evaluation_nosegmentation.xlsx"

    gpu_label = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"  Device : {gpu_label}")
    print(f"  Noseg checkpoints : {noseg_dir}")
    print(f"  Output            : {out_xlsx}\n")

    # pre-load all test DataFrames
    test_dfs: dict[str, pd.DataFrame] = {}
    for ds in datasets:
        try:
            df = _load_test_df(splits_dir, ds, melanoma_root)
            test_dfs[ds] = df
            print(
                f"  {ds.upper():<12}  test={len(df)}  "
                f"mel={int(df['binary_label'].sum())}  "
                f"non-mel={int((df['binary_label']==0).sum())}"
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"  [SKIP] {ds}: {exc}")
    print()

    all_summaries: list[dict] = []
    all_sweep:     list[dict] = []
    ckpt_idx = 0

    for train_ds in datasets:
        for aug_mode in aug_modes:
            for model_name in models:
                ckpt_idx += 1
                run_id = f"{model_name}_{aug_mode}"
                print(f"  [{ckpt_idx}/{n_models}]  train={train_ds} / {run_id}")

                config = load_config(model_name)
                try:
                    model = _load_checkpoint(train_ds, model_name, aug_mode,
                                             device, config, noseg_dir)
                except Exception as exc:
                    import traceback
                    print(f"  [ERROR] loading checkpoint: {exc}")
                    traceback.print_exc()
                    continue

                if model is None:
                    continue

                for eval_ds in datasets:
                    if eval_ds not in test_dfs:
                        continue
                    try:
                        summary, sweep = evaluate_one(
                            train_ds, eval_ds, model_name, aug_mode,
                            test_dfs[eval_ds], device, config, model,
                        )
                        all_summaries.append(summary)
                        all_sweep.extend(sweep)
                    except Exception as exc:
                        import traceback
                        print(f"  [ERROR] {train_ds}/{run_id} → {eval_ds}: {exc}")
                        traceback.print_exc()

    if not all_summaries:
        print("\n  No results to save.\n")
        return

    summary_cols = [
        "train_dataset", "eval_dataset", "model", "aug_mode", "auc",
        "best_threshold", "accuracy", "sensitivity", "specificity",
        "precision", "f1", "tp", "tn", "fp", "fn",
        "test_samples", "mel_count", "non_mel_count",
    ]
    sweep_cols = [
        "train_dataset", "eval_dataset", "model", "aug_mode", "auc",
        "threshold", "accuracy", "sensitivity", "specificity",
        "precision", "f1", "tp", "tn", "fp", "fn",
    ]

    df_summary = pd.DataFrame(all_summaries)[summary_cols]
    df_sweep   = pd.DataFrame(all_sweep)[sweep_cols]

    write_excel_sheet(out_xlsx, "Summary",         df_summary)
    write_excel_sheet(out_xlsx, "Threshold_Sweep", df_sweep)

    # per eval-dataset pivot sheets
    for eval_ds in datasets:
        ev_df = df_summary[df_summary["eval_dataset"] == eval_ds].copy()
        if ev_df.empty:
            continue
        pivot = ev_df.pivot_table(
            index   = ["train_dataset", "model"],
            columns = "aug_mode",
            values  = ["auc", "f1", "sensitivity", "specificity", "best_threshold"],
            aggfunc = "first",
        )
        pivot.columns = [f"{val}_{aug}" for val, aug in pivot.columns]
        pivot = pivot.reset_index()
        write_excel_sheet(out_xlsx, eval_ds.upper(), pivot)

    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  Results saved : {out_xlsx}")
    print(f"  Sheets        : Summary | Threshold_Sweep | {' | '.join(ds.upper() for ds in datasets)}")
    print(f"{sep}\n")

    display_cols = [
        "train_dataset", "eval_dataset", "model", "aug_mode",
        "auc", "best_threshold", "f1", "sensitivity", "specificity",
    ]
    print(
        df_summary[display_cols]
        .sort_values(["eval_dataset", "train_dataset", "auc"], ascending=[True, True, False])
        .to_string(index=False)
    )
    print()


if __name__ == "__main__":
    main()
