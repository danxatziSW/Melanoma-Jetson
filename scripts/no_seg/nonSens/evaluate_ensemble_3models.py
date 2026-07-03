# tries all 3-model combos (majority vote) and ranks them by avg f1 (no-seg version)
from __future__ import annotations

import argparse
import sys
from itertools import combinations
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
THRESHOLDS = np.round(np.arange(0.35, 0.66, 0.01), 2)   # 31 steps


class EvalDataset(Dataset):
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
    return df


def _load_checkpoint(
    train_dataset: str, model_name: str, aug_mode: str,
    device: torch.device, config, ablation_dir: Path,
) -> nn.Module | None:
    run_id    = f"{model_name}_{aug_mode}"
    ckpt_file = ablation_dir / train_dataset / run_id / "checkpoints" / f"{run_id}.pt"
    if not ckpt_file.exists():
        return None
    model = build_model(model_name, config, num_classes=2)
    model.load_state_dict(torch.load(ckpt_file, map_location=device))
    model.to(device)
    return model


def _run_inference(
    model: nn.Module, loader: DataLoader,
    device: torch.device, with_meta: bool,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="      infer", leave=False,
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
        accuracy=round(accuracy, 4), sensitivity=round(sensitivity, 4),
        specificity=round(specificity, 4), precision=round(precision, 4),
        f1=round(f1, 4), tp=tp, tn=tn, fp=fp, fn=fn,
    )


def _metrics_from_preds(preds: np.ndarray, labels: np.ndarray) -> dict:
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
        accuracy=round(accuracy, 4), sensitivity=round(sensitivity, 4),
        specificity=round(specificity, 4), precision=round(precision, 4),
        f1=round(f1, 4), tp=tp, tn=tn, fp=fp, fn=fn,
    )


def _global_threshold(ds_probs: dict[str, tuple[np.ndarray, np.ndarray]]) -> float:
    best_thr, best_avg = THRESHOLDS[0], -1.0
    for thr in THRESHOLDS:
        avg = float(np.mean([
            _metrics_at_threshold(p, l, thr)["f1"]
            for p, l in ds_probs.values()
        ]))
        if avg > best_avg:
            best_avg, best_thr = avg, thr
    return float(best_thr)


def run_group(
    aug_mode: str,
    train_datasets: list[str],
    eval_datasets: list[str],
    test_dfs: dict[str, pd.DataFrame],
    device: torch.device,
    ablation_dir: Path,
    models: list[str],
) -> tuple[list[dict], list[dict]]:
    results        = []
    single_results = []
    checkpoint_probs: dict[tuple[str, str], dict[str, tuple[np.ndarray, np.ndarray, float]]] = {}

    total = len(train_datasets) * len(models)
    idx   = 0
    for train_ds in train_datasets:
        for model_name in models:
            idx += 1
            config = load_config(model_name)
            model  = _load_checkpoint(train_ds, model_name, aug_mode, device, config, ablation_dir)
            if model is None:
                print(f"    [{idx}/{total}] [SKIP] {model_name}/{train_ds} not found")
                continue

            key  = (model_name, train_ds)
            meta = uses_metadata(model_name)
            inp_size = getattr(config, "input_size", 224)
            nw       = getattr(config, "num_workers", 0)
            checkpoint_probs[key] = {}
            print(f"    [{idx}/{total}] {model_name}/{train_ds} ...", end="", flush=True)

            for eval_ds in eval_datasets:
                if eval_ds not in test_dfs:
                    continue
                ds     = EvalDataset(test_dfs[eval_ds], inp_size, with_meta=meta)
                loader = DataLoader(ds, batch_size=config.batch_size, shuffle=False,
                                    num_workers=nw, pin_memory=(nw > 0))
                probs, labels = _run_inference(model, loader, device, meta)
                try:
                    auc = float(roc_auc_score(labels, probs))
                except Exception:
                    auc = float("nan")
                checkpoint_probs[key][eval_ds] = (probs, labels, auc)

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            aucs = ", ".join(
                f"{checkpoint_probs[key][ds][2]:.3f}"
                for ds in eval_datasets if ds in checkpoint_probs[key]
            )
            print(f" done  (AUC: {aucs})")

    if len(checkpoint_probs) < 3:
        print(f"    [SKIP] fewer than 3 checkpoints loaded for aug={aug_mode}")
        return results, single_results

    # global threshold per checkpoint
    checkpoint_thresholds: dict[tuple[str, str], float] = {}
    for key, ds_dict in checkpoint_probs.items():
        probs_only = {ds: (p, l) for ds, (p, l, _) in ds_dict.items()}
        checkpoint_thresholds[key] = _global_threshold(probs_only)

    # single model baselines
    for key, ds_dict in checkpoint_probs.items():
        model_name, train_ds = key
        thr = checkpoint_thresholds[key]
        row = dict(
            aug_mode         = aug_mode,
            model            = model_name,
            train_dataset    = train_ds,
            global_threshold = thr,
        )
        per_ds_f1s = []
        for eval_ds, (probs, labels, auc) in ds_dict.items():
            m = _metrics_at_threshold(probs, labels, thr)
            per_ds_f1s.append(m["f1"])
            row[f"auc_{eval_ds}"]         = round(auc, 4)
            row[f"f1_{eval_ds}"]          = m["f1"]
            row[f"sensitivity_{eval_ds}"] = m["sensitivity"]
            row[f"specificity_{eval_ds}"] = m["specificity"]
            row[f"accuracy_{eval_ds}"]    = m["accuracy"]
            row[f"tp_{eval_ds}"]          = m["tp"]
            row[f"tn_{eval_ds}"]          = m["tn"]
            row[f"fp_{eval_ds}"]          = m["fp"]
            row[f"fn_{eval_ds}"]          = m["fn"]
        row["avg_f1"] = round(float(np.mean(per_ds_f1s)), 4) if per_ds_f1s else 0.0
        row["avg_sensitivity"] = round(float(np.mean(
            [row.get(f"sensitivity_{ds}", 0.0) for ds in eval_datasets
             if f"sensitivity_{ds}" in row]
        )), 4)
        row["avg_specificity"] = round(float(np.mean(
            [row.get(f"specificity_{ds}", 0.0) for ds in eval_datasets
             if f"specificity_{ds}" in row]
        )), 4)
        row["avg_accuracy"] = round(float(np.mean(
            [row.get(f"accuracy_{ds}", 0.0) for ds in eval_datasets
             if f"accuracy_{ds}" in row]
        )), 4)
        single_results.append(row)

    # all triplet combos
    available = list(checkpoint_probs.keys())
    n = len(available)
    n_combos = len(list(combinations(available, 3)))
    print(f"    Evaluating {n_combos} combos from {n} checkpoints ...")

    for combo in combinations(available, 3):
        k1, k2, k3    = combo
        m1, ds1 = k1
        m2, ds2 = k2
        m3, ds3 = k3
        combo_name = f"{m1}_{ds1}+{m2}_{ds2}+{m3}_{ds3}"
        thr1 = checkpoint_thresholds[k1]
        thr2 = checkpoint_thresholds[k2]
        thr3 = checkpoint_thresholds[k3]

        summary = dict(
            aug_mode    = aug_mode,
            combo_name  = combo_name,
            model_1     = m1,  train_1 = ds1,
            model_2     = m2,  train_2 = ds2,
            model_3     = m3,  train_3 = ds3,
            threshold_1 = thr1,
            threshold_2 = thr2,
            threshold_3 = thr3,
        )

        per_ds_f1s = []
        for eval_ds in eval_datasets:
            if not all(eval_ds in checkpoint_probs[k] for k in combo):
                continue
            p1, labels, _ = checkpoint_probs[k1][eval_ds]
            p2, _,      _ = checkpoint_probs[k2][eval_ds]
            p3, _,      _ = checkpoint_probs[k3][eval_ds]

            votes = (
                (p1 >= thr1).astype(int) +
                (p2 >= thr2).astype(int) +
                (p3 >= thr3).astype(int)
            )
            ensemble_preds = (votes >= 2).astype(int)

            avg_probs = (p1 + p2 + p3) / 3.0
            try:
                auc = float(roc_auc_score(labels, avg_probs))
            except Exception:
                auc = float("nan")

            m = _metrics_from_preds(ensemble_preds, labels)
            per_ds_f1s.append(m["f1"])
            summary[f"auc_{eval_ds}"]         = round(auc, 4)
            summary[f"f1_{eval_ds}"]          = m["f1"]
            summary[f"sensitivity_{eval_ds}"] = m["sensitivity"]
            summary[f"specificity_{eval_ds}"] = m["specificity"]
            summary[f"accuracy_{eval_ds}"]    = m["accuracy"]
            summary[f"tp_{eval_ds}"]          = m["tp"]
            summary[f"tn_{eval_ds}"]          = m["tn"]
            summary[f"fp_{eval_ds}"]          = m["fp"]
            summary[f"fn_{eval_ds}"]          = m["fn"]

        summary["avg_f1"] = round(float(np.mean(per_ds_f1s)), 4) if per_ds_f1s else 0.0
        summary["avg_sensitivity"] = round(float(np.mean(
            [summary.get(f"sensitivity_{ds}", 0.0) for ds in eval_datasets
             if f"sensitivity_{ds}" in summary]
        )), 4)
        summary["avg_specificity"] = round(float(np.mean(
            [summary.get(f"specificity_{ds}", 0.0) for ds in eval_datasets
             if f"specificity_{ds}" in summary]
        )), 4)
        summary["avg_accuracy"] = round(float(np.mean(
            [summary.get(f"accuracy_{ds}", 0.0) for ds in eval_datasets
             if f"accuracy_{ds}" in summary]
        )), 4)
        results.append(summary)

    return results, single_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate all 3-model majority-vote ensembles (no-seg) across all train datasets."
    )
    parser.add_argument("--datasets", nargs="+", default=["all"],
                        choices=ALL_DATASETS + ["all"], metavar="DATASET")
    parser.add_argument("--models",   nargs="+", default=["all"],
                        choices=ALL_MODELS + ["all"],   metavar="MODEL")
    parser.add_argument("--aug",      nargs="+", default=["none"],
                        choices=ALL_AUGS + ["all"],     metavar="AUG")
    args = parser.parse_args()

    datasets  = ALL_DATASETS if "all" in args.datasets else args.datasets
    models    = ALL_MODELS   if "all" in args.models   else args.models
    aug_modes = ALL_AUGS     if "all" in args.aug      else args.aug

    n_checkpoints = len(models) * len(datasets)
    n_combos      = len(list(combinations(range(n_checkpoints), 3)))
    print(f"\n  Checkpoints per aug_mode : {n_checkpoints}  ({len(models)} models x {len(datasets)} train datasets)")
    print(f"  Combos per aug_mode      : {n_combos}  (C({n_checkpoints},3))")
    print(f"  Aug modes                : {len(aug_modes)}")
    print(f"  Total combo rows         : {len(aug_modes) * n_combos}\n")

    base_cfg     = load_config()
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits_dir   = Path(base_cfg.paths.data_splits)
    melanoma_root = Path(base_cfg.paths.melanoma_data)
    ablation_dir = Path(base_cfg.paths.outputs) / "ablation"
    noseg_dir    = Path(base_cfg.paths.outputs) / "ablation_noseg"
    noseg_dir.mkdir(parents=True, exist_ok=True)
    eval_xlsx    = noseg_dir / "evaluation_3models_ensemble.xlsx"

    gpu_label = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"  Device : {gpu_label}\n")

    # preload test sets
    test_dfs: dict[str, pd.DataFrame] = {}
    for ds in datasets:
        try:
            df = _load_test_df(splits_dir, ds, melanoma_root)
            test_dfs[ds] = df
            print(
                f"  {ds.upper():<12} test={len(df)}  "
                f"mel={int(df['binary_label'].sum())}  "
                f"non-mel={int((df['binary_label']==0).sum())}"
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"  [SKIP] {ds}: {exc}")
    print()

    all_results:  list[dict] = []
    all_singles:  list[dict] = []

    for aug_idx, aug_mode in enumerate(aug_modes, 1):
        print(f"  [{aug_idx}/{len(aug_modes)}]  aug={aug_mode}")
        try:
            rows, singles = run_group(
                aug_mode, datasets, datasets, test_dfs,
                device, ablation_dir, models,
            )
            all_results.extend(rows)
            all_singles.extend(singles)
        except Exception as exc:
            import traceback
            print(f"  [ERROR] aug={aug_mode}: {exc}")
            traceback.print_exc()
        print()

    if not all_results:
        print("\n  No results to save.\n")
        return

    df = pd.DataFrame(all_results)

    id_cols = [
        "aug_mode", "combo_name",
        "model_1", "train_1", "model_2", "train_2", "model_3", "train_3",
        "threshold_1", "threshold_2", "threshold_3",
        "avg_f1", "avg_sensitivity", "avg_specificity", "avg_accuracy",
    ]
    extra_cols = [c for c in df.columns if c not in id_cols]
    df = df[id_cols + extra_cols]

    write_excel_sheet(eval_xlsx, "Summary", df.sort_values("avg_f1", ascending=False))

    if all_singles:
        df_singles = pd.DataFrame(all_singles)
        single_id  = ["aug_mode", "model", "train_dataset", "global_threshold",
                      "avg_f1", "avg_sensitivity", "avg_specificity", "avg_accuracy"]
        extra_s    = [c for c in df_singles.columns if c not in single_id]
        df_singles = df_singles[single_id + extra_s].sort_values("avg_f1", ascending=False)
        write_excel_sheet(eval_xlsx, "Single_Models", df_singles)

    for eval_ds in datasets:
        f1_col = f"f1_{eval_ds}"
        if f1_col not in df.columns:
            continue
        ds_cols = ["aug_mode", "combo_name",
                   "model_1", "train_1", "model_2", "train_2", "model_3", "train_3",
                   "threshold_1", "threshold_2", "threshold_3"]
        ds_cols += [c for c in df.columns if c.endswith(f"_{eval_ds}")]
        ev_df = df[ds_cols].sort_values(f1_col, ascending=False)
        write_excel_sheet(eval_xlsx, eval_ds.upper(), ev_df)

    best = (
        df.groupby("combo_name", as_index=False)
        .agg(avg_f1=("avg_f1", "mean"),
             avg_accuracy=("avg_accuracy", "mean"),
             avg_sensitivity=("avg_sensitivity", "mean"),
             avg_specificity=("avg_specificity", "mean"))
        .sort_values("avg_f1", ascending=False)
    )
    combo_info = df.drop_duplicates("combo_name")[
        ["combo_name", "model_1", "train_1", "model_2", "train_2", "model_3", "train_3",
         "threshold_1", "threshold_2", "threshold_3"]
    ]
    best = best.merge(combo_info, on="combo_name", how="left")
    best_cols = ["combo_name", "model_1", "train_1", "model_2", "train_2", "model_3", "train_3",
                 "threshold_1", "threshold_2", "threshold_3",
                 "avg_f1", "avg_accuracy", "avg_sensitivity", "avg_specificity"]
    write_excel_sheet(eval_xlsx, "Best_Combos", best[best_cols])

    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  Results saved : {eval_xlsx}")
    print(f"  Sheets        : Summary | Single_Models | Best_Combos | {' | '.join(ds.upper() for ds in datasets)}")
    print(f"  Total combos  : {len(df)}")
    print(f"{sep}\n")

    print("  Top 10 combos (avg across all aug_modes):\n")
    print(best.head(10)[["combo_name", "avg_f1", "avg_accuracy", "avg_sensitivity", "avg_specificity"]].to_string(index=False))
    print()


if __name__ == "__main__":
    main()
