from __future__ import annotations

import sys
from itertools import combinations
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.models.registry import build_model, uses_metadata
from src.utils.config import load_config
from src.utils.io import resolve_dataset_paths, write_excel_sheet

ALL_DATASETS = ["ham10000", "isic2019", "isic2020"]
ALL_MODELS   = [
    "resnet50", "efficientnet_b2", "mobilenetv3_large", "convnext_tiny_se",
    "medfusionnet", "yolov8_cls",
]

AUG_MODE  = "none_sens"           # sensitivity fine-tuned checkpoints
BETA      = 2.0                   # F-beta for threshold optimisation (F2)
THR_RANGE = np.round(np.arange(0.20, 0.86, 0.01), 2)   # 66 steps

_MEAN      = (0.485, 0.456, 0.406)
_STD       = (0.229, 0.224, 0.225)
_SITE_CATS = [
    "head/neck", "upper extremity", "lower extremity",
    "torso", "palms/soles", "oral/genital",
]


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
    train_dataset: str, model_name: str,
    device: torch.device, config, ablation_dir: Path,
) -> nn.Module | None:
    run_id    = f"{model_name}_{AUG_MODE}"
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


def _fbeta(probs: np.ndarray, labels: np.ndarray, thr: float, beta: float) -> float:
    preds = (probs >= thr).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision   = tp / max(tp + fp, 1)
    sensitivity = tp / max(tp + fn, 1)
    b2 = beta ** 2
    return (1 + b2) * precision * sensitivity / max(b2 * precision + sensitivity, 1e-9)


def _f2_optimal_threshold(ds_probs: dict[str, tuple[np.ndarray, np.ndarray]]) -> float:
    """Global F2-optimal threshold across all available datasets."""
    best_thr, best_score = THR_RANGE[0], -1.0
    for thr in THR_RANGE:
        score = float(np.mean([
            _fbeta(p, l, thr, BETA) for p, l in ds_probs.values()
        ]))
        if score > best_score:
            best_score, best_thr = score, thr
    return float(best_thr)


def _metrics_from_preds(preds: np.ndarray, labels: np.ndarray) -> dict:
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    accuracy    = (tp + tn) / max(tp + tn + fp + fn, 1)
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision   = tp / max(tp + fp, 1)
    f1  = (2 * precision * sensitivity) / max(precision + sensitivity, 1e-9)
    b2  = BETA ** 2
    f2  = (1 + b2) * precision * sensitivity / max(b2 * precision + sensitivity, 1e-9)
    return dict(
        accuracy=round(accuracy, 4), sensitivity=round(sensitivity, 4),
        specificity=round(specificity, 4), precision=round(precision, 4),
        f1=round(f1, 4), f2=round(f2, 4),
        tp=tp, tn=tn, fp=fp, fn=fn,
    )


def main() -> None:
    base_cfg     = load_config()
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits_dir   = Path(base_cfg.paths.data_splits)
    melanoma_root = Path(base_cfg.paths.melanoma_data)
    ablation_dir = Path(base_cfg.paths.outputs) / "ablation_noseg"
    out_xlsx     = ablation_dir / "sens" / "evaluation_3models_majority_sens.xlsx"

    gpu_label = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    n_checkpoints = len(ALL_MODELS) * len(ALL_DATASETS)
    n_combos = len(list(combinations(range(n_checkpoints), 3)))

    print(f"\n  Sensitivity fine-tuned 3-model majority-vote ensemble")
    print(f"  Aug mode     : {AUG_MODE}")
    print(f"  Threshold    : F{int(BETA)}-optimal per model  (sweep {THR_RANGE[0]:.2f}–{THR_RANGE[-1]:.2f})")
    print(f"  Device       : {gpu_label}")
    print(f"  Checkpoints  : up to {n_checkpoints}  ({len(ALL_MODELS)} models × {len(ALL_DATASETS)} datasets)")
    print(f"  Combos       : up to {n_combos}\n")

    test_dfs: dict[str, pd.DataFrame] = {}
    for ds in ALL_DATASETS:
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

    checkpoint_probs: dict[tuple[str, str], dict[str, tuple[np.ndarray, np.ndarray, float]]] = {}
    total = len(ALL_DATASETS) * len(ALL_MODELS)
    idx   = 0

    for train_ds in ALL_DATASETS:
        for model_name in ALL_MODELS:
            idx += 1
            config = load_config(model_name)
            model  = _load_checkpoint(train_ds, model_name, device, config, ablation_dir)
            if model is None:
                print(f"  [{idx}/{total}] [SKIP] {model_name}/{train_ds}  checkpoint not found")
                continue

            key      = (model_name, train_ds)
            with_meta = uses_metadata(model_name)
            inp_size  = getattr(config, "input_size", 224)
            nw        = getattr(config, "num_workers", 0)
            checkpoint_probs[key] = {}
            print(f"  [{idx}/{total}] {model_name}/{train_ds} ...", end="", flush=True)

            for eval_ds, eval_df in test_dfs.items():
                ds_obj = EvalDataset(eval_df, inp_size, with_meta=with_meta)
                loader = DataLoader(ds_obj, batch_size=config.batch_size, shuffle=False,
                                    num_workers=nw, pin_memory=(nw > 0))
                probs, labels = _run_inference(model, loader, device, with_meta)
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
                for ds in ALL_DATASETS if ds in checkpoint_probs[key]
            )
            print(f" done  (AUC: {aucs})")

    if len(checkpoint_probs) < 3:
        print("\n  [ERROR] Fewer than 3 checkpoints loaded — cannot form any triplet.\n")
        return

    print(f"\n  Computing F2-optimal thresholds for {len(checkpoint_probs)} checkpoints ...")
    checkpoint_thresholds: dict[tuple[str, str], float] = {}
    for key, ds_dict in checkpoint_probs.items():
        probs_only = {ds: (p, l) for ds, (p, l, _) in ds_dict.items()}
        checkpoint_thresholds[key] = _f2_optimal_threshold(probs_only)

    single_rows: list[dict] = []
    for key, ds_dict in checkpoint_probs.items():
        model_name, train_ds = key
        thr = checkpoint_thresholds[key]
        row: dict = dict(model=model_name, train_dataset=train_ds, f2_threshold=thr)
        sens_vals, f2_vals = [], []
        for eval_ds, (probs, labels, auc) in ds_dict.items():
            m = _metrics_from_preds((probs >= thr).astype(int), labels)
            row[f"auc_{eval_ds}"]         = round(auc, 4)
            row[f"f2_{eval_ds}"]          = m["f2"]
            row[f"sensitivity_{eval_ds}"] = m["sensitivity"]
            row[f"specificity_{eval_ds}"] = m["specificity"]
            row[f"accuracy_{eval_ds}"]    = m["accuracy"]
            row[f"tp_{eval_ds}"]          = m["tp"]
            row[f"tn_{eval_ds}"]          = m["tn"]
            row[f"fp_{eval_ds}"]          = m["fp"]
            row[f"fn_{eval_ds}"]          = m["fn"]
            sens_vals.append(m["sensitivity"])
            f2_vals.append(m["f2"])
        row["avg_f2"]          = round(float(np.mean(f2_vals)), 4) if f2_vals else 0.0
        row["avg_sensitivity"] = round(float(np.mean(sens_vals)), 4) if sens_vals else 0.0
        row["min_sensitivity"] = round(float(np.min(sens_vals)), 4) if sens_vals else 0.0
        ham_s  = row.get("sensitivity_ham10000", 0.0)
        i19_s  = row.get("sensitivity_isic2019", 0.0)
        row["mean_sens_no2020"] = round((ham_s + i19_s) / 2, 4)
        row["mean_f2_no2020"]   = round(
            (row.get("f2_ham10000", 0.0) + row.get("f2_isic2019", 0.0)) / 2, 4
        )
        single_rows.append(row)

    available = list(checkpoint_probs.keys())
    combos    = list(combinations(available, 3))
    print(f"  Evaluating {len(combos)} triplet combinations ...\n")

    summary_rows: list[dict] = []

    for combo in combos:
        k1, k2, k3 = combo
        m1, ds1 = k1;  m2, ds2 = k2;  m3, ds3 = k3
        thr1 = checkpoint_thresholds[k1]
        thr2 = checkpoint_thresholds[k2]
        thr3 = checkpoint_thresholds[k3]

        row: dict = dict(
            combo_name  = f"{m1}_{ds1}+{m2}_{ds2}+{m3}_{ds3}",
            model_1=m1, train_1=ds1,
            model_2=m2, train_2=ds2,
            model_3=m3, train_3=ds3,
            threshold_1=thr1, threshold_2=thr2, threshold_3=thr3,
        )

        sens_vals: list[float] = []
        f2_vals:   list[float] = []

        for eval_ds, eval_df_entry in test_dfs.items():
            if not all(eval_ds in checkpoint_probs[k] for k in combo):
                continue
            p1, labels, _ = checkpoint_probs[k1][eval_ds]
            p2, _,      _ = checkpoint_probs[k2][eval_ds]
            p3, _,      _ = checkpoint_probs[k3][eval_ds]

            votes          = (p1 >= thr1).astype(int) + (p2 >= thr2).astype(int) + (p3 >= thr3).astype(int)
            ensemble_preds = (votes >= 2).astype(int)

            avg_probs = (p1 + p2 + p3) / 3.0
            try:
                auc = float(roc_auc_score(labels, avg_probs))
            except Exception:
                auc = float("nan")

            m = _metrics_from_preds(ensemble_preds, labels)
            row[f"auc_{eval_ds}"]         = round(auc, 4)
            row[f"f2_{eval_ds}"]          = m["f2"]
            row[f"sensitivity_{eval_ds}"] = m["sensitivity"]
            row[f"specificity_{eval_ds}"] = m["specificity"]
            row[f"accuracy_{eval_ds}"]    = m["accuracy"]
            row[f"tp_{eval_ds}"]          = m["tp"]
            row[f"tn_{eval_ds}"]          = m["tn"]
            row[f"fp_{eval_ds}"]          = m["fp"]
            row[f"fn_{eval_ds}"]          = m["fn"]
            sens_vals.append(m["sensitivity"])
            f2_vals.append(m["f2"])

        row["avg_f2"]          = round(float(np.mean(f2_vals)), 4) if f2_vals else 0.0
        row["avg_sensitivity"] = round(float(np.mean(sens_vals)), 4) if sens_vals else 0.0
        row["min_sensitivity"] = round(float(np.min(sens_vals)), 4) if sens_vals else 0.0
        ham_s  = row.get("sensitivity_ham10000", 0.0)
        i19_s  = row.get("sensitivity_isic2019", 0.0)
        row["mean_sens_no2020"] = round((ham_s + i19_s) / 2, 4)
        row["mean_f2_no2020"]   = round(
            (row.get("f2_ham10000", 0.0) + row.get("f2_isic2019", 0.0)) / 2, 4
        )
        summary_rows.append(row)

    df = pd.DataFrame(summary_rows)

    id_cols = [
        "combo_name",
        "model_1", "train_1", "model_2", "train_2", "model_3", "train_3",
        "threshold_1", "threshold_2", "threshold_3",
        "avg_f2", "avg_sensitivity", "min_sensitivity",
        "mean_sens_no2020", "mean_f2_no2020",
    ]
    extra_cols = [c for c in df.columns if c not in id_cols]
    df = df[id_cols + extra_cols]

    write_excel_sheet(out_xlsx, "Summary_MinSens",
                      df.sort_values("min_sensitivity", ascending=False))
    write_excel_sheet(out_xlsx, "Summary_SensNo2020",
                      df.sort_values("mean_sens_no2020", ascending=False))
    write_excel_sheet(out_xlsx, "Summary_F2",
                      df.sort_values("avg_f2", ascending=False))

    if single_rows:
        df_s = pd.DataFrame(single_rows)
        s_id = ["model", "train_dataset", "f2_threshold",
                "avg_f2", "avg_sensitivity", "min_sensitivity",
                "mean_sens_no2020", "mean_f2_no2020"]
        extra_s = [c for c in df_s.columns if c not in s_id]
        write_excel_sheet(out_xlsx, "Single_Models",
                          df_s[s_id + extra_s].sort_values("mean_sens_no2020", ascending=False))

    for eval_ds in ALL_DATASETS:
        sens_col = f"sensitivity_{eval_ds}"
        if sens_col not in df.columns:
            continue
        ds_cols  = ["combo_name", "model_1", "train_1", "model_2", "train_2", "model_3", "train_3",
                    "threshold_1", "threshold_2", "threshold_3"]
        ds_cols += [c for c in df.columns if c.endswith(f"_{eval_ds}")]
        write_excel_sheet(out_xlsx, eval_ds.upper(),
                          df[ds_cols].sort_values(sens_col, ascending=False))

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  Results saved : {out_xlsx}")
    print(f"  Total combos  : {len(df)}")
    print(f"  Sheets        : Summary_MinSens | Summary_SensNo2020 | Summary_F2")
    print(f"                  Single_Models | {' | '.join(ds.upper() for ds in ALL_DATASETS)}")
    print(f"{sep}\n")

    print("  Top 10 by min_sensitivity (best worst-case across all datasets):\n")
    top_cols = ["combo_name", "min_sensitivity", "mean_sens_no2020",
                "sensitivity_ham10000", "sensitivity_isic2019", "sensitivity_isic2020"]
    top_cols = [c for c in top_cols if c in df.columns]
    print(df.sort_values("min_sensitivity", ascending=False).head(10)[top_cols].to_string(index=False))
    print()

    print("  Top 10 by mean_sens_no2020 (HAM10000 + ISIC2019 only):\n")
    print(df.sort_values("mean_sens_no2020", ascending=False).head(10)[top_cols].to_string(index=False))
    print()


if __name__ == "__main__":
    main()
