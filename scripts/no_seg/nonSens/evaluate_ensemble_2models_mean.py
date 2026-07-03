# all c(n,2) pairs, mean probs + best shared threshold. aug=none only. no-seg version
from __future__ import annotations

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
# no aug models -- aug=none only
AUG_MODE          = "none"
THRESHOLDS        = np.round(np.arange(0.35, 0.66, 0.01), 2)   # 31 steps
DEFAULT_THRESHOLD = 0.50   # reference threshold -- stored to show gain from tuning

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
    train_dataset: str, model_name: str, device: torch.device,
    config, ablation_dir: Path,
) -> nn.Module | None:
    run_id    = f"{model_name}_{AUG_MODE}"
    ckpt_file = ablation_dir / train_dataset / run_id / "checkpoints" / f"{run_id}.pt"
    if not ckpt_file.exists():
        return None
    model = build_model(model_name, config, num_classes=2)
    model.load_state_dict(torch.load(ckpt_file, map_location=device, weights_only=True))
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


def _best_threshold(ds_probs: dict[str, tuple[np.ndarray, np.ndarray]]) -> float:
    best_thr, best_avg = THRESHOLDS[0], -1.0
    for thr in THRESHOLDS:
        avg = float(np.mean([
            _metrics_at_threshold(p, l, thr)["f1"]
            for p, l in ds_probs.values()
        ]))
        if avg > best_avg:
            best_avg, best_thr = avg, thr
    return float(best_thr)


def _fill_thr50(
    row: dict,
    ds_probs: dict[str, tuple[np.ndarray, np.ndarray]],
    eval_datasets: list[str],
) -> None:
    per_f1, per_sens, per_spec = [], [], []
    for eval_ds, (probs, labels) in ds_probs.items():
        m = _metrics_at_threshold(probs, labels, DEFAULT_THRESHOLD)
        row[f"f1_thr50_{eval_ds}"]          = m["f1"]
        row[f"sensitivity_thr50_{eval_ds}"] = m["sensitivity"]
        row[f"specificity_thr50_{eval_ds}"] = m["specificity"]
        row[f"accuracy_thr50_{eval_ds}"]    = m["accuracy"]
        per_f1.append(m["f1"])
        per_sens.append(m["sensitivity"])
        per_spec.append(m["specificity"])
    row["avg_f1_thr50"]          = round(float(np.mean(per_f1)),   4) if per_f1   else 0.0
    row["avg_sensitivity_thr50"] = round(float(np.mean(per_sens)), 4) if per_sens else 0.0
    row["avg_specificity_thr50"] = round(float(np.mean(per_spec)), 4) if per_spec else 0.0
    row["delta_avg_f1"]          = round(float(row.get("avg_f1", 0.0)) - row["avg_f1_thr50"], 4)


def run_group(
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
            model  = _load_checkpoint(train_ds, model_name, device, config, ablation_dir)
            if model is None:
                print(f"    [{idx}/{total}] [SKIP] {model_name}/{train_ds} not found")
                continue

            key      = (model_name, train_ds)
            meta     = uses_metadata(model_name)
            inp_size = getattr(config, "input_size", 224)
            nw       = getattr(config, "num_workers", 0)
            checkpoint_probs[key] = {}
            print(f"    [{idx}/{total}] {model_name}/{train_ds} ...", end="", flush=True)

            for eval_ds in eval_datasets:
                if eval_ds not in test_dfs:
                    continue
                ds_obj = EvalDataset(test_dfs[eval_ds], inp_size, with_meta=meta)
                loader = DataLoader(ds_obj, batch_size=config.batch_size, shuffle=False,
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

    if len(checkpoint_probs) < 2:
        print("    [SKIP] fewer than 2 checkpoints loaded, cannot build pairs")
        return results, single_results

    checkpoint_thresholds: dict[tuple[str, str], float] = {}
    for key, ds_dict in checkpoint_probs.items():
        probs_only = {ds: (p, l) for ds, (p, l, _) in ds_dict.items()}
        checkpoint_thresholds[key] = _best_threshold(probs_only)

    for key, ds_dict in checkpoint_probs.items():
        model_name, train_ds = key
        thr = checkpoint_thresholds[key]
        row = dict(
            model          = model_name,
            train_dataset  = train_ds,
            best_threshold = thr,
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
        _fill_thr50(row, {ds: (p, l) for ds, (p, l, _) in ds_dict.items()}, eval_datasets)
        single_results.append(row)

    available = list(checkpoint_probs.keys())
    n         = len(available)
    n_pairs   = len(list(combinations(available, 2)))
    print(f"    Evaluating {n_pairs} pairs from {n} checkpoints (mean prob + threshold sweep) ...")

    for combo in combinations(available, 2):
        k1, k2 = combo
        m1, ds1 = k1
        m2, ds2 = k2
        pair_name = f"{m1}_{ds1}+{m2}_{ds2}"

        pair_ds_probs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        pair_aucs:     dict[str, float] = {}
        for eval_ds in eval_datasets:
            if not all(eval_ds in checkpoint_probs[k] for k in combo):
                continue
            p1, labels, _ = checkpoint_probs[k1][eval_ds]
            p2, _,      _ = checkpoint_probs[k2][eval_ds]
            mean_p = (p1 + p2) / 2.0
            try:
                auc = float(roc_auc_score(labels, mean_p))
            except Exception:
                auc = float("nan")
            pair_ds_probs[eval_ds] = (mean_p, labels)
            pair_aucs[eval_ds]     = auc

        if not pair_ds_probs:
            continue

        mean_thr = _best_threshold(pair_ds_probs)

        summary = dict(
            pair_name      = pair_name,
            model_1        = m1,  train_1 = ds1,
            model_2        = m2,  train_2 = ds2,
            mean_threshold = mean_thr,
        )

        per_ds_f1s = []
        for eval_ds, (mean_p, labels) in pair_ds_probs.items():
            m = _metrics_at_threshold(mean_p, labels, mean_thr)
            per_ds_f1s.append(m["f1"])
            summary[f"auc_{eval_ds}"]         = round(pair_aucs[eval_ds], 4)
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
        _fill_thr50(summary, pair_ds_probs, eval_datasets)
        results.append(summary)

    return results, single_results


def main() -> None:
    base_cfg     = load_config()
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits_dir   = Path(base_cfg.paths.data_splits)
    melanoma_root = Path(base_cfg.paths.melanoma_data)
    ablation_dir = Path(base_cfg.paths.outputs) / "ablation"
    noseg_dir    = Path(base_cfg.paths.outputs) / "ablation_noseg"
    noseg_dir.mkdir(parents=True, exist_ok=True)
    eval_xlsx    = noseg_dir / "evaluation_2models_mean.xlsx"

    gpu_label    = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    n_ckpts      = len(ALL_MODELS) * len(ALL_DATASETS)
    n_pairs_est  = len(list(combinations(range(n_ckpts), 2)))
    print(f"\n  Device               : {gpu_label}")
    print(f"  Aug mode             : {AUG_MODE}  (no aug models -- proven worse)")
    print(f"  Checkpoints (max)    : {n_ckpts}  ({len(ALL_MODELS)} models x {len(ALL_DATASETS)} train datasets)")
    print(f"  Pairs (max)          : {n_pairs_est}  (C({n_ckpts},2))")
    print(f"  Method               : mean probability + best shared threshold per pair")
    print(f"  Default thr          : {DEFAULT_THRESHOLD}  (saved alongside best for comparison)\n")

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

    try:
        all_results, all_singles = run_group(
            ALL_DATASETS, ALL_DATASETS, test_dfs,
            device, ablation_dir, ALL_MODELS,
        )
    except Exception as exc:
        import traceback
        print(f"  [ERROR] {exc}")
        traceback.print_exc()
        return

    if not all_results:
        print("\n  No results to save.\n")
        return

    # save to excel

    df = pd.DataFrame(all_results)

    id_cols = [
        "pair_name",
        "model_1", "train_1", "model_2", "train_2",
        "mean_threshold",
        "avg_f1", "avg_f1_thr50", "delta_avg_f1",
        "avg_sensitivity", "avg_sensitivity_thr50",
        "avg_specificity", "avg_specificity_thr50",
        "avg_accuracy",
    ]
    extra_cols = [c for c in df.columns if c not in id_cols]
    df = df[[c for c in id_cols if c in df.columns] + extra_cols]

    write_excel_sheet(eval_xlsx, "Summary", df.sort_values("avg_f1", ascending=False))

    if all_singles:
        df_singles = pd.DataFrame(all_singles)
        single_id  = [
            "model", "train_dataset", "best_threshold",
            "avg_f1", "avg_f1_thr50", "delta_avg_f1",
            "avg_sensitivity", "avg_sensitivity_thr50",
            "avg_specificity", "avg_specificity_thr50",
            "avg_accuracy",
        ]
        extra_s    = [c for c in df_singles.columns if c not in single_id]
        df_singles = df_singles[[c for c in single_id if c in df_singles.columns] + extra_s]
        df_singles = df_singles.sort_values("avg_f1", ascending=False)
        write_excel_sheet(eval_xlsx, "Single_Models", df_singles)

    for eval_ds in ALL_DATASETS:
        f1_col = f"f1_{eval_ds}"
        if f1_col not in df.columns:
            continue
        ds_id   = ["pair_name", "model_1", "train_1", "model_2", "train_2", "mean_threshold"]
        ds_cols = ds_id + [c for c in df.columns
                           if c.endswith(f"_{eval_ds}") or c == f"f1_thr50_{eval_ds}"]
        thr50_cols = [c for c in df.columns if f"thr50_{eval_ds}" in c]
        all_ds_cols = list(dict.fromkeys(ds_cols + thr50_cols))
        ev_df = df[[c for c in all_ds_cols if c in df.columns]].sort_values(f1_col, ascending=False)
        write_excel_sheet(eval_xlsx, eval_ds.upper(), ev_df)

    best = (
        df.groupby("pair_name", as_index=False)
        .agg(avg_f1=("avg_f1", "mean"),
             avg_accuracy=("avg_accuracy", "mean"),
             avg_f1_thr50=("avg_f1_thr50", "mean"),
             delta_avg_f1=("delta_avg_f1", "mean"),
             avg_sensitivity=("avg_sensitivity", "mean"),
             avg_specificity=("avg_specificity", "mean"))
        .sort_values("avg_f1", ascending=False)
    )
    pair_info = df.drop_duplicates("pair_name")[
        ["pair_name", "model_1", "train_1", "model_2", "train_2", "mean_threshold"]
    ]
    best = best.merge(pair_info, on="pair_name", how="left")
    best_cols = ["pair_name", "model_1", "train_1", "model_2", "train_2", "mean_threshold",
                 "avg_f1", "avg_accuracy", "avg_f1_thr50", "delta_avg_f1",
                 "avg_sensitivity", "avg_specificity"]
    write_excel_sheet(eval_xlsx, "Best_Pairs", best[[c for c in best_cols if c in best.columns]])

    best_pair   = df.loc[df["avg_f1"].idxmax()]
    best_single = max(all_singles, key=lambda r: r["avg_f1"]) if all_singles else None

    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  Results saved : {eval_xlsx}")
    print(f"  Sheets        : Summary | Single_Models | Best_Pairs | {' | '.join(ds.upper() for ds in ALL_DATASETS)}")
    print(f"  Total pairs   : {len(df)}")

    print(f"\n  Best 2-model mean pair (no-seg):")
    print(f"    Pair             : {best_pair['pair_name']}")
    print(f"    Best threshold   : {best_pair['mean_threshold']:.2f}  ->  avg_f1={best_pair['avg_f1']:.4f}")
    print(f"    Thr=0.50         :                    avg_f1={best_pair['avg_f1_thr50']:.4f}")
    print(f"    Tuning gain      : {best_pair['delta_avg_f1']:+.4f}")
    for eval_ds in ALL_DATASETS:
        if f"f1_{eval_ds}" in best_pair:
            print(
                f"    {eval_ds.upper():<12}  "
                f"AUC={best_pair.get(f'auc_{eval_ds}', float('nan')):.4f}  "
                f"F1_best={best_pair[f'f1_{eval_ds}']:.4f}  "
                f"F1_thr50={best_pair.get(f'f1_thr50_{eval_ds}', float('nan')):.4f}  "
                f"sens={best_pair[f'sensitivity_{eval_ds}']:.4f}  "
                f"spec={best_pair[f'specificity_{eval_ds}']:.4f}"
            )

    if best_single:
        delta = round(float(best_pair["avg_f1"]) - float(best_single["avg_f1"]), 4)
        sign  = "+" if delta >= 0 else ""
        print(f"\n  vs best single model:")
        print(f"    Single  : {best_single['model']}|{best_single['train_dataset']}  "
              f"avg_f1={best_single['avg_f1']:.4f}  thr={best_single['best_threshold']:.2f}")
        print(f"    Delta   : {sign}{delta:.4f}")

    print(f"\n  Top 5 pairs:\n")
    print(df.head(5)[["pair_name", "mean_threshold", "avg_f1", "avg_accuracy",
                       "avg_f1_thr50", "delta_avg_f1"]].to_string(index=False))
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
