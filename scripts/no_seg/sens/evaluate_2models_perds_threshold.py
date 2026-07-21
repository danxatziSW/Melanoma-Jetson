"""Two-model ensembles scored with a threshold tuned per dataset."""
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
ALL_MODELS   = ["resnet50", "efficientnet_b2", "mobilenetv3_large", "medfusionnet", "yolov8_cls"]

AUG_MODE   = "none_sens"
THRESHOLDS = np.round(np.arange(0.20, 0.86, 0.01), 2)

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
    config, ablation_noseg_dir: Path,
) -> nn.Module | None:
    run_id    = f"{model_name}_{AUG_MODE}"
    ckpt_file = ablation_noseg_dir / train_dataset / run_id / "checkpoints" / f"{run_id}.pt"
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
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1   = (2 * prec * sens) / max(prec + sens, 1e-9)
    f2   = (5 * prec * sens) / max(4 * prec + sens, 1e-9)
    acc  = (tp + tn) / max(tp + tn + fp + fn, 1)
    return dict(
        accuracy=round(acc, 4),    sensitivity=round(sens, 4),
        specificity=round(spec, 4), precision=round(prec, 4),
        f1=round(f1, 4),           f2=round(f2, 4),
        tp=tp, tn=tn, fp=fp, fn=fn,
    )


def _best_f2_thr_single(probs: np.ndarray, labels: np.ndarray) -> tuple[float, dict]:
    """Threshold maximising F2 for one dataset."""
    best_thr, best_f2 = THRESHOLDS[0], -1.0
    for thr in THRESHOLDS:
        m = _metrics_at_threshold(probs, labels, thr)
        if m["f2"] > best_f2:
            best_f2 = m["f2"]
            best_thr = thr
    return float(best_thr), _metrics_at_threshold(probs, labels, best_thr)


def run_group(
    train_datasets: list[str],
    eval_datasets: list[str],
    test_dfs: dict[str, pd.DataFrame],
    device: torch.device,
    ablation_noseg_dir: Path,
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
            model  = _load_checkpoint(train_ds, model_name, device, config, ablation_noseg_dir)
            if model is None:
                print(f"    [{idx}/{total}] [SKIP] {model_name}/{train_ds}: not found")
                continue

            key      = (model_name, train_ds)
            meta     = uses_metadata(model_name)
            inp_size = getattr(config, "input_size", 224)
            nw       = getattr(config, "num_workers", 4)
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
            aucs = ", ".join(f"{checkpoint_probs[key][ds][2]:.3f}"
                             for ds in eval_datasets if ds in checkpoint_probs[key])
            print(f" done  (AUC: {aucs})")

    if len(checkpoint_probs) < 2:
        print("    [SKIP] fewer than 2 checkpoints loaded")
        return results, single_results

    for key, ds_dict in checkpoint_probs.items():
        model_name, train_ds = key
        row = dict(model=model_name, train_dataset=train_ds)
        per_f2, per_f1, per_sens, per_spec, per_acc = [], [], [], [], []
        for eval_ds, (probs, labels, auc) in ds_dict.items():
            thr, met = _best_f2_thr_single(probs, labels)
            row[f"thr_{eval_ds}"]         = thr
            row[f"auc_{eval_ds}"]         = round(auc, 4)
            row[f"f2_{eval_ds}"]          = met["f2"]
            row[f"f1_{eval_ds}"]          = met["f1"]
            row[f"sensitivity_{eval_ds}"] = met["sensitivity"]
            row[f"specificity_{eval_ds}"] = met["specificity"]
            row[f"accuracy_{eval_ds}"]    = met["accuracy"]
            row[f"tp_{eval_ds}"]          = met["tp"]
            row[f"tn_{eval_ds}"]          = met["tn"]
            row[f"fp_{eval_ds}"]          = met["fp"]
            row[f"fn_{eval_ds}"]          = met["fn"]
            per_f2.append(met["f2"]); per_f1.append(met["f1"])
            per_sens.append(met["sensitivity"]); per_spec.append(met["specificity"])
            per_acc.append(met["accuracy"])
        row["avg_f2"]          = round(float(np.mean(per_f2)),   4)
        row["avg_f1"]          = round(float(np.mean(per_f1)),   4)
        row["avg_sensitivity"] = round(float(np.mean(per_sens)), 4)
        row["avg_specificity"] = round(float(np.mean(per_spec)), 4)
        row["avg_accuracy"]    = round(float(np.mean(per_acc)),  4)
        row["min_sensitivity"] = round(float(np.min(per_sens)),  4)
        single_results.append(row)

    available = list(checkpoint_probs.keys())
    n_pairs   = len(list(combinations(available, 2)))
    print(f"\n    Evaluating {n_pairs} pairs with per-dataset thresholds ...")

    for k1, k2 in combinations(available, 2):
        m1, ds1 = k1
        m2, ds2 = k2
        pair_name = f"{m1}_{ds1}+{m2}_{ds2}"

        pair_ds_probs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        pair_aucs:     dict[str, float] = {}
        for eval_ds in eval_datasets:
            if not all(eval_ds in checkpoint_probs[k] for k in (k1, k2)):
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

        summary = dict(
            pair_name=pair_name, model_1=m1, train_1=ds1, model_2=m2, train_2=ds2,
        )

        per_f2, per_f1, per_sens, per_spec, per_acc = [], [], [], [], []
        for eval_ds in eval_datasets:
            if eval_ds not in pair_ds_probs:
                continue
            mean_p, labels = pair_ds_probs[eval_ds]
            thr, met = _best_f2_thr_single(mean_p, labels)
            summary[f"thr_{eval_ds}"]         = thr
            summary[f"auc_{eval_ds}"]         = round(pair_aucs.get(eval_ds, float("nan")), 4)
            summary[f"f2_{eval_ds}"]          = met["f2"]
            summary[f"f1_{eval_ds}"]          = met["f1"]
            summary[f"sensitivity_{eval_ds}"] = met["sensitivity"]
            summary[f"specificity_{eval_ds}"] = met["specificity"]
            summary[f"accuracy_{eval_ds}"]    = met["accuracy"]
            summary[f"tp_{eval_ds}"]          = met["tp"]
            summary[f"tn_{eval_ds}"]          = met["tn"]
            summary[f"fp_{eval_ds}"]          = met["fp"]
            summary[f"fn_{eval_ds}"]          = met["fn"]
            per_f2.append(met["f2"]); per_f1.append(met["f1"])
            per_sens.append(met["sensitivity"]); per_spec.append(met["specificity"])
            per_acc.append(met["accuracy"])

        active = [ds for ds in eval_datasets if f"f2_{ds}" in summary]
        summary["avg_f2"]          = round(float(np.mean(per_f2)),   4)
        summary["avg_f1"]          = round(float(np.mean(per_f1)),   4)
        summary["avg_sensitivity"] = round(float(np.mean(per_sens)), 4)
        summary["avg_specificity"] = round(float(np.mean(per_spec)), 4)
        summary["avg_accuracy"]    = round(float(np.mean(per_acc)),  4)
        summary["min_sensitivity"] = round(float(np.min(per_sens)),  4)
        summary["avg_auc"]         = round(float(np.nanmean(
            [pair_aucs.get(ds, float("nan")) for ds in active])), 4)
        results.append(summary)

    return results, single_results


def main() -> None:
    base_cfg           = load_config()
    device             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits_dir         = Path(base_cfg.paths.data_splits)
    melanoma_root      = Path(base_cfg.paths.melanoma_data)
    ablation_noseg_dir = Path(base_cfg.paths.outputs) / "ablation_noseg"
    ablation_noseg_dir.mkdir(parents=True, exist_ok=True)
    out_xlsx           = ablation_noseg_dir / "evaluation_2models_perds_threshold.xlsx"

    gpu_label   = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    n_ckpts     = len(ALL_MODELS) * len(ALL_DATASETS)
    n_pairs_est = len(list(combinations(range(n_ckpts), 2)))
    print(f"\n  Device        : {gpu_label}")
    print(f"  Suffix        : _{AUG_MODE}")
    print(f"  Checkpoints   : {n_ckpts} max")
    print(f"  Pairs max     : {n_pairs_est}  C({n_ckpts},2)")
    print(f"  Mode          : per-dataset F2-optimal threshold")
    print(f"  Threshold rng : {THRESHOLDS[0]} to {THRESHOLDS[-1]}\n")

    test_dfs: dict[str, pd.DataFrame] = {}
    for ds in ALL_DATASETS:
        try:
            df = _load_test_df(splits_dir, ds, melanoma_root)
            test_dfs[ds] = df
            print(f"  {ds.upper():<12} test={len(df)}  mel={int(df['binary_label'].sum())}  "
                  f"non-mel={int((df['binary_label']==0).sum())}")
        except (FileNotFoundError, ValueError) as exc:
            print(f"  [SKIP] {ds}: {exc}")
    print()

    all_results, all_singles = run_group(
        ALL_DATASETS, ALL_DATASETS, test_dfs,
        device, ablation_noseg_dir, ALL_MODELS,
    )

    if not all_results:
        print("\n  No results to save.\n")
        return

    df = pd.DataFrame(all_results)
    id_cols = ["pair_name", "model_1", "train_1", "model_2", "train_2",
               "avg_f2", "avg_f1", "avg_sensitivity", "avg_specificity",
               "avg_accuracy", "min_sensitivity", "avg_auc"]
    thr_cols  = [f"thr_{ds}"         for ds in ALL_DATASETS if f"thr_{ds}"  in df.columns]
    auc_cols  = [f"auc_{ds}"         for ds in ALL_DATASETS if f"auc_{ds}"  in df.columns]
    extra     = [c for c in df.columns if c not in id_cols + thr_cols + auc_cols]
    ordered   = [c for c in id_cols + thr_cols + auc_cols + extra if c in df.columns]
    df        = df[ordered]

    df_f2  = df.sort_values("avg_f2",          ascending=False)
    df_s   = df.sort_values("avg_sensitivity",  ascending=False)
    df_ms  = df.sort_values("min_sensitivity",  ascending=False)

    write_excel_sheet(out_xlsx, "Summary_F2",           df_f2)
    write_excel_sheet(out_xlsx, "Summary_Sensitivity",  df_s)
    write_excel_sheet(out_xlsx, "Summary_MinSens",      df_ms)

    if all_singles:
        df_singles = pd.DataFrame(all_singles)
        s_id = ["model", "train_dataset",
                "avg_f2", "avg_f1", "avg_sensitivity", "avg_specificity",
                "avg_accuracy", "min_sensitivity"]
        s_thr   = [f"thr_{ds}" for ds in ALL_DATASETS if f"thr_{ds}" in df_singles.columns]
        s_extra = [c for c in df_singles.columns if c not in s_id + s_thr]
        s_ord   = [c for c in s_id + s_thr + s_extra if c in df_singles.columns]
        write_excel_sheet(out_xlsx, "Single_Models",
                          df_singles[s_ord].sort_values("avg_f2", ascending=False))

    for eval_ds in ALL_DATASETS:
        if f"f2_{eval_ds}" not in df.columns:
            continue
        ds_cols = (["pair_name", "model_1", "train_1", "model_2", "train_2",
                    f"thr_{eval_ds}"]
                   + [c for c in df.columns if c.endswith(f"_{eval_ds}")])
        write_excel_sheet(out_xlsx, eval_ds.upper(),
                          df[[c for c in ds_cols if c in df.columns]]
                          .sort_values(f"f2_{eval_ds}", ascending=False))

    best_f2   = df.loc[df["avg_f2"].idxmax()]
    best_sens = df.loc[df["avg_sensitivity"].idxmax()]
    best_ms   = df.loc[df["min_sensitivity"].idxmax()]

    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  Saved : {out_xlsx}")
    print(f"  Pairs : {len(df)}  (from {len(all_singles)} loaded checkpoints)")

    print(f"\n  ── Best by avg F2 (per-dataset thresholds) ──")
    print(f"     {best_f2['pair_name']}")
    print(f"     avg_f2={best_f2['avg_f2']:.4f}  avg_f1={best_f2['avg_f1']:.4f}  "
          f"avg_sens={best_f2['avg_sensitivity']:.4f}  avg_spec={best_f2['avg_specificity']:.4f}")
    for ds in ALL_DATASETS:
        if f"f2_{ds}" in best_f2:
            print(f"     {ds.upper():<12} thr={best_f2[f'thr_{ds}']:.2f}  "
                  f"F2={best_f2[f'f2_{ds}']:.4f}  "
                  f"sens={best_f2[f'sensitivity_{ds}']:.4f}  "
                  f"spec={best_f2[f'specificity_{ds}']:.4f}")

    print(f"\n  ── Best by avg Sensitivity ──")
    print(f"     {best_sens['pair_name']}")
    print(f"     avg_sens={best_sens['avg_sensitivity']:.4f}  avg_f2={best_sens['avg_f2']:.4f}")
    for ds in ALL_DATASETS:
        if f"sensitivity_{ds}" in best_sens:
            print(f"     {ds.upper():<12} thr={best_sens[f'thr_{ds}']:.2f}  "
                  f"sens={best_sens[f'sensitivity_{ds}']:.4f}  "
                  f"spec={best_sens[f'specificity_{ds}']:.4f}")

    print(f"\n  ── Best by min Sensitivity (best worst-case dataset) ──")
    print(f"     {best_ms['pair_name']}")
    print(f"     min_sens={best_ms['min_sensitivity']:.4f}  avg_sens={best_ms['avg_sensitivity']:.4f}  "
          f"avg_f2={best_ms['avg_f2']:.4f}")
    for ds in ALL_DATASETS:
        if f"sensitivity_{ds}" in best_ms:
            print(f"     {ds.upper():<12} thr={best_ms[f'thr_{ds}']:.2f}  "
                  f"sens={best_ms[f'sensitivity_{ds}']:.4f}  "
                  f"spec={best_ms[f'specificity_{ds}']:.4f}")

    print(f"\n  Top 5 by avg F2 (per-dataset thresholds):\n")
    top5 = df_f2.head(5)[
        ["pair_name", "avg_f2", "avg_f1", "avg_sensitivity", "min_sensitivity", "avg_specificity"]
        + [f"thr_{ds}" for ds in ALL_DATASETS if f"thr_{ds}" in df.columns]
    ]
    print(top5.to_string(index=False))
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
