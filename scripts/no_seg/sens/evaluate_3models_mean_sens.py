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
ALL_MODELS   = ["resnet50", "efficientnet_b2", "mobilenetv3_large", "convnext_tiny_se", "medfusionnet", "yolov8_cls"]

# _none_sens suffix matches train_sensitivity_all.py output
AUG_MODE          = "none_sens"
THRESHOLDS        = np.round(np.arange(0.20, 0.86, 0.01), 2)   # wider: sens models output higher probs
DEFAULT_THRESHOLD = 0.50

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


def _best_threshold_f1(ds_probs: dict[str, tuple[np.ndarray, np.ndarray]]) -> float:
    best_thr, best_avg = THRESHOLDS[0], -1.0
    for thr in THRESHOLDS:
        avg = float(np.mean([_metrics_at_threshold(p, l, thr)["f1"]
                             for p, l in ds_probs.values()]))
        if avg > best_avg:
            best_avg, best_thr = avg, thr
    return float(best_thr)


def _best_threshold_f2(ds_probs: dict[str, tuple[np.ndarray, np.ndarray]]) -> float:
    best_thr, best_avg = THRESHOLDS[0], -1.0
    for thr in THRESHOLDS:
        avg = float(np.mean([_metrics_at_threshold(p, l, thr)["f2"]
                             for p, l in ds_probs.values()]))
        if avg > best_avg:
            best_avg, best_thr = avg, thr
    return float(best_thr)


def _fill_thr50(row: dict, ds_probs: dict, eval_datasets: list[str]) -> None:
    per_f1, per_sens, per_spec, per_f2 = [], [], [], []
    for eval_ds, (probs, labels) in ds_probs.items():
        m = _metrics_at_threshold(probs, labels, DEFAULT_THRESHOLD)
        row[f"f1_thr50_{eval_ds}"]   = m["f1"]
        row[f"f2_thr50_{eval_ds}"]   = m["f2"]
        row[f"sens_thr50_{eval_ds}"] = m["sensitivity"]
        row[f"spec_thr50_{eval_ds}"] = m["specificity"]
        row[f"acc_thr50_{eval_ds}"]  = m["accuracy"]
        per_f1.append(m["f1"]); per_f2.append(m["f2"])
        per_sens.append(m["sensitivity"]); per_spec.append(m["specificity"])
    row["avg_f1_thr50"]          = round(float(np.mean(per_f1)),   4)
    row["avg_f2_thr50"]          = round(float(np.mean(per_f2)),   4)
    row["avg_sensitivity_thr50"] = round(float(np.mean(per_sens)), 4)
    row["avg_specificity_thr50"] = round(float(np.mean(per_spec)), 4)
    row["delta_avg_f1"]          = round(float(row.get("avg_f1", 0.0)) - row["avg_f1_thr50"], 4)
    row["delta_avg_f2"]          = round(float(row.get("avg_f2", 0.0)) - row["avg_f2_thr50"], 4)


def run_group(
    train_datasets: list[str], eval_datasets: list[str],
    test_dfs: dict[str, pd.DataFrame],
    device: torch.device, ablation_noseg_dir: Path, models: list[str],
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
                print(f"    [{idx}/{total}] [SKIP] {model_name}/{train_ds} — not found")
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

    if len(checkpoint_probs) < 3:
        print("    [SKIP] fewer than 3 checkpoints loaded — can't form a triplet")
        return results, single_results

    # single-model rows, same as the pair script — gives a baseline to compare the triplets against
    for key, ds_dict in checkpoint_probs.items():
        model_name, train_ds = key
        probs_only = {ds: (p, l) for ds, (p, l, _) in ds_dict.items()}
        thr_f1 = _best_threshold_f1(probs_only)
        thr_f2 = _best_threshold_f2(probs_only)
        row = dict(model=model_name, train_dataset=train_ds,
                   best_threshold_f1=thr_f1, best_threshold_f2=thr_f2)
        per_ds_f1s, per_ds_f2s = [], []
        for eval_ds, (probs, labels, auc) in ds_dict.items():
            m = _metrics_at_threshold(probs, labels, thr_f1)
            per_ds_f1s.append(m["f1"]); per_ds_f2s.append(m["f2"])
            row[f"auc_{eval_ds}"]         = round(auc, 4)
            row[f"f1_{eval_ds}"]          = m["f1"]
            row[f"f2_{eval_ds}"]          = m["f2"]
            row[f"sensitivity_{eval_ds}"] = m["sensitivity"]
            row[f"specificity_{eval_ds}"] = m["specificity"]
            row[f"accuracy_{eval_ds}"]    = m["accuracy"]
            row[f"tp_{eval_ds}"]          = m["tp"]
            row[f"tn_{eval_ds}"]          = m["tn"]
            row[f"fp_{eval_ds}"]          = m["fp"]
            row[f"fn_{eval_ds}"]          = m["fn"]
        row["avg_f1"] = round(float(np.mean(per_ds_f1s)), 4)
        row["avg_f2"] = round(float(np.mean(per_ds_f2s)), 4)
        row["avg_sensitivity"] = round(float(np.mean(
            [row[f"sensitivity_{ds}"] for ds in eval_datasets if f"sensitivity_{ds}" in row])), 4)
        row["avg_specificity"] = round(float(np.mean(
            [row[f"specificity_{ds}"] for ds in eval_datasets if f"specificity_{ds}" in row])), 4)
        row["avg_accuracy"] = round(float(np.mean(
            [row[f"accuracy_{ds}"] for ds in eval_datasets if f"accuracy_{ds}" in row])), 4)
        _fill_thr50(row, probs_only, eval_datasets)
        single_results.append(row)

    # triplets — average the 3 probabilities first, then sweep one shared threshold over the mean
    # (this is the "mean" ensemble, as opposed to the per-model-threshold majority vote elsewhere)
    available = list(checkpoint_probs.keys())
    combos    = list(combinations(available, 3))
    print(f"\n    Evaluating {len(combos)} triplets from {len(available)} checkpoints ...")

    for k1, k2, k3 in combos:
        m1, ds1 = k1
        m2, ds2 = k2
        m3, ds3 = k3
        combo_name = f"{m1}_{ds1}+{m2}_{ds2}+{m3}_{ds3}"

        combo_ds_probs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        combo_aucs:     dict[str, float] = {}
        for eval_ds in eval_datasets:
            if not all(eval_ds in checkpoint_probs[k] for k in (k1, k2, k3)):
                continue
            p1, labels, _ = checkpoint_probs[k1][eval_ds]
            p2, _,      _ = checkpoint_probs[k2][eval_ds]
            p3, _,      _ = checkpoint_probs[k3][eval_ds]
            mean_p = (p1 + p2 + p3) / 3.0
            try:
                auc = float(roc_auc_score(labels, mean_p))
            except Exception:
                auc = float("nan")
            combo_ds_probs[eval_ds] = (mean_p, labels)
            combo_aucs[eval_ds]     = auc

        if not combo_ds_probs:
            continue

        thr_f1 = _best_threshold_f1(combo_ds_probs)
        thr_f2 = _best_threshold_f2(combo_ds_probs)

        summary = dict(
            combo_name=combo_name,
            model_1=m1, train_1=ds1, model_2=m2, train_2=ds2, model_3=m3, train_3=ds3,
            best_threshold_f1=thr_f1, best_threshold_f2=thr_f2,
        )

        per_ds_f1s, per_ds_f2s, sens_f1_vals = [], [], []
        for eval_ds, (mean_p, labels) in combo_ds_probs.items():
            m_f1 = _metrics_at_threshold(mean_p, labels, thr_f1)
            m_f2 = _metrics_at_threshold(mean_p, labels, thr_f2)
            per_ds_f1s.append(m_f1["f1"]); per_ds_f2s.append(m_f2["f2"])
            sens_f1_vals.append(m_f1["sensitivity"])
            summary[f"auc_{eval_ds}"]         = round(combo_aucs[eval_ds], 4)
            summary[f"f1_{eval_ds}"]          = m_f1["f1"]
            summary[f"f2_{eval_ds}"]          = m_f2["f2"]
            summary[f"sensitivity_{eval_ds}"] = m_f1["sensitivity"]
            summary[f"sens_f2_{eval_ds}"]     = m_f2["sensitivity"]
            summary[f"specificity_{eval_ds}"] = m_f1["specificity"]
            summary[f"spec_f2_{eval_ds}"]     = m_f2["specificity"]
            summary[f"accuracy_{eval_ds}"]    = m_f1["accuracy"]
            summary[f"tp_{eval_ds}"]          = m_f1["tp"]
            summary[f"tn_{eval_ds}"]          = m_f1["tn"]
            summary[f"fp_{eval_ds}"]          = m_f1["fp"]
            summary[f"fn_{eval_ds}"]          = m_f1["fn"]

        summary["avg_f1"] = round(float(np.mean(per_ds_f1s)), 4)
        summary["avg_f2"] = round(float(np.mean(per_ds_f2s)), 4)
        summary["avg_sensitivity"] = round(float(np.mean(sens_f1_vals)), 4)
        # worst-case sensitivity across datasets — the number that actually matters clinically
        summary["min_sensitivity"] = round(float(np.min(sens_f1_vals)), 4)
        summary["avg_specificity"] = round(float(np.mean(
            [summary[f"specificity_{ds}"] for ds in eval_datasets if f"specificity_{ds}" in summary])), 4)
        summary["avg_accuracy"] = round(float(np.mean(
            [summary[f"accuracy_{ds}"] for ds in eval_datasets if f"accuracy_{ds}" in summary])), 4)
        summary["avg_sens_f2"] = round(float(np.mean(
            [summary[f"sens_f2_{ds}"] for ds in eval_datasets if f"sens_f2_{ds}" in summary])), 4)
        summary["avg_spec_f2"] = round(float(np.mean(
            [summary[f"spec_f2_{ds}"] for ds in eval_datasets if f"spec_f2_{ds}" in summary])), 4)
        # ISIC2020 skews heavily benign, so track HAM10000 + ISIC2019 sensitivity on their own too
        ham_s = summary.get("sensitivity_ham10000", 0.0)
        i19_s = summary.get("sensitivity_isic2019", 0.0)
        summary["mean_sens_no2020"] = round((ham_s + i19_s) / 2, 4)
        _fill_thr50(summary, combo_ds_probs, eval_datasets)
        results.append(summary)

    return results, single_results


def main() -> None:
    base_cfg          = load_config()
    device            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits_dir        = Path(base_cfg.paths.data_splits)
    melanoma_root     = Path(base_cfg.paths.melanoma_data)
    ablation_noseg_dir = Path(base_cfg.paths.outputs) / "ablation_noseg"
    eval_xlsx         = ablation_noseg_dir / "sens" / "evaluation_3models_mean_sens.xlsx"
    eval_xlsx.parent.mkdir(parents=True, exist_ok=True)

    gpu_label   = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    n_ckpts     = len(ALL_MODELS) * len(ALL_DATASETS)
    n_combos_est = len(list(combinations(range(n_ckpts), 3)))
    print(f"\n  Device        : {gpu_label}")
    print(f"  Suffix        : _{AUG_MODE}  (sensitivity fine-tuned checkpoints)")
    print(f"  Checkpoints   : {n_ckpts} max  ({len(ALL_MODELS)} models x {len(ALL_DATASETS)} datasets)")
    print(f"  Triplets max  : {n_combos_est}  C({n_ckpts},3)")
    print(f"  Threshold rng : {THRESHOLDS[0]} – {THRESHOLDS[-1]}  (wider for shifted probs)")
    print(f"  Ensemble      : mean of the 3 probabilities, one shared threshold\n")

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
    id_cols = ["combo_name", "model_1", "train_1", "model_2", "train_2", "model_3", "train_3",
               "best_threshold_f1", "best_threshold_f2",
               "avg_f1", "avg_f2", "avg_sensitivity", "min_sensitivity",
               "avg_specificity", "avg_accuracy", "avg_sens_f2", "avg_spec_f2",
               "mean_sens_no2020",
               "avg_f1_thr50", "avg_f2_thr50", "delta_avg_f1", "delta_avg_f2",
               "avg_sensitivity_thr50", "avg_specificity_thr50"]
    extra_cols = [c for c in df.columns if c not in id_cols]
    df = df[[c for c in id_cols if c in df.columns] + extra_cols]

    write_excel_sheet(eval_xlsx, "Summary_F1",
                      df.sort_values("avg_f1", ascending=False))
    write_excel_sheet(eval_xlsx, "Summary_F2",
                      df.sort_values("avg_f2", ascending=False))
    write_excel_sheet(eval_xlsx, "Summary_MinSens",
                      df.sort_values("min_sensitivity", ascending=False))
    write_excel_sheet(eval_xlsx, "Summary_SensNo2020",
                      df.sort_values("mean_sens_no2020", ascending=False))

    if all_singles:
        df_singles = pd.DataFrame(all_singles)
        single_id  = ["model", "train_dataset", "best_threshold_f1", "best_threshold_f2",
                      "avg_f1", "avg_f2", "avg_sensitivity", "avg_specificity", "avg_accuracy",
                      "avg_f1_thr50", "avg_f2_thr50", "delta_avg_f1", "delta_avg_f2"]
        extra_s    = [c for c in df_singles.columns if c not in single_id]
        write_excel_sheet(eval_xlsx, "Single_Models",
                          df_singles[[c for c in single_id if c in df_singles.columns] + extra_s]
                          .sort_values("avg_f1", ascending=False))

    for eval_ds in ALL_DATASETS:
        if f"f1_{eval_ds}" not in df.columns:
            continue
        ds_id   = ["combo_name", "model_1", "train_1", "model_2", "train_2", "model_3", "train_3",
                   "best_threshold_f1", "best_threshold_f2"]
        ds_cols = ds_id + [c for c in df.columns if c.endswith(f"_{eval_ds}")]
        write_excel_sheet(eval_xlsx, eval_ds.upper(),
                          df[[c for c in ds_cols if c in df.columns]]
                          .sort_values(f"sensitivity_{eval_ds}", ascending=False))

    best = (df.groupby("combo_name", as_index=False)
              .agg(avg_f1=("avg_f1", "mean"), avg_f2=("avg_f2", "mean"),
                   avg_sensitivity=("avg_sensitivity", "mean"),
                   min_sensitivity=("min_sensitivity", "mean"),
                   avg_specificity=("avg_specificity", "mean"),
                   avg_accuracy=("avg_accuracy", "mean"),
                   avg_sens_f2=("avg_sens_f2", "mean"))
              .sort_values("avg_f1", ascending=False))
    combo_info = df.drop_duplicates("combo_name")[
        ["combo_name", "model_1", "train_1", "model_2", "train_2", "model_3", "train_3",
         "best_threshold_f1", "best_threshold_f2"]]
    best = best.merge(combo_info, on="combo_name", how="left")
    best_cols = ["combo_name", "model_1", "train_1", "model_2", "train_2", "model_3", "train_3",
                 "best_threshold_f1", "best_threshold_f2",
                 "avg_f1", "avg_f2", "avg_sensitivity", "min_sensitivity",
                 "avg_specificity", "avg_accuracy", "avg_sens_f2"]
    write_excel_sheet(eval_xlsx, "Best_Combos_F1",
                      best[[c for c in best_cols if c in best.columns]])
    write_excel_sheet(eval_xlsx, "Best_Combos_F2",
                      best[[c for c in best_cols if c in best.columns]]
                      .sort_values("avg_f2", ascending=False))

    best_combo_f1  = df.loc[df["avg_f1"].idxmax()]
    best_combo_f2  = df.loc[df["avg_f2"].idxmax()]
    best_combo_sens = df.loc[df["min_sensitivity"].idxmax()]

    sep = "=" * 80
    print(f"\n{sep}")
    print(f"  Saved   : {eval_xlsx}")
    print(f"  Triplets: {len(df)}  (from {len(all_singles)} loaded checkpoints)")
    print(f"\n  ── Best triplet by F1 (thr={best_combo_f1['best_threshold_f1']:.2f}) ──")
    print(f"     {best_combo_f1['combo_name']}")
    print(f"     avg_f1={best_combo_f1['avg_f1']:.4f}  avg_f2={best_combo_f1['avg_f2']:.4f}  "
          f"avg_sens={best_combo_f1['avg_sensitivity']:.4f}  avg_spec={best_combo_f1['avg_specificity']:.4f}")
    for ds in ALL_DATASETS:
        if f"f1_{ds}" in best_combo_f1:
            print(f"     {ds.upper():<12} F1={best_combo_f1[f'f1_{ds}']:.4f}  "
                  f"sens={best_combo_f1[f'sensitivity_{ds}']:.4f}  "
                  f"spec={best_combo_f1[f'specificity_{ds}']:.4f}")
    print(f"\n  ── Best triplet by F2 (thr={best_combo_f2['best_threshold_f2']:.2f}) ──")
    print(f"     {best_combo_f2['combo_name']}")
    print(f"     avg_f1={best_combo_f2['avg_f1']:.4f}  avg_f2={best_combo_f2['avg_f2']:.4f}  "
          f"avg_sens={best_combo_f2['avg_sens_f2']:.4f}  avg_spec={best_combo_f2['avg_spec_f2']:.4f}")
    for ds in ALL_DATASETS:
        if f"f2_{ds}" in best_combo_f2:
            print(f"     {ds.upper():<12} F2={best_combo_f2[f'f2_{ds}']:.4f}  "
                  f"sens={best_combo_f2.get(f'sens_f2_{ds}', float('nan')):.4f}  "
                  f"spec={best_combo_f2.get(f'spec_f2_{ds}', float('nan')):.4f}")
    print(f"\n  ── Best triplet by worst-case sensitivity ──")
    print(f"     {best_combo_sens['combo_name']}")
    print(f"     min_sens={best_combo_sens['min_sensitivity']:.4f}  "
          f"mean_sens_no2020={best_combo_sens['mean_sens_no2020']:.4f}  "
          f"avg_f2={best_combo_sens['avg_f2']:.4f}")

    print(f"\n  Top 5 by min_sensitivity:\n")
    top5 = df.sort_values("min_sensitivity", ascending=False).head(5)
    print(top5[["combo_name", "best_threshold_f1", "min_sensitivity",
                "avg_sensitivity", "avg_f1", "avg_f2"]].to_string(index=False))
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
