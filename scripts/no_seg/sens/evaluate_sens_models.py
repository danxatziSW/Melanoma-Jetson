from __future__ import annotations

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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.models.registry import build_model, uses_metadata
from src.utils.config import load_config
from src.utils.io import resolve_dataset_paths, write_excel_sheet

ALL_DATASETS = ["ham10000", "isic2019", "isic2020"]
ALL_MODELS   = [
    "resnet50", "efficientnet_b2", "mobilenetv3_large",
    "medfusionnet", "yolov8_cls",
]

AUG_MODE  = "none_sens"
BETA      = 2.0
THR_RANGE = np.round(np.arange(0.20, 0.86, 0.01), 2)

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


def _run_inference(
    model: nn.Module, loader: DataLoader,
    device: torch.device, with_meta: bool,
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


def _metrics_at_thr(probs: np.ndarray, labels: np.ndarray, thr: float) -> dict:
    preds = (probs >= thr).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    b2   = BETA ** 2
    f2   = (1 + b2) * prec * sens / max(b2 * prec + sens, 1e-9)
    f1   = 2 * prec * sens / max(prec + sens, 1e-9)
    acc  = (tp + tn) / max(tp + tn + fp + fn, 1)
    return dict(
        sensitivity=round(sens, 4), specificity=round(spec, 4),
        precision=round(prec, 4), f2=round(f2, 4), f1=round(f1, 4),
        accuracy=round(acc, 4), tp=tp, tn=tn, fp=fp, fn=fn,
    )


def _best_f2_threshold(
    ds_probs: dict[str, tuple[np.ndarray, np.ndarray]]
) -> float:
    best_thr, best_score = THR_RANGE[0], -1.0
    for thr in THR_RANGE:
        score = float(np.mean([_metrics_at_thr(p, l, thr)["f2"] for p, l in ds_probs.values()]))
        if score > best_score:
            best_score, best_thr = score, thr
    return float(best_thr)


def main() -> None:
    base_cfg     = load_config()
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits_dir   = Path(base_cfg.paths.data_splits)
    melanoma_root = Path(base_cfg.paths.melanoma_data)
    ablation_dir = Path(base_cfg.paths.outputs) / "ablation_noseg"
    out_xlsx     = ablation_dir / "sens" / "evaluation_sens_models.xlsx"

    gpu_label = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    total = len(ALL_DATASETS) * len(ALL_MODELS)
    print(f"\n  Single-model evaluation — {AUG_MODE} checkpoints")
    print(f"  Threshold : F{int(BETA)}-optimal (sweep {THR_RANGE[0]:.2f}–{THR_RANGE[-1]:.2f})")
    print(f"  Device    : {gpu_label}")
    print(f"  Models    : {total}  ({len(ALL_MODELS)} architectures × {len(ALL_DATASETS)} train datasets)\n")

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

    summary_rows: list[dict] = []
    detail_rows:  list[dict] = []
    idx = 0

    for train_ds in ALL_DATASETS:
        for model_name in ALL_MODELS:
            idx += 1
            run_id    = f"{model_name}_{AUG_MODE}"
            ckpt_file = ablation_dir / train_ds / run_id / "checkpoints" / f"{run_id}.pt"

            if not ckpt_file.exists():
                print(f"  [{idx}/{total}] [SKIP] {model_name}/{train_ds} — checkpoint not found")
                continue

            print(f"  [{idx}/{total}] {model_name}/{train_ds} ...", end="", flush=True)
            config    = load_config(model_name)
            with_meta = uses_metadata(model_name)
            inp_size  = getattr(config, "input_size", 224)
            nw        = getattr(config, "num_workers", 0)

            model = build_model(model_name, config, num_classes=2)
            model.load_state_dict(torch.load(ckpt_file, map_location=device))
            model.to(device).eval()

            all_probs: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
            for eval_ds, eval_df in test_dfs.items():
                ds_obj = EvalDataset(eval_df, inp_size, with_meta=with_meta)
                loader = DataLoader(ds_obj, batch_size=config.batch_size, shuffle=False,
                                    num_workers=nw, pin_memory=(nw > 0))
                probs, labels = _run_inference(model, loader, device, with_meta)
                try:
                    auc = float(roc_auc_score(labels, probs))
                except Exception:
                    auc = float("nan")
                all_probs[eval_ds] = (probs, labels, auc)

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

            thr = _best_f2_threshold({ds: (p, l) for ds, (p, l, _) in all_probs.items()})

            row: dict = dict(
                train_dataset=train_ds,
                model=model_name,
                f2_threshold=thr,
            )
            sens_vals, f2_vals = [], []
            for eval_ds, (probs, labels, auc) in all_probs.items():
                m = _metrics_at_thr(probs, labels, thr)
                row[f"auc_{eval_ds}"]         = round(auc, 4)
                row[f"sensitivity_{eval_ds}"] = m["sensitivity"]
                row[f"specificity_{eval_ds}"] = m["specificity"]
                row[f"f2_{eval_ds}"]          = m["f2"]
                row[f"f1_{eval_ds}"]          = m["f1"]
                row[f"accuracy_{eval_ds}"]    = m["accuracy"]
                row[f"tp_{eval_ds}"]          = m["tp"]
                row[f"tn_{eval_ds}"]          = m["tn"]
                row[f"fp_{eval_ds}"]          = m["fp"]
                row[f"fn_{eval_ds}"]          = m["fn"]
                sens_vals.append(m["sensitivity"])
                f2_vals.append(m["f2"])

                detail_rows.append(dict(
                    train_dataset=train_ds, model=model_name,
                    eval_dataset=eval_ds, f2_threshold=thr,
                    auc=round(auc, 4), **m,
                ))

            row["avg_sensitivity"]  = round(float(np.mean(sens_vals)), 4)
            row["avg_f2"]           = round(float(np.mean(f2_vals)), 4)
            row["min_sensitivity"]  = round(float(np.min(sens_vals)), 4)
            ham_s = row.get("sensitivity_ham10000", 0.0)
            i19_s = row.get("sensitivity_isic2019", 0.0)
            row["mean_sens_no2020"] = round((ham_s + i19_s) / 2, 4)
            row["mean_f2_no2020"]   = round(
                (row.get("f2_ham10000", 0.0) + row.get("f2_isic2019", 0.0)) / 2, 4
            )
            summary_rows.append(row)

            aucs = "  ".join(
                f"{ds}={all_probs[ds][2]:.3f}" for ds in ALL_DATASETS if ds in all_probs
            )
            print(f"  thr={thr:.2f}  {aucs}")

    if not summary_rows:
        print("\n  No results — no checkpoints found.\n")
        return

    df = pd.DataFrame(summary_rows)

    id_cols = [
        "train_dataset", "model", "f2_threshold",
        "avg_f2", "avg_sensitivity", "min_sensitivity",
        "mean_sens_no2020", "mean_f2_no2020",
    ]
    extra = [c for c in df.columns if c not in id_cols]
    df = df[id_cols + extra]

    write_excel_sheet(out_xlsx, "Summary",
                      df.sort_values("mean_sens_no2020", ascending=False))

    df_det = pd.DataFrame(detail_rows)
    for eval_ds in ALL_DATASETS:
        sub = df_det[df_det["eval_dataset"] == eval_ds].copy()
        if sub.empty:
            continue
        write_excel_sheet(out_xlsx, eval_ds.upper(),
                          sub.sort_values("sensitivity", ascending=False))

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  Results saved : {out_xlsx}")
    print(f"  Sheets        : Summary | {' | '.join(ds.upper() for ds in ALL_DATASETS)}")
    print(f"  Total models  : {len(df)}")
    print(f"{sep}\n")

    top_cols = ["train_dataset", "model",
                "sensitivity_ham10000", "sensitivity_isic2019", "sensitivity_isic2020",
                "mean_sens_no2020", "min_sensitivity", "f2_threshold"]
    top_cols = [c for c in top_cols if c in df.columns]
    print("  Top 10 by mean_sens_no2020:\n")
    print(df.sort_values("mean_sens_no2020", ascending=False).head(10)[top_cols].to_string(index=False))
    print()


if __name__ == "__main__":
    main()
