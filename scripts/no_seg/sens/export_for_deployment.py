from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
import torch.nn as nn
import albumentations as A
import cv2
import pandas as pd
from albumentations.pytorch import ToTensorV2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.models.registry import build_model, uses_metadata
from src.utils.config import load_config
from src.utils.io import resolve_dataset_paths

MODEL1, DATASET1 = "resnet50",    "isic2019"
MODEL2, DATASET2 = "medfusionnet", "isic2020"
AUG_MODE         = "none_sens"

BETA      = 2.0
THR_RANGE = np.round(np.arange(0.20, 0.86, 0.01), 2)
ALL_DATASETS = ["ham10000", "isic2019", "isic2020"]

_ONNX_OPSET  = 17
_METADATA_DIM = 8
_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)
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


def _load_df(splits_dir: Path, csv_name: str, dataset_source: str, melanoma_root: Path) -> pd.DataFrame:
    df = pd.read_csv(splits_dir / csv_name)
    df = df[df["dataset_source"] == dataset_source].copy()
    df["binary_label"] = (df["label_str"] == "mel").astype(int)
    df = resolve_dataset_paths(df, melanoma_root)
    return df


def _run_inference(model, loader, device, with_meta):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  infer", leave=False, unit="batch",
                          dynamic_ncols=True, file=sys.stdout):
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


def _best_thr(probs, labels):
    best_thr, best = THR_RANGE[0], -1.0
    for thr in THR_RANGE:
        preds = (probs >= thr).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        sens = tp / max(tp + fn, 1)
        prec = tp / max(tp + fp, 1)
        b2 = BETA ** 2
        f2 = (1 + b2) * prec * sens / max(b2 * prec + sens, 1e-9)
        if f2 > best:
            best, best_thr = f2, thr
    return float(best_thr)


def _export_onnx(model, onnx_path: Path, with_meta: bool, input_size: int = 224) -> None:
    model.eval()
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_img = torch.zeros(1, 3, input_size, input_size)
    kwargs = dict(opset_version=_ONNX_OPSET, do_constant_folding=True, dynamo=False)
    if with_meta:
        dummy_meta = torch.zeros(1, _METADATA_DIM)
        torch.onnx.export(model, (dummy_img, dummy_meta), str(onnx_path),
                          input_names=["image", "metadata"], output_names=["logits"], **kwargs)
    else:
        torch.onnx.export(model, dummy_img, str(onnx_path),
                          input_names=["image"], output_names=["logits"], **kwargs)


def main() -> None:
    base_cfg     = load_config()
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits_dir   = Path(base_cfg.paths.data_splits)
    melanoma_root = Path(base_cfg.paths.melanoma_data)
    ablation_dir = Path(base_cfg.paths.outputs) / "ablation_noseg"
    out_dir      = ablation_dir / "meta" / "deployment"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Exporting deployment artifacts for:")
    print(f"    Model 1 : {MODEL1} trained on {DATASET1}")
    print(f"    Model 2 : {MODEL2} trained on {DATASET2}")
    print(f"  Output dir : {out_dir}\n")

    val_dfs: dict[str, pd.DataFrame] = {}
    for ds in ALL_DATASETS:
        try:
            val_dfs[ds] = _load_df(splits_dir, "cls_val.csv", ds, melanoma_root)
        except Exception as exc:
            print(f"  [WARN] could not load val split for {ds}: {exc}")

    val_probs: dict[str, np.ndarray] = {}

    for model_name, dataset_name in [(MODEL1, DATASET1), (MODEL2, DATASET2)]:
        run_id   = f"{model_name}_{AUG_MODE}"
        ckpt_f   = ablation_dir / dataset_name / run_id / "checkpoints" / f"{run_id}.pt"
        onnx_out = out_dir / f"{run_id}.onnx"

        print(f"  [{model_name}/{dataset_name}]")

        if not ckpt_f.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_f}")

        config = load_config(model_name)
        wm     = uses_metadata(model_name)
        model  = build_model(model_name, config, num_classes=2)
        model.load_state_dict(torch.load(ckpt_f, map_location=device))
        model.to(device).eval().requires_grad_(False)

        # export ONNX (CPU required for export)
        print(f"    -> ONNX ... ", end="", flush=True)
        model_cpu = model.cpu()
        inp_size  = getattr(config, "input_size", 224)
        _export_onnx(model_cpu, onnx_out, wm, input_size=inp_size)
        model.to(device)
        print(f"done  [{onnx_out.stat().st_size // 1024} KB]  →  {onnx_out.name}")

        print(f"    -> val inference for meta-learner fitting ...")
        all_p: list[np.ndarray] = []
        all_l: list[np.ndarray] = []
        per_ds_p: dict[str, np.ndarray] = {}
        for ds in ALL_DATASETS:
            if ds not in val_dfs:
                continue
            loader = DataLoader(
                EvalDataset(val_dfs[ds], inp_size, with_meta=wm),
                batch_size=config.batch_size, shuffle=False,
                num_workers=getattr(config, "num_workers", 0),
            )
            p, l = _run_inference(model, loader, device, wm)
            per_ds_p[ds] = p
            all_p.append(p)
            all_l.append(l)

        val_probs[f"{model_name}/{dataset_name}"] = per_ds_p

        del model
        torch.cuda.empty_cache()
        print()

    print("  Fitting meta-learner ...")
    k1 = f"{MODEL1}/{DATASET1}"
    k2 = f"{MODEL2}/{DATASET2}"

    X_parts, y_parts = [], []
    for ds in ALL_DATASETS:
        if ds not in val_dfs:
            continue
        p1 = val_probs[k1].get(ds)
        p2 = val_probs[k2].get(ds)
        if p1 is None or p2 is None:
            continue
        X_parts.append(np.column_stack([p1, p2]))
        y_parts.append(val_dfs[ds]["binary_label"].values)

    X_meta   = np.vstack(X_parts)
    y_meta   = np.concatenate(y_parts)
    scaler   = StandardScaler()
    X_meta_s = scaler.fit_transform(X_meta)
    clf      = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced")
    clf.fit(X_meta_s, y_meta)

    print(f"    weights : [{clf.coef_[0][0]:.4f}, {clf.coef_[0][1]:.4f}]")
    print(f"    bias    : {clf.intercept_[0]:.4f}")

    thresholds: dict[str, float] = {}
    print("\n  Computing per-dataset thresholds (F2-optimal on val):")
    for ds in ALL_DATASETS:
        if ds not in val_dfs:
            continue
        p1 = val_probs[k1].get(ds)
        p2 = val_probs[k2].get(ds)
        if p1 is None or p2 is None:
            continue
        X_val_s   = scaler.transform(np.column_stack([p1, p2]))
        meta_prob = clf.predict_proba(X_val_s)[:, 1]
        labels    = val_dfs[ds]["binary_label"].values
        thr       = _best_thr(meta_prob, labels)
        thresholds[ds] = thr
        print(f"    {ds:<12} threshold = {thr}")

    # global threshold (conservative — maximise sensitivity across all datasets)
    all_val_p1 = np.concatenate([val_probs[k1][ds] for ds in ALL_DATASETS if ds in val_probs[k1]])
    all_val_p2 = np.concatenate([val_probs[k2][ds] for ds in ALL_DATASETS if ds in val_probs[k2]])
    all_labels = np.concatenate([val_dfs[ds]["binary_label"].values for ds in ALL_DATASETS if ds in val_dfs])
    X_all_s    = scaler.transform(np.column_stack([all_val_p1, all_val_p2]))
    all_meta_p = clf.predict_proba(X_all_s)[:, 1]
    global_thr = _best_thr(all_meta_p, all_labels)
    thresholds["global"] = global_thr
    print(f"    {'global':<12} threshold = {global_thr}  (use this when dataset is unknown)")

    meta_path = out_dir / "meta_learner.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump({
            "model1": f"{MODEL1}/{DATASET1}",
            "model2": f"{MODEL2}/{DATASET2}",
            "scaler": scaler,
            "clf":    clf,
            "thresholds": thresholds,
        }, f)
    print(f"\n  Saved meta-learner → {meta_path.name}")

    sep = "=" * 65
    onnx1 = f"{MODEL1}_{AUG_MODE}.onnx"
    onnx2 = f"{MODEL2}_{AUG_MODE}.onnx"
    print(f"\n{sep}")
    print(f"  Files to copy to Jetson (from {out_dir}):")
    print(f"    {onnx1}")
    print(f"    {onnx2}")
    print(f"    meta_learner.pkl")
    print(f"\n  On the Jetson, run:")
    print(f"    trtexec --onnx={onnx1} --saveEngine=resnet50_none_sens.engine --fp16")
    print(f"    trtexec --onnx={onnx2} --saveEngine=medfusionnet_none_sens.engine --fp16")
    print(f"\n  trtexec is at: /usr/src/tensorrt/bin/trtexec")
    print(f"  (add to PATH or use the full path)")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
