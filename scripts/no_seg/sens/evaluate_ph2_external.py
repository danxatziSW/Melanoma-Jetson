"""External validation of the deployed meta-learner on PH2.

PH2 comes from a different hospital and dermoscope and was not used in training,
validation or threshold selection. Loads meta_learner.pkl as-is with no refitting,
scores four configurations at its global threshold, and splits errors by subtype.

  python scripts/no_seg/sens/evaluate_ph2_external.py
"""
from __future__ import annotations

import pickle
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
from src.utils.io import write_excel_sheet

MODEL1, DATASET1 = "resnet50",     "isic2019"
MODEL2, DATASET2 = "medfusionnet", "isic2020"
AUG_MODE         = "none_sens"
BETA             = 2.0

_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)


class EvalDataset(Dataset):
    """PH2 has no metadata columns, so the neutral defaults apply."""

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
            n = len(self.df)
            self.age      = np.zeros(n, dtype=np.float32)
            self.sex      = np.full(n, 0.5, dtype=np.float32)
            self.site_ohe = np.zeros((n, 6), dtype=np.float32)

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


def _load_ph2(melanoma_root: Path) -> pd.DataFrame:
    ph2_root = melanoma_root / "PH2"
    df = pd.read_csv(ph2_root / "labels.csv")
    df["binary_label"] = df["is_melanoma"].astype(int)
    images_root = ph2_root / "PH2 Dataset images"
    df["image_path"] = df["image_name"].apply(
        lambda name: str(images_root / name / f"{name}_Dermoscopic_Image" / f"{name}.bmp")
    )
    return df.reset_index(drop=True)


def _run_inference(model: nn.Module, loader: DataLoader,
                   device: torch.device, with_meta: bool) -> np.ndarray:
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in tqdm(loader, leave=False, unit="batch", dynamic_ncols=True, file=sys.stdout):
            if with_meta:
                imgs, mdata, _ = batch
                logits = model(imgs.to(device), mdata.to(device))
            else:
                imgs, _ = batch
                logits = model(imgs.to(device))
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
    return np.array(all_probs)


def _metrics(probs: np.ndarray, labels: np.ndarray, thr: float) -> dict:
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
    return dict(sensitivity=round(sens, 4), specificity=round(spec, 4),
                precision=round(prec, 4), f2=round(f2, 4), f1=round(f1, 4),
                tp=tp, tn=tn, fp=fp, fn=fn, threshold=round(thr, 2))


def _auc(probs: np.ndarray, labels: np.ndarray) -> float:
    try:
        return round(float(roc_auc_score(labels, probs)), 4)
    except Exception:
        return float("nan")


def main() -> None:
    base_cfg      = load_config()
    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    melanoma_root = Path(base_cfg.paths.melanoma_data)
    ablation_dir  = Path(base_cfg.paths.outputs) / "ablation_noseg"
    meta_pkl_path = ablation_dir / "meta" / "deployment" / "meta_learner.pkl"
    out_dir       = ablation_dir / "meta" / "external_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_xlsx      = out_dir / "evaluation_ph2.xlsx"

    if not meta_pkl_path.exists():
        raise FileNotFoundError(f"Deployed meta-learner not found: {meta_pkl_path}")
    with open(meta_pkl_path, "rb") as f:
        meta_bundle = pickle.load(f)
    scaler, clf = meta_bundle["scaler"], meta_bundle["clf"]
    global_thr  = meta_bundle["thresholds"]["global"]

    print(f"\n  External validation on PH2 (held-out, unseen dataset)")
    print(f"  Model 1 : {MODEL1} / {DATASET1}")
    print(f"  Model 2 : {MODEL2} / {DATASET2}")
    print(f"  Deployed meta-learner : {meta_pkl_path}")
    print(f"  Global threshold      : {global_thr}\n")

    df = _load_ph2(melanoma_root)
    labels = df["binary_label"].values
    print(f"  PH2  n={len(df)}  melanoma={int(labels.sum())}  "
          f"({df['diagnosis'].value_counts().to_dict()})\n")

    probs: dict[str, np.ndarray] = {}
    for model_name, dataset_name in [(MODEL1, DATASET1), (MODEL2, DATASET2)]:
        key    = f"{model_name}/{dataset_name}"
        run_id = f"{model_name}_{AUG_MODE}"
        ckpt_f = ablation_dir / dataset_name / run_id / "checkpoints" / f"{run_id}.pt"
        if not ckpt_f.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_f}")

        print(f"  Loading {key} ...")
        config = load_config(model_name)
        wm     = uses_metadata(model_name)
        model  = build_model(model_name, config, num_classes=2)
        model.load_state_dict(torch.load(ckpt_f, map_location=device, weights_only=True))
        model.to(device).eval().requires_grad_(False)

        inp, nw = getattr(config, "input_size", 224), getattr(config, "num_workers", 0)
        loader  = DataLoader(EvalDataset(df, inp, with_meta=wm),
                             batch_size=config.batch_size, shuffle=False,
                             num_workers=nw, pin_memory=(nw > 0))
        probs[key] = _run_inference(model, loader, device, wm)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    k1, k2 = f"{MODEL1}/{DATASET1}", f"{MODEL2}/{DATASET2}"
    p1, p2 = probs[k1], probs[k2]
    mean_prob = (p1 + p2) / 2
    meta_prob = clf.predict_proba(scaler.transform(np.column_stack([p1, p2])))[:, 1]

    rows = []
    sep = "-" * 72
    print(f"\n  {'Approach':<22}  {'Sens':>6}  {'Spec':>6}  {'F2':>6}  {'AUC':>6}  {'Thr':>5}")
    print(f"  {sep}")
    for approach, prob in [
        (f"{MODEL1}/{DATASET1}", p1),
        (f"{MODEL2}/{DATASET2}", p2),
        ("mean ensemble",        mean_prob),
        ("meta-learner (deployed)", meta_prob),
    ]:
        m   = _metrics(prob, labels, global_thr)
        auc = _auc(prob, labels)
        print(f"  {approach:<22}  {m['sensitivity']:>6.4f}  {m['specificity']:>6.4f}  "
              f"{m['f2']:>6.4f}  {auc:>6.4f}  {m['threshold']:>5.2f}")
        rows.append(dict(
            approach=approach, sensitivity=m["sensitivity"], specificity=m["specificity"],
            precision=m["precision"], f2=m["f2"], f1=m["f1"], auc=auc,
            threshold=m["threshold"], tp=m["tp"], tn=m["tn"], fp=m["fp"], fn=m["fn"],
        ))
    print(f"  {sep}\n")

    preds = (meta_prob >= global_thr).astype(int)
    subtype_rows = []
    print(f"  Meta-learner predictions by PH2 diagnosis subtype:")
    print(f"  {'Diagnosis':<16}  {'n':>4}  {'Predicted melanoma':>19}  {'Rate':>6}")
    for dx in ["Common Nevus", "Atypical Nevus", "Melanoma"]:
        mask = (df["diagnosis"] == dx).values
        n = int(mask.sum())
        flagged = int(preds[mask].sum())
        rate = flagged / max(n, 1)
        print(f"  {dx:<16}  {n:>4}  {flagged:>19}  {rate:>6.2%}")
        subtype_rows.append(dict(diagnosis=dx, n=n, predicted_melanoma=flagged, rate=round(rate, 4)))
    print()

    df_all     = pd.DataFrame(rows)
    df_subtype = pd.DataFrame(subtype_rows)
    write_excel_sheet(out_xlsx, "PH2", df_all)
    write_excel_sheet(out_xlsx, "PH2_by_subtype", df_subtype)
    print(f"  Results saved -> {out_xlsx}\n")


if __name__ == "__main__":
    main()
