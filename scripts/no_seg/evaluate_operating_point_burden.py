"""Confusion matrix and referral burden at the deployed operating point.

Scores each dataset twice from meta_learner.pkl without refitting: at the global
threshold 0.85 and at the pkl's per-dataset value. Tables 6/7 claim 0.85 for every
row, but the pkl stores 0.74 for isic2019 and 0.20 for isic2020.

  python scripts/no_seg/evaluate_operating_point_burden.py [--neutral-meta] [--refresh-cache]
"""
from __future__ import annotations

import argparse
import math
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from scipy.stats import beta as beta_dist
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.models.registry import build_model, uses_metadata
from src.utils.config import load_config
from src.utils.io import resolve_dataset_paths, write_excel_sheet

MODEL1, DATASET1 = "resnet50",     "isic2019"
MODEL2, DATASET2 = "medfusionnet", "isic2020"
AUG_MODE         = "none_sens"
ISIC_DATASETS    = ["ham10000", "isic2019", "isic2020"]
BETA             = 2.0

PREVALENCE_SWEEP = [
    (0.001, "Population screening"),
    (0.005, "General practice"),
    (0.021, "ISIC 2020 (actual)"),
    (0.050, "Primary-care referral"),
    (0.100, "Specialist clinic"),
    (0.200, "~ISIC 2019"),
    (0.500, "Balanced (reference)"),
]

_MEAN      = (0.485, 0.456, 0.406)
_STD       = (0.229, 0.224, 0.225)
_SITE_CATS = ["head/neck", "upper extremity", "lower extremity",
              "torso", "palms/soles", "oral/genital"]


class EvalDataset(Dataset):
    """`neutral_meta` feeds the Section 3.7 deployment vector instead of real metadata."""

    def __init__(self, df: pd.DataFrame, input_size: int,
                 with_meta: bool = False, neutral_meta: bool = False):
        self.df           = df.reset_index(drop=True)
        self.with_meta    = with_meta
        self.neutral_meta = neutral_meta
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
        n  = len(df)
        if self.neutral_meta:
            self.age      = np.full(n, 0.5, dtype=np.float32)
            self.sex      = np.full(n, 0.5, dtype=np.float32)
            self.site_ohe = np.zeros((n, len(_SITE_CATS)), dtype=np.float32)
            return

        age_col = "age_approx" if "age_approx" in df.columns else None
        self.age = (
            (df[age_col].fillna(df[age_col].median()) / 100.0).values.astype(np.float32)
            if age_col else np.zeros(n, dtype=np.float32)
        )
        sex_col = "sex" if "sex" in df.columns else None
        self.sex = (
            df[sex_col].map({"male": 1.0, "female": 0.0}).fillna(0.5).values.astype(np.float32)
            if sex_col else np.full(n, 0.5, dtype=np.float32)
        )
        site_col = "anatom_site_general_challenge" if "anatom_site_general_challenge" in df.columns else None
        self.site_ohe = np.zeros((n, len(_SITE_CATS)), dtype=np.float32)
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


def _load_isic_df(splits_dir: Path, dataset_source: str, melanoma_root: Path) -> pd.DataFrame:
    df = pd.read_csv(splits_dir / "cls_test.csv")
    df = df[df["dataset_source"] == dataset_source].copy()
    df["binary_label"] = (df["label_str"] == "mel").astype(int)
    df = resolve_dataset_paths(df, melanoma_root)
    return df.reset_index(drop=True)


def _load_ph2_df(melanoma_root: Path) -> pd.DataFrame:
    ph2_root    = melanoma_root / "PH2"
    df          = pd.read_csv(ph2_root / "labels.csv")
    images_root = ph2_root / "PH2 Dataset images"
    df["binary_label"] = df["is_melanoma"].astype(int)
    df["image_path"]   = df["image_name"].apply(
        lambda name: str(images_root / name / f"{name}_Dermoscopic_Image" / f"{name}.bmp")
    )
    return df.reset_index(drop=True)


def _run_inference(model: nn.Module, loader: DataLoader,
                   device: torch.device, with_meta: bool) -> np.ndarray:
    model.eval()
    all_probs: list[float] = []
    with torch.no_grad():
        for batch in tqdm(loader, leave=False, unit="batch", dynamic_ncols=True, file=sys.stdout):
            if with_meta:
                imgs, mdata, _ = batch
                logits = model(imgs.to(device), mdata.to(device))
            else:
                imgs, _ = batch
                logits = model(imgs.to(device))
            all_probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.array(all_probs)


def _confusion(probs: np.ndarray, labels: np.ndarray, thr: float) -> dict:
    """Confusion matrix plus derived rates."""
    preds = (probs >= thr).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv  = tp / (tp + fp) if (tp + fp) else float("nan")   # == precision
    npv  = tn / (tn + fn) if (tn + fn) else float("nan")

    b2 = BETA ** 2
    f1 = 2 * ppv * sens / (ppv + sens) if (ppv + sens) else 0.0
    f2 = (1 + b2) * ppv * sens / (b2 * ppv + sens) if (b2 * ppv + sens) else 0.0

    return dict(threshold=round(float(thr), 2), tp=tp, fn=fn, fp=fp, tn=tn,
                sensitivity=sens, specificity=spec, precision=ppv, ppv=ppv, npv=npv,
                f1=f1, f2=f2, fp_per_tp=(fp / tp if tp else float("inf")))


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    phat   = k / n
    denom  = 1 + z ** 2 / n
    center = (phat + z ** 2 / (2 * n)) / denom
    half   = (z * math.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    lo = beta_dist.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta_dist.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return (float(lo), float(hi))


def _sensitivity_ci(tp: int, fn: int) -> tuple[str, float, float]:
    n = tp + fn
    if n == 0:
        return "n/a", float("nan"), float("nan")
    if tp == n:
        lo, hi = _clopper_pearson_ci(tp, n)
        return "Clopper-Pearson (exact)", lo, hi
    lo, hi = _wilson_ci(tp, n)
    return "Wilson score", lo, hi


def _prevalence_sweep(sens: float, spec: float, per: int = 1000) -> list[dict]:
    """Expected counts per `per` people screened, varying only prevalence."""
    rows = []
    for pi, setting in PREVALENCE_SWEEP:
        mel  = per * pi
        tp   = mel * sens
        fn   = mel - tp
        fp   = per * (1 - pi) * (1 - spec)
        refs = tp + fp
        ppv  = tp / refs if refs else float("nan")
        tn   = per * (1 - pi) * spec
        npv  = tn / (tn + fn) if (tn + fn) else float("nan")
        f2   = (5 * ppv * sens) / (4 * ppv + sens) if (4 * ppv + sens) else 0.0
        rows.append(dict(prevalence=pi, setting=setting,
                         melanomas=round(mel, 1), tp=round(tp, 1), fn=round(fn, 1),
                         fp=round(fp, 1), referrals=round(refs, 1),
                         ppv=round(ppv, 4), npv=round(npv, 4), f2=round(f2, 3),
                         fp_per_tp=round(fp / tp, 1) if tp else float("inf")))
    return rows


def _gather_probabilities(cfg, device, splits_dir: Path, melanoma_root: Path,
                          ablation_dir: Path, neutral_meta: bool) -> tuple[dict, dict]:
    """Runs both deployed checkpoints over every test set. Returns (probs, labels)."""
    dfs = {ds: _load_isic_df(splits_dir, ds, melanoma_root) for ds in ISIC_DATASETS}
    dfs["ph2"] = _load_ph2_df(melanoma_root)

    labels = {ds: df["binary_label"].values for ds, df in dfs.items()}
    probs: dict[str, dict[str, np.ndarray]] = {}

    for model_name, dataset_name in [(MODEL1, DATASET1), (MODEL2, DATASET2)]:
        key    = f"{model_name}/{dataset_name}"
        run_id = f"{model_name}_{AUG_MODE}"
        ckpt_f = ablation_dir / dataset_name / run_id / "checkpoints" / f"{run_id}.pt"
        if not ckpt_f.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_f}")

        print(f"  Loading {key}  <-  {ckpt_f.relative_to(ablation_dir.parent)}")
        config = load_config(model_name)
        wm     = uses_metadata(model_name)
        model  = build_model(model_name, config, num_classes=2)
        model.load_state_dict(torch.load(ckpt_f, map_location=device, weights_only=True))
        model.to(device).eval().requires_grad_(False)
        inp = getattr(config, "input_size", 224)
        nw  = getattr(config, "num_workers", 0)

        per_ds = {}
        for ds, df in dfs.items():
            loader = DataLoader(
                EvalDataset(df, inp, with_meta=wm,
                            neutral_meta=(neutral_meta or ds == "ph2")),
                batch_size=config.batch_size, shuffle=False,
                num_workers=nw, pin_memory=(nw > 0),
            )
            per_ds[ds] = _run_inference(model, loader, device, wm)
        probs[key] = per_ds

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return probs, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--neutral-meta", action="store_true",
                        help="Feed MedFusionNet the fixed deployment metadata vector "
                             "(Section 3.7) instead of real patient metadata.")
    parser.add_argument("--refresh-cache", action="store_true",
                        help="Re-run inference even if a probability cache exists.")
    args = parser.parse_args()

    base_cfg      = load_config()
    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits_dir    = Path(base_cfg.paths.data_splits)
    melanoma_root = Path(base_cfg.paths.melanoma_data)
    ablation_dir  = Path(base_cfg.paths.outputs) / "ablation_noseg"
    deploy_dir    = ablation_dir / "meta" / "deployment"
    out_dir       = ablation_dir / "meta" / "operating_point"
    out_dir.mkdir(parents=True, exist_ok=True)

    tag         = "neutralmeta" if args.neutral_meta else "realmeta"
    probs_cache = out_dir / f"_test_probs_{tag}.pkl"
    out_xlsx    = out_dir / f"operating_point_burden_{tag}.xlsx"

    with open(deploy_dir / "meta_learner.pkl", "rb") as f:
        deployed = pickle.load(f)
    scaler, clf, thresholds = deployed["scaler"], deployed["clf"], deployed["thresholds"]
    global_thr = float(thresholds["global"])

    print(f"\n{'=' * 82}")
    print("  Deployed operating point: confusion matrix and referral burden")
    print(f"{'=' * 82}")
    print(f"  Artifact      : {deploy_dir / 'meta_learner.pkl'}")
    print(f"  Base models   : {deployed['model1']}  +  {deployed['model2']}")
    print(f"  Meta-learner  : coef={clf.coef_[0].round(4).tolist()}  bias={clf.intercept_[0]:.4f}")
    print(f"  Thresholds    : {thresholds}")
    print(f"  Metadata mode : {'fixed neutral vector (Section 3.7)' if args.neutral_meta else 'real patient metadata'}")
    print()
    print("  NOTE: the manuscript states Tables 6/7 use the global threshold "
          f"({global_thr}) for every")
    print("  row. The pkl stores different per-dataset thresholds, and the scripts that")
    print("  produced the published numbers used those. Both readings are reported below.")
    print()

    if probs_cache.exists() and not args.refresh_cache:
        print(f"  [CACHE] loading {probs_cache.name}  (--refresh-cache to recompute)\n")
        with open(probs_cache, "rb") as f:
            cached = pickle.load(f)
        probs, labels = cached["probs"], cached["labels"]
    else:
        probs, labels = _gather_probabilities(
            base_cfg, device, splits_dir, melanoma_root, ablation_dir, args.neutral_meta)
        with open(probs_cache, "wb") as f:
            pickle.dump({"probs": probs, "labels": labels}, f)
        print(f"\n  [CACHE] saved -> {probs_cache.name}\n")

    k1, k2 = f"{MODEL1}/{DATASET1}", f"{MODEL2}/{DATASET2}"

    cm_rows, sweep_rows, compare_rows = [], [], []

    for ds in ["ham10000", "isic2019", "isic2020", "ph2"]:
        y         = labels[ds]
        meta_prob = clf.predict_proba(
            scaler.transform(np.column_stack([probs[k1][ds], probs[k2][ds]])))[:, 1]

        auc = float(roc_auc_score(y, meta_prob))
        ap  = float(average_precision_score(y, meta_prob))
        tuned_thr = float(thresholds.get(ds, global_thr))

        print(f"  {'-' * 78}")
        print(f"  {ds.upper()}   n={len(y)}   melanoma={int(y.sum())}   "
              f"prevalence={y.mean():.3%}   AUC={auc:.4f}   AP={ap:.4f}")

        for mode, thr in (("GLOBAL", global_thr), ("TUNED", tuned_thr)):
            m = _confusion(meta_prob, y, thr)
            ci_method, ci_lo, ci_hi = _sensitivity_ci(m["tp"], m["fn"])

            same = " (same as global)" if abs(thr - global_thr) < 1e-9 else ""
            print(f"    [{mode:6s} theta={thr:.2f}]{same}")
            print(f"      TP={m['tp']:<5d} FN={m['fn']:<5d} FP={m['fp']:<5d} TN={m['tn']:<5d}")
            print(f"      Sens={m['sensitivity']:.4f} [{ci_lo:.3f}-{ci_hi:.3f}] {ci_method}")
            print(f"      Spec={m['specificity']:.4f}   PPV={m['ppv']:.4f}   "
                  f"NPV={m['npv']:.4f}")
            print(f"      F1={m['f1']:.4f}   F2={m['f2']:.4f}   "
                  f"FP per detected melanoma = {m['fp_per_tp']:.2f}")

            row = dict(dataset=ds, threshold_mode=mode, n=len(y),
                       n_melanoma=int(y.sum()), prevalence=round(float(y.mean()), 4),
                       **{k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in m.items()},
                       auc=round(auc, 4), average_precision=round(ap, 4),
                       sens_ci_method=ci_method,
                       sens_ci_lo=round(ci_lo, 4), sens_ci_hi=round(ci_hi, 4))
            cm_rows.append(row)

            if mode == "TUNED":
                g = cm_rows[-2]
                compare_rows.append(dict(
                    dataset=ds,
                    global_threshold=global_thr, tuned_threshold=thr,
                    global_sens=g["sensitivity"], tuned_sens=row["sensitivity"],
                    global_spec=g["specificity"], tuned_spec=row["specificity"],
                    global_f2=g["f2"], tuned_f2=row["f2"],
                    global_fp=g["fp"], tuned_fp=row["fp"],
                    global_fp_per_tp=g["fp_per_tp"], tuned_fp_per_tp=row["fp_per_tp"],
                    thresholds_agree=abs(thr - global_thr) < 1e-9,
                ))

                for srow in _prevalence_sweep(row["sensitivity"], row["specificity"]):
                    srow["dataset"] = ds
                    srow["source_threshold"] = thr
                    sweep_rows.append(srow)
        print()

    df_cm      = pd.DataFrame(cm_rows)
    df_cmp     = pd.DataFrame(compare_rows)
    df_sweep   = pd.DataFrame(sweep_rows)

    print(f"  {'=' * 78}")
    print("  Threshold audit: does the published per-dataset threshold match the")
    print(f"  global threshold the manuscript claims (theta = {global_thr})?")
    print(f"  {'=' * 78}")
    print(df_cmp[["dataset", "global_threshold", "tuned_threshold", "thresholds_agree",
                  "global_sens", "tuned_sens", "global_f2", "tuned_f2",
                  "global_fp_per_tp", "tuned_fp_per_tp"]].to_string(index=False))

    print(f"\n  {'=' * 78}")
    print("  Referral burden per 1,000 screened (ISIC 2020 operating point)")
    print(f"  {'=' * 78}")
    sub = df_sweep[df_sweep["dataset"] == "isic2020"]
    print(sub[["prevalence", "setting", "melanomas", "tp", "fn", "fp",
               "referrals", "ppv", "npv", "f2", "fp_per_tp"]].to_string(index=False))

    write_excel_sheet(out_xlsx, "ConfusionMatrix",  df_cm)
    write_excel_sheet(out_xlsx, "ThresholdAudit",   df_cmp)
    write_excel_sheet(out_xlsx, "PrevalenceSweep",  df_sweep)
    print(f"\n  Results saved -> {out_xlsx}\n")


if __name__ == "__main__":
    main()
