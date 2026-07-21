"""Statistical robustness of the ISIC 2020 result under 2% melanoma prevalence.

Reports point estimates at a validation-selected threshold, analytic sensitivity
CIs (Wilson, or Clopper-Pearson at 100%), bootstrap CIs over 10,000 resamples, and
a prevalence sweep showing F2 tracks base rate rather than classifier quality.

evaluate_deployment_pair.py picks its threshold with test labels. This one selects
on validation only, matching export_for_deployment.py.

  python scripts/no_seg/evaluate_class_imbalance_robustness.py [--n-boot 20000]
"""
from __future__ import annotations

import argparse
import math
import pickle
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
from scipy.stats import beta as beta_dist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.models.registry import build_model, uses_metadata
from src.utils.config import load_config
from src.utils.io import resolve_dataset_paths, write_excel_sheet

MODEL1, DATASET1 = "resnet50",     "isic2019"
MODEL2, DATASET2 = "medfusionnet", "isic2020"
AUG_MODE         = "none_sens"
ALL_DATASETS     = ["ham10000", "isic2019", "isic2020"]
BETA             = 2.0
THR_RANGE        = np.round(np.arange(0.20, 0.86, 0.01), 2)
PREVALENCE_SWEEP = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]

_MEAN      = (0.485, 0.456, 0.406)
_STD       = (0.229, 0.224, 0.225)
_SITE_CATS = ["head/neck", "upper extremity", "lower extremity",
              "torso", "palms/soles", "oral/genital"]


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
    return df.reset_index(drop=True)


def _run_inference(model: nn.Module, loader: DataLoader,
                   device: torch.device, with_meta: bool) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, leave=False, unit="batch", dynamic_ncols=True, file=sys.stdout):
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
    return dict(sensitivity=sens, specificity=spec, precision=prec, f2=f2, f1=f1,
                tp=tp, tn=tn, fp=fp, fn=fn, threshold=round(thr, 2))


def _best_thr_on_val(probs: np.ndarray, labels: np.ndarray) -> float:
    """Selects the F2-optimal threshold using ONLY the given (validation) probabilities/labels.

    This is the leakage-free counterpart to evaluate_deployment_pair.py's `_best_thr`, which is
    called there with test-set labels. Matches export_for_deployment.py's methodology instead.
    """
    best_thr, best = float(THR_RANGE[0]), -1.0
    for thr in THR_RANGE:
        f2 = _metrics(probs, labels, thr)["f2"]
        if f2 > best:
            best, best_thr = f2, float(thr)
    return best_thr


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    phat  = k / n
    denom = 1 + z ** 2 / n
    center = (phat + z ** 2 / (2 * n)) / denom
    half   = (z * math.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    lo = beta_dist.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta_dist.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return (float(lo), float(hi))


def _sensitivity_ci(tp: int, fn: int) -> tuple[str, tuple[float, float]]:
    n = tp + fn
    if tp == n:
        return "Clopper-Pearson (exact)", _clopper_pearson_ci(tp, n)
    return "Wilson score", _wilson_ci(tp, n)


def _bootstrap_ci(probs: np.ndarray, labels: np.ndarray, thr: float,
                  n_boot: int, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    n = len(labels)
    keys = ("sensitivity", "specificity", "precision", "f1", "f2", "auc", "average_precision")
    samples: dict[str, list[float]] = {k: [] for k in keys}

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        p, l = probs[idx], labels[idx]
        if l.sum() == 0 or l.sum() == n:
            continue  # degenerate resample, cannot score
        m = _metrics(p, l, thr)
        samples["sensitivity"].append(m["sensitivity"])
        samples["specificity"].append(m["specificity"])
        samples["precision"].append(m["precision"])
        samples["f1"].append(m["f1"])
        samples["f2"].append(m["f2"])
        try:
            samples["auc"].append(float(roc_auc_score(l, p)))
        except Exception:
            pass
        try:
            samples["average_precision"].append(float(average_precision_score(l, p)))
        except Exception:
            pass

    out = {"n_valid_resamples": len(samples["sensitivity"])}
    for k in keys:
        arr = np.array(samples[k])
        if len(arr) == 0:
            out[k] = (float("nan"), float("nan"))
        else:
            out[k] = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
    return out


def _prevalence_sweep(sens: float, spec: float, prevalences: list[float]) -> list[dict]:
    rows = []
    for pi in prevalences:
        denom = sens * pi + (1 - spec) * (1 - pi)
        prec  = (sens * pi) / denom if denom > 0 else 0.0
        f2den = 4 * prec + sens
        f2    = (5 * prec * sens) / f2den if f2den > 0 else 0.0
        rows.append(dict(prevalence=pi, precision=round(prec, 4), f2=round(f2, 4)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-boot", type=int, default=10_000,
                        help="Number of bootstrap resamples per dataset (default 10000).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_cfg      = load_config()
    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits_dir    = Path(base_cfg.paths.data_splits)
    melanoma_root = Path(base_cfg.paths.melanoma_data)
    ablation_dir  = Path(base_cfg.paths.outputs) / "ablation_noseg"
    out_dir       = ablation_dir / "meta" / "imbalance_robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_xlsx      = out_dir / "class_imbalance_robustness.xlsx"
    probs_cache   = out_dir / "_deployed_pair_probs_cache.pkl"

    print(f"\n{'=' * 78}")
    print(f"  Class-imbalance / statistical-confidence robustness check")
    print(f"  Deployed pair: {MODEL1}/{DATASET1}  +  {MODEL2}/{DATASET2}")
    print(f"  Bootstrap resamples: {args.n_boot}")
    print(f"{'=' * 78}\n")

    val_dfs, test_dfs = {}, {}
    for ds in ALL_DATASETS:
        val_dfs[ds]  = _load_df(splits_dir, "cls_val.csv",  ds, melanoma_root)
        test_dfs[ds] = _load_df(splits_dir, "cls_test.csv", ds, melanoma_root)
        print(f"  {ds.upper():<12}  val={len(val_dfs[ds])} (mel={int(val_dfs[ds]['binary_label'].sum())})"
              f"   test={len(test_dfs[ds])} (mel={int(test_dfs[ds]['binary_label'].sum())})")
    print()

    if probs_cache.exists():
        print(f"  [CACHE] loading precomputed probabilities from {probs_cache.name} ...\n")
        with open(probs_cache, "rb") as f:
            cache = pickle.load(f)
        val_probs, test_probs = cache["val_probs"], cache["test_probs"]
    else:
        val_probs, test_probs = {}, {}
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

            vp, tp = {}, {}
            for ds in ALL_DATASETS:
                v_loader = DataLoader(EvalDataset(val_dfs[ds], inp, with_meta=wm),
                                      batch_size=config.batch_size, shuffle=False,
                                      num_workers=nw, pin_memory=(nw > 0))
                t_loader = DataLoader(EvalDataset(test_dfs[ds], inp, with_meta=wm),
                                      batch_size=config.batch_size, shuffle=False,
                                      num_workers=nw, pin_memory=(nw > 0))
                vp[ds], _ = _run_inference(model, v_loader, device, wm)
                tp[ds], _ = _run_inference(model, t_loader, device, wm)
            val_probs[key], test_probs[key] = vp, tp

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        with open(probs_cache, "wb") as f:
            pickle.dump({"val_probs": val_probs, "test_probs": test_probs}, f)
        print(f"\n  [CACHE] saved -> {probs_cache.name}\n")

    k1, k2 = f"{MODEL1}/{DATASET1}", f"{MODEL2}/{DATASET2}"

    print("  Fitting meta-learner on pooled validation set (matches paper Section 3.6) ...")
    X_parts, y_parts = [], []
    for ds in ALL_DATASETS:
        X_parts.append(np.column_stack([val_probs[k1][ds], val_probs[k2][ds]]))
        y_parts.append(val_dfs[ds]["binary_label"].values)
    X_meta   = np.vstack(X_parts)
    y_meta   = np.concatenate(y_parts)
    scaler   = StandardScaler().fit(X_meta)
    clf      = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced")
    clf.fit(scaler.transform(X_meta), y_meta)
    print(f"  weights=[{clf.coef_[0][0]:.4f}, {clf.coef_[0][1]:.4f}]  bias={clf.intercept_[0]:.4f}\n")

    point_rows, ci_rows, boot_rows, sweep_rows = [], [], [], []

    for ds in ALL_DATASETS:
        val_labels  = val_dfs[ds]["binary_label"].values
        test_labels = test_dfs[ds]["binary_label"].values

        val_meta_prob  = clf.predict_proba(scaler.transform(
            np.column_stack([val_probs[k1][ds], val_probs[k2][ds]])))[:, 1]
        test_meta_prob = clf.predict_proba(scaler.transform(
            np.column_stack([test_probs[k1][ds], test_probs[k2][ds]])))[:, 1]

        # leakage-free: threshold chosen on validation only, never touches test labels/probs
        thr = _best_thr_on_val(val_meta_prob, val_labels)

        m   = _metrics(test_meta_prob, test_labels, thr)
        auc = float(roc_auc_score(test_labels, test_meta_prob))
        ap  = float(average_precision_score(test_labels, test_meta_prob))
        prevalence = test_labels.mean()

        ci_method, (ci_lo, ci_hi) = _sensitivity_ci(m["tp"], m["fn"])
        boot = _bootstrap_ci(test_meta_prob, test_labels, thr, args.n_boot, seed=args.seed)

        print(f"  {'-' * 74}")
        print(f"  {ds.upper()}  (test n={len(test_labels)}, mel={m['tp'] + m['fn']}, "
              f"prevalence={prevalence:.3%}, threshold={thr:.2f})")
        print(f"    Sensitivity = {m['sensitivity']:.4f}   {ci_method} 95% CI = "
              f"[{ci_lo:.4f}, {ci_hi:.4f}]   bootstrap 95% CI = "
              f"[{boot['sensitivity'][0]:.4f}, {boot['sensitivity'][1]:.4f}]")
        print(f"    Specificity = {m['specificity']:.4f}   bootstrap 95% CI = "
              f"[{boot['specificity'][0]:.4f}, {boot['specificity'][1]:.4f}]")
        print(f"    Precision   = {m['precision']:.4f}   bootstrap 95% CI = "
              f"[{boot['precision'][0]:.4f}, {boot['precision'][1]:.4f}]")
        print(f"    F2          = {m['f2']:.4f}   bootstrap 95% CI = "
              f"[{boot['f2'][0]:.4f}, {boot['f2'][1]:.4f}]")
        print(f"    AUC         = {auc:.4f}   bootstrap 95% CI = "
              f"[{boot['auc'][0]:.4f}, {boot['auc'][1]:.4f}]")
        print(f"    Avg.Prec.   = {ap:.4f}   bootstrap 95% CI = "
              f"[{boot['average_precision'][0]:.4f}, {boot['average_precision'][1]:.4f}]"
              f"   <- threshold-independent, imbalance-robust companion to F2")

        point_rows.append(dict(
            dataset=ds, n_test=len(test_labels), n_positive=m["tp"] + m["fn"],
            prevalence=round(float(prevalence), 4), threshold=m["threshold"],
            sensitivity=round(m["sensitivity"], 4), specificity=round(m["specificity"], 4),
            precision=round(m["precision"], 4), f1=round(m["f1"], 4), f2=round(m["f2"], 4),
            auc=round(auc, 4), average_precision=round(ap, 4),
            tp=m["tp"], tn=m["tn"], fp=m["fp"], fn=m["fn"],
        ))
        ci_rows.append(dict(dataset=ds, metric="sensitivity", method=ci_method,
                            estimate=round(m["sensitivity"], 4),
                            ci_lo=round(ci_lo, 4), ci_hi=round(ci_hi, 4)))
        for metric in ("sensitivity", "specificity", "precision", "f1", "f2", "auc", "average_precision"):
            lo, hi = boot[metric]
            boot_rows.append(dict(dataset=ds, metric=metric, n_boot=args.n_boot,
                                  n_valid_resamples=boot["n_valid_resamples"],
                                  ci_lo=round(lo, 4), ci_hi=round(hi, 4)))

        for row in _prevalence_sweep(m["sensitivity"], m["specificity"], PREVALENCE_SWEEP):
            row["dataset"] = ds
            row["actual_prevalence"] = abs(row["prevalence"] - prevalence) < 0.005
            sweep_rows.append(row)

    print(f"\n  {'=' * 74}")
    print("  Prevalence-sensitivity sweep (sens/spec held at each dataset's actual operating")
    print("  point; only the assumed positive-class prevalence changes). F2 is driven")
    print("  by base rate, not by classifier quality:")
    print(f"  {'=' * 74}")
    df_sweep = pd.DataFrame(sweep_rows)
    for ds in ALL_DATASETS:
        sub = df_sweep[df_sweep["dataset"] == ds]
        print(f"\n  {ds.upper()}")
        print(f"  {'Prevalence':>10}  {'Precision':>10}  {'F2':>8}")
        for _, r in sub.iterrows():
            marker = "  <- actual test prevalence" if r["actual_prevalence"] else ""
            print(f"  {r['prevalence']:>10.0%}  {r['precision']:>10.4f}  {r['f2']:>8.4f}{marker}")

    df_point = pd.DataFrame(point_rows)
    df_ci    = pd.DataFrame(ci_rows)
    df_boot  = pd.DataFrame(boot_rows)
    write_excel_sheet(out_xlsx, "PointEstimates", df_point)
    write_excel_sheet(out_xlsx, "AnalyticSensitivityCI", df_ci)
    write_excel_sheet(out_xlsx, "BootstrapCI", df_boot)
    write_excel_sheet(out_xlsx, "PrevalenceSweep", df_sweep.drop(columns=["actual_prevalence"]))

    print(f"\n  Results saved -> {out_xlsx}\n")


if __name__ == "__main__":
    main()
