"""Recomputes every classification number in the manuscript and diffs it against print.

Sweeps thresholds 0.01 to 0.99 for both single models, the mean ensemble and the
meta-learner, then reports confusion matrices, sensitivity CIs, referral burden and
the PH2 subtype split. Mismatches against the PAPER_* tables are listed at the end.

  python scripts/no_seg/verify_all_paper_numbers.py [--refresh-cache]
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
ALL_TEST_SETS    = ["ham10000", "isic2019", "isic2020", "ph2"]
BETA             = 2.0

SWEEP      = np.round(np.arange(0.01, 1.00, 0.01), 2)
PAPER_RANGE = np.round(np.arange(0.20, 0.86, 0.01), 2)   # Section 3.6

CONFIGS = ["resnet50/isic2019", "medfusionnet/isic2020", "mean", "meta"]

PREVALENCE_SWEEP = [
    (0.001, "Population screening"),
    (0.005, "General practice"),
    (0.021, "ISIC 2020 (actual)"),
    (0.050, "Primary-care referral"),
    (0.100, "Specialist clinic"),
    (0.200, "~ISIC 2019"),
    (0.500, "Balanced (reference)"),
]

PAPER_TABLE6 = {
    "ham10000": dict(sens=1.000, spec=0.966, f2=0.947, auc=0.985, tp=158, fn=0),
    "isic2019": dict(sens=0.977, spec=0.973, f2=0.961, auc=0.988, tp=707, fn=17),
    "isic2020": dict(sens=0.949, spec=0.905, f2=0.499, auc=0.940, tp=92,  fn=5),
    "ph2":      dict(sens=0.725, spec=0.956, f2=0.740, auc=0.916, tp=29,  fn=11),
}
PAPER_TABLE7 = {
    "ham10000": dict(tp=158, fn=0,  fp=44,  tn=1255, ppv=0.782, npv=1.000, fp_per_tp=0.28),
    "isic2019": dict(tp=707, fn=17, fp=78,  tn=2794, ppv=0.901, npv=0.994, fp_per_tp=0.11),
    "isic2020": dict(tp=92,  fn=5,  fp=440, tn=4192, ppv=0.173, npv=0.999, fp_per_tp=4.78),
    "ph2":      dict(tp=29,  fn=11, fp=7,   tn=153,  ppv=0.806, npv=0.933, fp_per_tp=0.24),
}
PAPER_TABLE9_AP = {"ham10000": 0.823, "isic2019": 0.960, "isic2020": 0.463, "ph2": 0.776}
PAPER_TABLE3 = {
    "resnet50/isic2019":     dict(ham10000=1.000, isic2019=0.967, isic2020=0.402),
    "medfusionnet/isic2020": dict(ham10000=0.722, isic2019=0.698, isic2020=0.866),
}
PAPER_TABLE5 = {
    "resnet50/isic2019":     dict(min_sens=0.402, avg_sens=0.790, avg_f2=0.736, avg_auc=0.933, ph2=0.675),
    "medfusionnet/isic2020": dict(min_sens=0.698, avg_sens=0.762, avg_f2=0.646, avg_auc=0.857, ph2=0.200),
    "mean":                  dict(min_sens=0.907, avg_sens=0.955, avg_f2=0.829, avg_auc=0.971, ph2=0.125),
    "meta":                  dict(min_sens=0.949, avg_sens=0.975, avg_f2=0.802, avg_auc=0.971, ph2=0.725),
}
PAPER_THRESHOLD_CLAIM = 0.85

_MEAN      = (0.485, 0.456, 0.406)
_STD       = (0.229, 0.224, 0.225)
_SITE_CATS = ["head/neck", "upper extremity", "lower extremity",
              "torso", "palms/soles", "oral/genital"]


class EvalDataset(Dataset):
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
        df, n = self.df, len(self.df)
        if self.neutral_meta:
            self.age      = np.full(n, 0.5, dtype=np.float32)
            self.sex      = np.full(n, 0.5, dtype=np.float32)
            self.site_ohe = np.zeros((n, len(_SITE_CATS)), dtype=np.float32)
            return
        age_col = "age_approx" if "age_approx" in df.columns else None
        self.age = ((df[age_col].fillna(df[age_col].median()) / 100.0).values.astype(np.float32)
                    if age_col else np.zeros(n, dtype=np.float32))
        sex_col = "sex" if "sex" in df.columns else None
        self.sex = (df[sex_col].map({"male": 1.0, "female": 0.0}).fillna(0.5).values.astype(np.float32)
                    if sex_col else np.full(n, 0.5, dtype=np.float32))
        site_col = ("anatom_site_general_challenge"
                    if "anatom_site_general_challenge" in df.columns else None)
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
        image = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image is not None
                 else np.zeros((224, 224, 3), dtype=np.uint8))
        img_t = self.transform(image=image)["image"]
        label = int(row["binary_label"])
        if self.with_meta:
            meta = np.concatenate([[self.age[idx], self.sex[idx]], self.site_ohe[idx]])
            return img_t, torch.from_numpy(meta), label
        return img_t, label


def _load_isic(splits_dir: Path, csv_name: str, source: str, root: Path) -> pd.DataFrame:
    df = pd.read_csv(splits_dir / csv_name)
    df = df[df["dataset_source"] == source].copy()
    df["binary_label"] = (df["label_str"] == "mel").astype(int)
    return resolve_dataset_paths(df, root).reset_index(drop=True)


def _load_ph2(root: Path) -> pd.DataFrame:
    ph2 = root / "PH2"
    df  = pd.read_csv(ph2 / "labels.csv")
    imgs = ph2 / "PH2 Dataset images"
    df["binary_label"] = df["is_melanoma"].astype(int)
    df["image_path"]   = df["image_name"].apply(
        lambda n: str(imgs / n / f"{n}_Dermoscopic_Image" / f"{n}.bmp"))
    return df.reset_index(drop=True)


def _infer(model: nn.Module, loader: DataLoader, device, with_meta: bool) -> np.ndarray:
    model.eval()
    out: list[float] = []
    with torch.no_grad():
        for batch in tqdm(loader, leave=False, unit="b", dynamic_ncols=True, file=sys.stdout):
            if with_meta:
                imgs, mdata, _ = batch
                logits = model(imgs.to(device), mdata.to(device))
            else:
                imgs, _ = batch
                logits = model(imgs.to(device))
            out.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    return np.array(out)


def _metrics(probs: np.ndarray, labels: np.ndarray, thr: float) -> dict:
    """Every rate at one threshold, NaN-safe."""
    preds = (probs >= thr).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    n  = tp + tn + fp + fn

    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv  = tp / (tp + fp) if (tp + fp) else float("nan")
    npv  = tn / (tn + fn) if (tn + fn) else float("nan")
    acc  = (tp + tn) / n if n else float("nan")
    bacc = (sens + spec) / 2

    b2 = BETA ** 2
    f1 = 2 * ppv * sens / (ppv + sens) if (ppv + sens) else 0.0
    f2 = (1 + b2) * ppv * sens / (b2 * ppv + sens) if (b2 * ppv + sens) else 0.0

    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / den) if den else float("nan")

    return dict(threshold=round(float(thr), 2), tp=tp, fn=fn, fp=fp, tn=tn,
                accuracy=acc, balanced_accuracy=bacc, sensitivity=sens,
                specificity=spec, precision=ppv, ppv=ppv, npv=npv,
                f1=f1, f2=f2, mcc=mcc, youden_j=sens + spec - 1,
                fp_per_tp=(fp / tp if tp else float("inf")))


def _best_thr(probs, labels, metric="f2", grid=PAPER_RANGE) -> float:
    best_thr, best = float(grid[0]), -1.0
    for thr in grid:
        v = _metrics(probs, labels, thr)[metric]
        if v > best:
            best, best_thr = v, float(thr)
    return best_thr


def _wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p, d = k / n, 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = (z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _clopper(k, n, alpha=0.05):
    lo = beta_dist.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta_dist.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return (float(lo), float(hi))


def _sens_ci(tp, fn):
    n = tp + fn
    if n == 0:
        return "n/a", float("nan"), float("nan")
    if tp == n:
        lo, hi = _clopper(tp, n)
        return "Clopper-Pearson", lo, hi
    lo, hi = _wilson(tp, n)
    return "Wilson", lo, hi


def _gather(device, splits_dir, root, ablation_dir):
    val_dfs  = {ds: _load_isic(splits_dir, "cls_val.csv",  ds, root) for ds in ISIC_DATASETS}
    test_dfs = {ds: _load_isic(splits_dir, "cls_test.csv", ds, root) for ds in ISIC_DATASETS}
    test_dfs["ph2"] = _load_ph2(root)

    val_probs, test_probs = {}, {}
    for mname, dname in [(MODEL1, DATASET1), (MODEL2, DATASET2)]:
        key    = f"{mname}/{dname}"
        run_id = f"{mname}_{AUG_MODE}"
        ckpt   = ablation_dir / dname / run_id / "checkpoints" / f"{run_id}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        print(f"  Loading {key}  <-  {ckpt.relative_to(ablation_dir.parent)}")
        cfg   = load_config(mname)
        wm    = uses_metadata(mname)
        model = build_model(mname, cfg, num_classes=2)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model.to(device).eval().requires_grad_(False)
        inp = getattr(cfg, "input_size", 224)
        nw  = getattr(cfg, "num_workers", 0)

        vp, tp_ = {}, {}
        for ds in ISIC_DATASETS:
            vp[ds] = _infer(model, DataLoader(
                EvalDataset(val_dfs[ds], inp, wm), batch_size=cfg.batch_size,
                shuffle=False, num_workers=nw, pin_memory=(nw > 0)), device, wm)
        for ds in ALL_TEST_SETS:
            tp_[ds] = _infer(model, DataLoader(
                EvalDataset(test_dfs[ds], inp, wm, neutral_meta=(ds == "ph2")),
                batch_size=cfg.batch_size, shuffle=False,
                num_workers=nw, pin_memory=(nw > 0)), device, wm)
        val_probs[key], test_probs[key] = vp, tp_

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return (val_probs, test_probs,
            {ds: df["binary_label"].values for ds, df in val_dfs.items()},
            {ds: df["binary_label"].values for ds, df in test_dfs.items()},
            test_dfs["ph2"])


def _check(label, got, want, tol, issues, unit=""):
    ok = abs(got - want) <= tol
    flag = "OK " if ok else "MISMATCH"
    print(f"    [{flag}] {label:<44} recomputed={got:>9.3f}{unit}  paper={want:>9.3f}{unit}")
    if not ok:
        issues.append((label, got, want))
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-cache", action="store_true")
    args = ap.parse_args()

    cfg    = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = Path(cfg.paths.data_splits)
    root   = Path(cfg.paths.melanoma_data)
    abl    = Path(cfg.paths.outputs) / "ablation_noseg"
    deploy = abl / "meta" / "deployment"
    out    = abl / "meta" / "verification"
    out.mkdir(parents=True, exist_ok=True)
    cache  = out / "_all_probs.pkl"
    xlsx   = out / "verify_all_paper_numbers.xlsx"

    with open(deploy / "meta_learner.pkl", "rb") as f:
        dep = pickle.load(f)
    scaler, clf, thresholds = dep["scaler"], dep["clf"], dep["thresholds"]
    global_thr = float(thresholds["global"])

    print(f"\n{'=' * 88}")
    print("  FULL VERIFICATION OF MANUSCRIPT CLASSIFICATION NUMBERS")
    print(f"{'=' * 88}")
    print(f"  Artifact     : {deploy / 'meta_learner.pkl'}")
    print(f"  Base models  : {dep['model1']} + {dep['model2']}")
    print(f"  Meta-learner : coef={clf.coef_[0].round(4).tolist()} bias={clf.intercept_[0]:.4f}")
    print(f"  Thresholds   : {thresholds}\n")

    if cache.exists() and not args.refresh_cache:
        print(f"  [CACHE] {cache.name}\n")
        c = pickle.load(open(cache, "rb"))
        vprobs, tprobs, vlab, tlab, ph2_df = (
            c["vprobs"], c["tprobs"], c["vlab"], c["tlab"], c["ph2_df"])
    else:
        vprobs, tprobs, vlab, tlab, ph2_df = _gather(device, splits, root, abl)
        pickle.dump(dict(vprobs=vprobs, tprobs=tprobs, vlab=vlab, tlab=tlab, ph2_df=ph2_df),
                    open(cache, "wb"))
        print(f"\n  [CACHE] saved -> {cache.name}\n")

    k1, k2 = f"{MODEL1}/{DATASET1}", f"{MODEL2}/{DATASET2}"

    def probs_for(config: str, split: str, ds: str) -> np.ndarray:
        src = vprobs if split == "val" else tprobs
        if config == k1:
            return src[k1][ds]
        if config == k2:
            return src[k2][ds]
        if config == "mean":
            return (src[k1][ds] + src[k2][ds]) / 2.0
        return clf.predict_proba(
            scaler.transform(np.column_stack([src[k1][ds], src[k2][ds]])))[:, 1]

    issues: list[tuple] = []

    print(f"  {'=' * 84}")
    print("  [1/7] Full threshold sweep, all configs x all datasets (theta 0.01-0.99)")
    print(f"  {'=' * 84}")
    sweep_rows = []
    for config in CONFIGS:
        for ds in ALL_TEST_SETS:
            y, p = tlab[ds], probs_for(config, "test", ds)
            for thr in SWEEP:
                m = _metrics(p, y, thr)
                m.update(config=config, dataset=ds)
                sweep_rows.append(m)
    df_sweep = pd.DataFrame(sweep_rows)
    print(f"    {len(df_sweep)} rows -> sheet 'ThresholdSweep'\n")

    print(f"  {'=' * 84}")
    print("  [2/7] Single-global-threshold sweep (meta-learner, one theta for all sets)")
    print(f"  {'=' * 84}")
    print(f"    {'theta':>6} {'HAM':>7} {'I2019':>7} {'I2020':>7} {'PH2':>7} "
          f"{'MIN':>7} {'AVG':>7} {'avgF2':>7} {'avgACC':>7} {'I20 FP/TP':>10}")
    grows = []
    for thr in SWEEP:
        per = {ds: _metrics(probs_for("meta", "test", ds), tlab[ds], thr)
               for ds in ALL_TEST_SETS}
        sens = {ds: per[ds]["sensitivity"] for ds in ALL_TEST_SETS}
        row = dict(threshold=float(thr),
                   **{f"sens_{ds}": sens[ds] for ds in ALL_TEST_SETS},
                   **{f"spec_{ds}": per[ds]["specificity"] for ds in ALL_TEST_SETS},
                   **{f"f2_{ds}": per[ds]["f2"] for ds in ALL_TEST_SETS},
                   **{f"acc_{ds}": per[ds]["accuracy"] for ds in ALL_TEST_SETS},
                   min_sens=min(sens.values()),
                   avg_sens=float(np.mean(list(sens.values()))),
                   avg_sens_isic=float(np.mean([sens[d] for d in ISIC_DATASETS])),
                   avg_f2=float(np.mean([per[d]["f2"] for d in ISIC_DATASETS])),
                   avg_acc=float(np.mean([per[d]["accuracy"] for d in ISIC_DATASETS])),
                   isic2020_fp_per_tp=per["isic2020"]["fp_per_tp"])
        grows.append(row)
        if round(float(thr) * 100) % 5 == 0:
            print(f"    {thr:>6.2f} " + " ".join(f"{sens[ds]:>7.3f}" for ds in ALL_TEST_SETS)
                  + f" {row['min_sens']:>7.3f} {row['avg_sens']:>7.3f}"
                  f" {row['avg_f2']:>7.3f} {row['avg_acc']:>7.3f}"
                  f" {row['isic2020_fp_per_tp']:>10.2f}")
    df_global = pd.DataFrame(grows)

    best_min = df_global.loc[df_global["min_sens"].idxmax()]
    # BROKEN: degenerates to theta=0.01. Max worst-case F2 is right, lands on 0.20.
    tied  = df_global[df_global["min_sens"] >= best_min["min_sens"] - 0.005]
    chosen = tied.loc[tied["threshold"].idxmax()]
    print(f"\n    Best worst-case sensitivity : {best_min['min_sens']:.4f} "
          f"at theta={best_min['threshold']:.2f}")
    print(f"    Highest theta within 0.5pp  : theta={chosen['threshold']:.2f}  "
          f"(min_sens={chosen['min_sens']:.4f}, avg_sens={chosen['avg_sens']:.4f})")
    print(f"    Manuscript claims           : theta={PAPER_THRESHOLD_CLAIM:.2f}  "
          f"(min_sens={df_global.loc[df_global['threshold'].round(2) == PAPER_THRESHOLD_CLAIM, 'min_sens'].iloc[0]:.4f})\n")
    recommended_thr = float(chosen["threshold"])

    print(f"  {'=' * 84}")
    print(f"  [3/7] Confusion matrices at theta=0.85 (claimed) and "
          f"theta={recommended_thr:.2f} (recommended)")
    print(f"  {'=' * 84}")
    cm_rows = []
    for ds in ALL_TEST_SETS:
        y, p = tlab[ds], probs_for("meta", "test", ds)
        auc  = float(roc_auc_score(y, p))
        apr  = float(average_precision_score(y, p))
        tuned = float(thresholds.get(ds, global_thr))
        print(f"\n    {ds.upper():<10} n={len(y):<6} mel={int(y.sum()):<5} "
              f"prev={y.mean():.3%}  AUC={auc:.4f}  AP={apr:.4f}")
        for name, thr in [("claimed-global", global_thr),
                          ("paper-actual", tuned),
                          ("recommended", recommended_thr)]:
            m = _metrics(p, y, thr)
            meth, lo, hi = _sens_ci(m["tp"], m["fn"])
            print(f"      {name:<15} th={thr:.2f}  TP={m['tp']:<4d} FN={m['fn']:<4d} "
                  f"FP={m['fp']:<4d} TN={m['tn']:<5d} "
                  f"sens={m['sensitivity']:.4f} [{lo:.3f}-{hi:.3f}] "
                  f"spec={m['specificity']:.4f} acc={m['accuracy']:.4f} "
                  f"PPV={m['ppv']:.4f} NPV={m['npv']:.4f} F1={m['f1']:.4f} "
                  f"F2={m['f2']:.4f} FP/TP={m['fp_per_tp']:.2f}")
            cm_rows.append(dict(dataset=ds, variant=name, n=len(y),
                                n_melanoma=int(y.sum()), prevalence=float(y.mean()),
                                **m, auc=auc, average_precision=apr,
                                sens_ci_method=meth, sens_ci_lo=lo, sens_ci_hi=hi))
    df_cm = pd.DataFrame(cm_rows)

    print(f"\n  {'=' * 84}")
    print("  [4/7] Diff against Tables 6 / 7 / 9  (at the threshold the paper ACTUALLY used)")
    print(f"  {'=' * 84}")
    for ds in ALL_TEST_SETS:
        y, p  = tlab[ds], probs_for("meta", "test", ds)
        tuned = float(thresholds.get(ds, global_thr))
        m     = _metrics(p, y, tuned)
        auc   = float(roc_auc_score(y, p))
        apr   = float(average_precision_score(y, p))
        t6, t7 = PAPER_TABLE6[ds], PAPER_TABLE7[ds]
        print(f"\n    --- {ds.upper()}  (paper threshold = {tuned:.2f}) ---")
        _check(f"{ds} sensitivity",  m["sensitivity"], t6["sens"], 0.002, issues)
        _check(f"{ds} specificity",  m["specificity"], t6["spec"], 0.002, issues)
        _check(f"{ds} F2",           m["f2"],          t6["f2"],   0.002, issues)
        _check(f"{ds} AUC",          m["auc"] if "auc" in m else auc, t6["auc"], 0.002, issues)
        _check(f"{ds} AP",           apr, PAPER_TABLE9_AP[ds], 0.003, issues)
        _check(f"{ds} TP",           m["tp"], t7["tp"], 0, issues)
        _check(f"{ds} FN",           m["fn"], t7["fn"], 0, issues)
        _check(f"{ds} FP",           m["fp"], t7["fp"], 0, issues)
        _check(f"{ds} TN",           m["tn"], t7["tn"], 0, issues)
        _check(f"{ds} PPV",          m["ppv"], t7["ppv"], 0.002, issues)
        _check(f"{ds} NPV",          m["npv"], t7["npv"], 0.002, issues)
        _check(f"{ds} FP/TP",        m["fp_per_tp"], t7["fp_per_tp"], 0.02, issues)

    print(f"\n  {'=' * 84}")
    print("  [5/7] Diff against Table 3 (single-model cross-dataset sensitivity)")
    print(f"  {'=' * 84}")
    t3_rows = []
    for config in [k1, k2]:
        print(f"\n    --- {config} ---")
        for ds in ISIC_DATASETS:
            thr = _best_thr(probs_for(config, "val", ds), vlab[ds])
            m   = _metrics(probs_for(config, "test", ds), tlab[ds], thr)
            _check(f"{config} on {ds} sens (val-thr={thr:.2f})",
                   m["sensitivity"], PAPER_TABLE3[config][ds], 0.01, issues)
            t3_rows.append(dict(config=config, dataset=ds, val_threshold=thr, **m))
    df_t3 = pd.DataFrame(t3_rows)

    print(f"\n  {'=' * 84}")
    print("  [6/7] Diff against Table 5 (config comparison: min/avg sens, avg F2, avg AUC, PH2)")
    print(f"  {'=' * 84}")
    t5_rows = []
    for config in CONFIGS:
        per = {}
        for ds in ISIC_DATASETS:
            thr = (float(thresholds[ds]) if config == "meta"
                   else _best_thr(probs_for(config, "val", ds), vlab[ds]))
            per[ds] = _metrics(probs_for(config, "test", ds), tlab[ds], thr)
            per[ds]["auc"] = float(roc_auc_score(tlab[ds], probs_for(config, "test", ds)))
        sens = [per[d]["sensitivity"] for d in ISIC_DATASETS]
        ph2m = _metrics(probs_for(config, "test", "ph2"), tlab["ph2"], global_thr)
        want = PAPER_TABLE5[config]
        print(f"\n    --- {config} ---")
        _check(f"{config} min sensitivity",  min(sens),  want["min_sens"], 0.01, issues)
        _check(f"{config} avg sensitivity",  float(np.mean(sens)), want["avg_sens"], 0.01, issues)
        _check(f"{config} avg F2",
               float(np.mean([per[d]["f2"] for d in ISIC_DATASETS])), want["avg_f2"], 0.01, issues)
        _check(f"{config} avg AUC",
               float(np.mean([per[d]["auc"] for d in ISIC_DATASETS])), want["avg_auc"], 0.01, issues)
        _check(f"{config} PH2 sensitivity", ph2m["sensitivity"], want["ph2"], 0.03, issues)
        t5_rows.append(dict(config=config, min_sens=min(sens),
                            avg_sens=float(np.mean(sens)),
                            avg_f2=float(np.mean([per[d]["f2"] for d in ISIC_DATASETS])),
                            avg_auc=float(np.mean([per[d]["auc"] for d in ISIC_DATASETS])),
                            ph2_sens=ph2m["sensitivity"]))
    df_t5 = pd.DataFrame(t5_rows)

    print(f"\n  {'=' * 84}")
    print("  [7/7] Prevalence sweep (per 1,000 screened) and PH2 subtype breakdown")
    print(f"  {'=' * 84}")
    prev_rows = []
    for ds in ALL_TEST_SETS:
        for variant, thr in [("paper-actual", float(thresholds.get(ds, global_thr))),
                             ("recommended", recommended_thr)]:
            m = _metrics(probs_for("meta", "test", ds), tlab[ds], thr)
            for pi, setting in PREVALENCE_SWEEP:
                mel = 1000 * pi
                tp  = mel * m["sensitivity"]
                fn  = mel - tp
                fp  = 1000 * (1 - pi) * (1 - m["specificity"])
                tn  = 1000 * (1 - pi) * m["specificity"]
                refs = tp + fp
                ppv = tp / refs if refs else float("nan")
                npv = tn / (tn + fn) if (tn + fn) else float("nan")
                f2  = (5 * ppv * m["sensitivity"]) / (4 * ppv + m["sensitivity"]) \
                      if (4 * ppv + m["sensitivity"]) else 0.0
                prev_rows.append(dict(dataset=ds, variant=variant, threshold=thr,
                                      prevalence=pi, setting=setting,
                                      melanomas=round(mel, 1), tp=round(tp, 1),
                                      fn=round(fn, 1), fp=round(fp, 1),
                                      referrals=round(refs, 1), ppv=round(ppv, 4),
                                      npv=round(npv, 4), f2=round(f2, 3),
                                      fp_per_tp=round(fp / tp, 1) if tp else float("inf")))
    df_prev = pd.DataFrame(prev_rows)
    sub = df_prev[(df_prev["dataset"] == "isic2020") & (df_prev["variant"] == "paper-actual")]
    print("\n    ISIC 2020 operating point, per 1,000 screened:")
    print(sub[["prevalence", "setting", "melanomas", "tp", "fn", "fp",
               "referrals", "ppv", "npv", "f2", "fp_per_tp"]].to_string(index=False))

    ph2_rows = []
    ph2_p = probs_for("meta", "test", "ph2")
    diag_col = next((c for c in ("diagnosis", "label", "class") if c in ph2_df.columns), None)
    if diag_col:
        print("\n    PH2 by subtype:")
        for variant, thr in [("claimed-global", global_thr), ("recommended", recommended_thr)]:
            for diag, grp in ph2_df.groupby(diag_col):
                idx  = grp.index.values
                flag = int((ph2_p[idx] >= thr).sum())
                ph2_rows.append(dict(variant=variant, threshold=thr, diagnosis=diag,
                                     n=len(idx), predicted_melanoma=flag,
                                     rate=round(flag / len(idx), 4)))
                print(f"      [{variant:<14} th={thr:.2f}] {str(diag):<16} "
                      f"n={len(idx):<4} flagged={flag:<4} rate={flag / len(idx):.4f}")
    df_ph2 = pd.DataFrame(ph2_rows)

    write_excel_sheet(xlsx, "ThresholdSweep",    df_sweep)
    write_excel_sheet(xlsx, "GlobalThresholdSweep", df_global)
    write_excel_sheet(xlsx, "ConfusionMatrices", df_cm)
    write_excel_sheet(xlsx, "Table3_SingleModel", df_t3)
    write_excel_sheet(xlsx, "Table5_Configs",    df_t5)
    write_excel_sheet(xlsx, "PrevalenceSweep",   df_prev)
    if not df_ph2.empty:
        write_excel_sheet(xlsx, "PH2_Subtypes",  df_ph2)

    print(f"\n{'=' * 88}")
    if issues:
        print(f"  {len(issues)} MISMATCH(ES) against the manuscript:")
        for label, got, want in issues:
            print(f"    - {label:<46} recomputed={got:<10.3f} paper={want:<10.3f} "
                  f"delta={got - want:+.3f}")
    else:
        print("  All checked values reproduce the manuscript within tolerance.")
    print(f"{'=' * 88}")
    print(f"  Saved -> {xlsx}\n")


if __name__ == "__main__":
    main()
