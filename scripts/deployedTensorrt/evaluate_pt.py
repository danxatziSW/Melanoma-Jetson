"""Evaluates the deployment meta-learner pipeline using PyTorch on CPU (the Jetson fallback path).

Usage: python3 scripts/deployedTensorrt/evaluate_pt.py [--latency-only] [--accuracy-only]
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.models.medfusionnet import MedFusionNet
from src.models.resnet import build_resnet
from src.utils.config import load_config
from src.utils.io import resolve_dataset_paths

_BASE_CFG   = load_config()
_ROOT       = Path(__file__).resolve().parents[3]
_DEPLOY_DIR = _ROOT / "outputs" / "ablation_noseg" / "meta" / "deployment"
_OUT_DIR    = _DEPLOY_DIR / "JetsonPT"
_SPLITS_DIR = _ROOT / "data_splits"
# raw dataset location — override via MELANOMA_DATA_DIR (see configs/base.yaml)
_MELANOMA_ROOT = Path(_BASE_CFG.paths.melanoma_data)

_DEVICE = torch.device("cpu")

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_ALL_DS = ["ham10000", "isic2019", "isic2020"]
_SITE_CATS = ["head/neck", "upper extremity", "lower extremity",
              "torso", "palms/soles", "oral/genital"]
_NEUTRAL_META = np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
_THRESHOLDS   = np.round(np.arange(0.05, 0.96, 0.01), 2)


_R="\033[0m"; _B="\033[1m"; _G="\033[32m"; _Y="\033[33m"
_RE="\033[31m"; _C="\033[36m"; _D="\033[2m"; _W="\033[97m"; _BL="\033[34m"
def _c(t, *codes): return "".join(codes) + t + _R
_SEP  = _c("═"*68, _BL)
_SEP2 = _c("─"*68, _D)


def _preprocess(image_bgr: np.ndarray, size: int = 224) -> np.ndarray:
    big = int(size * 1.1)
    img = cv2.resize(image_bgr, (big, big))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    st  = (big - size) // 2
    img = img[st:st + size, st:st + size]
    return ((img - _MEAN) / _STD).transpose(2, 0, 1).astype(np.float32)


def _build_meta(row) -> np.ndarray:
    import math
    _age_raw = row.get("age_approx", 50)
    age      = float(_age_raw) / 100.0 if (_age_raw is not None and not (isinstance(_age_raw, float) and math.isnan(_age_raw))) else 0.5
    sex_raw  = str(row.get("sex", "")).lower()
    sex      = 1.0 if sex_raw == "male" else (0.0 if sex_raw == "female" else 0.5)
    site     = np.zeros(len(_SITE_CATS), dtype=np.float32)
    site_raw = str(row.get("anatom_site_general_challenge", ""))
    for i, cat in enumerate(_SITE_CATS):
        if site_raw == cat:
            site[i] = 1.0
    return np.concatenate([[age, sex], site]).astype(np.float32)


class _PTResNet:
    def __init__(self, pt_path: Path):
        import types
        sd = torch.load(pt_path, map_location=_DEVICE)
        n_classes = list(sd.values())[-1].shape[0]
        cfg = types.SimpleNamespace(pretrained=False, dropout=0.3)
        self.model = build_resnet(cfg, num_classes=n_classes)
        self.model.load_state_dict(sd)
        self.model.eval()
        self._binary = (n_classes == 1)

    def predict_prob(self, image_bgr: np.ndarray) -> float:
        chw = _preprocess(image_bgr)
        x   = torch.from_numpy(chw).unsqueeze(0)
        with torch.no_grad():
            out = self.model(x)
        if self._binary:
            return float(torch.sigmoid(out).squeeze())
        return float(torch.softmax(out, dim=1)[0, 1])


class _PTMedFusion:
    def __init__(self, pt_path: Path):
        sd = torch.load(pt_path, map_location=_DEVICE)
        n_classes = sd[list(sd.keys())[-1]].shape[0]
        self.model = MedFusionNet(num_classes=n_classes, metadata_dim=8,
                                  fusion_hidden=256, dropout=0.4, pretrained=False)
        self.model.load_state_dict(sd)
        self.model.eval()
        self._binary = (n_classes == 1)

    def predict_prob(self, image_bgr: np.ndarray,
                     meta_np: Optional[np.ndarray] = None) -> float:
        chw  = _preprocess(image_bgr)
        meta = meta_np if meta_np is not None else _NEUTRAL_META
        x    = torch.from_numpy(chw).unsqueeze(0)
        m    = torch.from_numpy(meta).unsqueeze(0)
        with torch.no_grad():
            out = self.model(x, m)
        if self._binary:
            return float(torch.sigmoid(out).squeeze())
        return float(torch.softmax(out, dim=1)[0, 1])


class MetaLearner:
    def __init__(self, pkl_path: Path):
        with open(pkl_path, "rb") as f:
            bundle = pickle.load(f)
        self.scaler     = bundle["scaler"]
        self.clf        = bundle["clf"]
        self.thresholds = bundle["thresholds"]
        if not hasattr(self.clf, "multi_class"):
            self.clf.multi_class = "auto"

        self.r50 = _PTResNet(
            _DEPLOY_DIR / "resnet50_none_sens.pt")
        self.mfn = _PTMedFusion(
            _DEPLOY_DIR / "medfusionnet_none_sens.pt")

    def predict_probs_only(self, image_bgr: np.ndarray,
                           meta_np: Optional[np.ndarray] = None
                           ) -> Tuple[float, float, float]:
        import math
        prob_r = self.r50.predict_prob(image_bgr)
        prob_m = self.mfn.predict_prob(image_bgr, meta_np)
        if math.isnan(prob_r) or math.isnan(prob_m):
            return float("nan"), float("nan"), float("nan")
        feat   = self.scaler.transform([[prob_r, prob_m]])
        prob   = float(self.clf.predict_proba(feat)[0, 1])
        return prob_r, prob_m, prob

    def predict(self, image_bgr: np.ndarray,
                meta_np: Optional[np.ndarray] = None,
                dataset: str = "global") -> Tuple[float, int]:
        _, _, prob = self.predict_probs_only(image_bgr, meta_np)
        thr = self.thresholds.get(dataset, self.thresholds["global"])
        return prob, int(prob >= thr)

    def close(self):
        pass  # no GPU resources to free


def run_accuracy(meta: MetaLearner, img_dir: Path) -> Dict[str, Any]:
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    csv = _SPLITS_DIR / "cls_test.csv"
    df  = pd.read_csv(csv)
    df  = resolve_dataset_paths(df, _MELANOMA_ROOT)

    results = {}
    for ds in _ALL_DS:
        sub = df[df["dataset_source"] == ds].reset_index(drop=True)
        print(f"\n  {_c(ds.upper(), _B)}  ({len(sub)} images)")
        probs, labels = [], []
        skipped = 0
        for _, row in sub.iterrows():
            path = str(row["image_path"])
            img  = cv2.imread(path)
            if img is None:
                skipped += 1
                continue
            meta_np = _build_meta(row)
            _, _, prob = meta.predict_probs_only(img, meta_np)
            import math
            if math.isnan(prob):
                skipped += 1
                continue
            probs.append(prob)
            labels.append(int(row["label_str"] == "mel"))

        if skipped:
            print(f"    {_c(f'[WARN] {skipped} images skipped (unreadable)', _Y)}")
        if not probs:
            print(f"    {_c('No images loaded — skipping', _RE)}")
            continue

        probs  = np.array(probs)
        labels = np.array(labels)
        thr    = meta.thresholds.get(ds, meta.thresholds["global"])

        preds = (probs >= thr).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1   = 2*prec*sens/(prec+sens) if (prec+sens) > 0 else 0.0
        f2   = 5*prec*sens/(4*prec+sens) if (4*prec+sens) > 0 else 0.0
        acc  = (tp+tn)/(tp+tn+fp+fn)
        auc  = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.0

        # sweep thresholds for the F2-optimal one
        best_thr, best_f2 = thr, f2
        for t in _THRESHOLDS:
            p2 = (probs >= t).astype(int)
            tp2 = int(((p2==1)&(labels==1)).sum())
            fp2 = int(((p2==1)&(labels==0)).sum())
            fn2 = int(((p2==0)&(labels==1)).sum())
            pr2 = tp2/(tp2+fp2) if (tp2+fp2) > 0 else 0
            se2 = tp2/(tp2+fn2) if (tp2+fn2) > 0 else 0
            f22 = 5*pr2*se2/(4*pr2+se2) if (4*pr2+se2) > 0 else 0
            if f22 > best_f2:
                best_f2, best_thr = f22, t

        opt_preds = (probs >= best_thr).astype(int)
        otp = int(((opt_preds==1)&(labels==1)).sum())
        otn = int(((opt_preds==0)&(labels==0)).sum())
        ofp = int(((opt_preds==1)&(labels==0)).sum())
        ofn = int(((opt_preds==0)&(labels==1)).sum())
        opr = otp/(otp+ofp) if (otp+ofp)>0 else 0
        ose = otp/(otp+ofn) if (otp+ofn)>0 else 0
        of1 = 2*opr*ose/(opr+ose) if (opr+ose)>0 else 0
        of2 = 5*opr*ose/(4*opr+ose) if (4*opr+ose)>0 else 0
        oac = (otp+otn)/(otp+otn+ofp+ofn)

        print(f"    AUC={auc:.4f}  Sen={sens:.4f}  Spe={spec:.4f}  F2={f2:.4f}  "
              f"(thr={thr:.2f})")

        results[ds] = dict(
            backend="PT_CPU",
            dataset=ds, n_total=len(labels), n_mel=int(labels.sum()),
            threshold_deployed=thr, threshold_optimal=best_thr,
            auc_meta=auc,
            deployed_sensitivity=sens, deployed_specificity=spec,
            deployed_precision=prec, deployed_f1=f1, deployed_f2=f2,
            deployed_accuracy=acc,
            deployed_tp=tp, deployed_tn=tn, deployed_fp=fp, deployed_fn=fn,
            optimal_sensitivity=ose, optimal_specificity=(otn/(otn+ofp) if (otn+ofp)>0 else 0),
            optimal_precision=opr, optimal_f1=of1, optimal_f2=of2, optimal_accuracy=oac,
            optimal_tp=otp, optimal_tn=otn, optimal_fp=ofp, optimal_fn=ofn,
        )
    return results


def run_latency(meta: MetaLearner, images: List[np.ndarray],
                warmup: int, runs: int) -> Dict[str, Any]:
    dummy_meta = _NEUTRAL_META.copy()

    print(f"\n{_SEP2}")
    print(f"  {_c(f'Latency ({warmup} warmup + {runs} runs × {len(images)} images)', _D)}")

    for _ in range(warmup):
        meta.predict_probs_only(images[0], dummy_meta)

    times_r50 = []; times_mfn = []; times_meta = []; times_total = []

    for _ in range(runs):
        for img in images:
            chw = _preprocess(img)
            x   = torch.from_numpy(chw).unsqueeze(0)
            m   = torch.from_numpy(dummy_meta).unsqueeze(0)

            t0 = time.perf_counter()
            with torch.no_grad():
                out_r = meta.r50.model(x)
            prob_r = (float(torch.sigmoid(out_r).squeeze())
                      if meta.r50._binary
                      else float(torch.softmax(out_r, dim=1)[0, 1]))
            t1 = time.perf_counter()

            with torch.no_grad():
                out_m = meta.mfn.model(x, m)
            prob_m = (float(torch.sigmoid(out_m).squeeze())
                      if meta.mfn._binary
                      else float(torch.softmax(out_m, dim=1)[0, 1]))
            t2 = time.perf_counter()

            feat = meta.scaler.transform([[prob_r, prob_m]])
            meta.clf.predict_proba(feat)
            t3 = time.perf_counter()

            times_r50.append((t1 - t0) * 1e3)
            times_mfn.append((t2 - t1) * 1e3)
            times_meta.append((t3 - t2) * 1e3)
            times_total.append((t3 - t0) * 1e3)

    def _stats(arr):
        a = np.array(arr)
        return {"median": float(np.median(a)), "mean": float(np.mean(a)),
                "std": float(np.std(a)),  "p95": float(np.percentile(a, 95))}

    lat = {
        "resnet50":    _stats(times_r50),
        "medfusionnet": _stats(times_mfn),
        "meta_learner": _stats(times_meta),
        "total":       _stats(times_total),
        "fps":         1000.0 / float(np.median(times_total)),
    }
    return lat


def _make_plots(acc_results: Dict, lat: Optional[Dict], plot_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    })

    _COL = "#9467BD"   # purple for PT-CPU
    DS_ORDER = ["ham10000", "isic2019", "isic2020"]
    DS_LABEL = {"ham10000": "HAM10000", "isic2019": "ISIC-2019", "isic2020": "ISIC-2020"}
    DS_COL   = {"ham10000": "#4C72B0", "isic2019": "#DD8452", "isic2020": "#55A868"}
    plot_dir.mkdir(parents=True, exist_ok=True)

    def _save(fig, name):
        p = plot_dir / name
        fig.tight_layout()
        fig.savefig(p, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  plot → {p.relative_to(_ROOT)}")

    if acc_results:
        ds_avail = [d for d in DS_ORDER if d in acc_results]
        ds_labels = [DS_LABEL[d] for d in ds_avail]
        xs = np.arange(len(ds_avail))

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("PT CPU: Meta-Learner Performance")
        for ax, col2, ylabel in [
            (axes[0], "deployed_sensitivity", "Sensitivity"),
            (axes[1], "deployed_specificity", "Specificity"),
        ]:
            vals = [acc_results[d][col2] for d in ds_avail]
            bars = ax.bar(xs, vals, color=[DS_COL[d] for d in ds_avail], zorder=3)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, v+0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=9)
            if ylabel == "Sensitivity":
                ax.axhline(0.85, color="#C44E52", linestyle="--", lw=1.2, label="Target >=0.85")
                ax.legend(fontsize=9)
            ax.set_xticks(xs); ax.set_xticklabels(ds_labels)
            ax.set_ylim(0, 1.12); ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} @ deployed threshold")
            ax.grid(axis="y", alpha=0.35)
        _save(fig, "01_sensitivity_specificity.png")

        fig, ax = plt.subplots(figsize=(8, 5))
        vals = [acc_results[d]["auc_meta"] for d in ds_avail]
        bars = ax.bar(xs, vals, color=[DS_COL[d] for d in ds_avail], zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.003,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(xs); ax.set_xticklabels(ds_labels)
        ax.set_ylim(0.8, 1.05); ax.set_ylabel("AUC-ROC")
        ax.set_title("Meta-Learner AUC-ROC: PT CPU")
        ax.grid(axis="y", alpha=0.35)
        _save(fig, "02_auc.png")

    if lat:
        components  = ["resnet50", "medfusionnet", "meta_learner"]
        comp_labels = ["ResNet-50", "MedFusionNet", "Meta-Learner"]
        comp_cols   = ["#4C72B0", "#DD8452", "#55A868"]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Deployment Pipeline Latency: PT CPU")

        ax = axes[0]
        bottoms = np.zeros(1)
        for comp, label, col in zip(components, comp_labels, comp_cols):
            val = np.array([lat[comp]["median"]])
            ax.bar([0], val, 0.45, bottom=bottoms, label=label, color=col, zorder=3)
            if val[0] > 5.0:
                ax.text(0, float(bottoms[0]) + float(val[0])/2,
                        f"{val[0]:.1f}ms", ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")
            bottoms += val
        fps = lat["fps"]
        total_ms = float(bottoms[0])
        ax.set_ylim(0, total_ms * 1.25)   # 25% headroom so label never clips title
        ax.text(0, total_ms + total_ms * 0.03,
                f"{total_ms:.1f}ms  ({fps:.0f} FPS)",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_xticks([0]); ax.set_xticklabels(["PT CPU"])
        ax.set_ylabel("Latency (ms)"); ax.set_title("Pipeline Latency Breakdown")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(axis="y", alpha=0.35)

        ax2 = axes[1]
        xi = np.arange(len(components))
        meds   = [lat[c]["median"] for c in components]
        p95s   = [lat[c]["p95"]    for c in components]
        spikes = [p - m for p, m in zip(p95s, meds)]
        ax2.bar(xi, meds,   0.5, color=comp_cols, zorder=3)
        ax2.bar(xi, spikes, 0.5, bottom=meds, color=comp_cols, alpha=0.35, zorder=3)
        for i, p95 in enumerate(p95s):
            ax2.text(xi[i], p95+0.5, f"p95={p95:.1f}", ha="center", va="bottom",
                     fontsize=7.5, color="gray")
        ax2.set_xticks(xi); ax2.set_xticklabels(comp_labels)
        ax2.set_ylabel("Latency (ms)"); ax2.set_title("Per-Component Median + P95 Tail")
        ax2.grid(axis="y", alpha=0.35)
        _save(fig, "03_latency_breakdown.png")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate deployment meta-learner with PyTorch (CPU) on Jetson.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs",   type=int, default=50)
    parser.add_argument("--latency-only",  action="store_true")
    parser.add_argument("--accuracy-only", action="store_true")
    args = parser.parse_args()

    do_acc = not args.latency_only
    do_lat = not args.accuracy_only

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_dir = _OUT_DIR / "plots"

    print()
    print(_SEP)
    print(_c("  Deployment Evaluation: PT CPU Meta-Learner", _B, _W))
    print(_SEP2)
    print(f"  Backend  : {_c('PyTorch CPU', _C)}")
    print(f"  Output   : {_c(str(_OUT_DIR.relative_to(_ROOT)), _D)}")
    print(_SEP)

    print(f"\n  {_c('Loading PT models ...', _D)}")
    try:
        meta = MetaLearner(_DEPLOY_DIR / "meta_learner.pkl")
        print(f"  {_c('✓', _G)} ResNet-50 PT loaded")
        print(f"  {_c('✓', _G)} MedFusionNet PT loaded")
        print(f"  {_c('✓', _G)} Meta-learner loaded")
        print(f"  Thresholds : {meta.thresholds}")
    except Exception as e:
        print(f"  {_c('✗ Failed to load models:', _RE)} {e}")
        traceback.print_exc()
        return

    acc_results = {}
    lat = None

    if do_acc:
        img_dir = _MELANOMA_ROOT
        print(f"\n{_SEP2}")
        print(f"  {_c('Accuracy evaluation', _B)}  (images from {img_dir})")
        if not img_dir.exists():
            print(f"  {_c('[WARN] melanoma_data path not found — skipping accuracy', _Y)}")
        else:
            try:
                acc_results = run_accuracy(meta, img_dir)
            except Exception as e:
                print(f"  {_c('✗ Accuracy failed:', _RE)} {e}")
                traceback.print_exc()
        print()

    if do_lat:
        # timing uses synthetic images — latency doesn't depend on content
        rng = np.random.default_rng(42)
        images = [rng.integers(0, 256, (300, 300, 3), dtype=np.uint8) for _ in range(10)]
        try:
            lat = run_latency(meta, images, args.warmup, args.runs)
            print(f"  ResNet-50    : {lat['resnet50']['median']:.2f}ms  "
                  f"(p95={lat['resnet50']['p95']:.2f}ms)")
            print(f"  MedFusionNet : {lat['medfusionnet']['median']:.2f}ms  "
                  f"(p95={lat['medfusionnet']['p95']:.2f}ms)")
            print(f"  Meta-learner : {lat['meta_learner']['median']:.2f}ms  "
                  f"(CPU — scaler + logistic)")
            _tot = f'{lat["total"]["median"]:.2f}ms'
            _fps = f'{lat["fps"]:.1f} FPS'
            print(f"  {_c('Total pipeline', _B)} : "
                  f"{_c(_tot, _B, _G)}  {_c(_fps, _B, _G)}")
        except Exception as e:
            print(f"  {_c('✗ Latency failed:', _RE)} {e}")
            traceback.print_exc()
        print()

    import pandas as pd
    xlsx_path = _OUT_DIR / "evaluation_pt.xlsx"
    rows = []
    if acc_results:
        for ds, r in acc_results.items():
            row = {"backend": "PT_CPU", **r}
            if lat:
                row["latency_total_ms"] = lat["total"]["median"]
                row["fps"]              = lat["fps"]
                row["latency_r50_ms"]   = lat["resnet50"]["median"]
                row["latency_mfn_ms"]   = lat["medfusionnet"]["median"]
            rows.append(row)

    if not rows and not lat:
        print(f"  {_c('[WARN] Nothing to save', _Y)}")
    else:
        mode = "a" if xlsx_path.exists() else "w"
        kw   = {"if_sheet_exists": "replace"} if mode == "a" else {}
        with pd.ExcelWriter(xlsx_path, engine="openpyxl", mode=mode, **kw) as writer:
            if rows:
                pd.DataFrame(rows).to_excel(writer, sheet_name="Results", index=False)
            if lat:
                lat_rows = [{"component": k, **v}
                            for k, v in lat.items() if isinstance(v, dict)]
                lat_rows.append({"component": "fps_total", "median": lat["fps"]})
                pd.DataFrame(lat_rows).to_excel(writer, sheet_name="Latency", index=False)
        print(f"  {_c('Excel →', _G)} {xlsx_path.relative_to(_ROOT)}")

    _make_plots(acc_results, lat, plot_dir)

    print()
    print(_SEP)
    print(f"  {_c('Done.', _G, _B)}  Output: {_c(str(_OUT_DIR.relative_to(_ROOT)), _D)}")
    print(_SEP)
    print()


if __name__ == "__main__":
    main()
