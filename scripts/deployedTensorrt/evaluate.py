"""Evaluates the deployment meta-learner (resnet50 + medfusionnet -> logistic regression) on Jetson via TensorRT.

Usage: python3 scripts/deployedTensorrt/evaluate.py [--precision fp16|fp32] [--latency-only] [--accuracy-only]
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

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.utils.config import load_config
from src.utils.io import resolve_dataset_paths

_BASE_CFG   = load_config()
_ROOT       = Path(__file__).resolve().parents[3]
_DEPLOY_DIR = _ROOT / "outputs" / "ablation_noseg" / "meta" / "deployment"
_TRT_DIR    = _DEPLOY_DIR / "tensorrt"
_SPLITS_DIR = _ROOT / "data_splits"
# raw dataset location — override via MELANOMA_DATA_DIR (see configs/base.yaml)
_MELANOMA_ROOT = Path(_BASE_CFG.paths.melanoma_data)

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
    age     = float(row.get("age_approx", 50) or 50) / 100.0
    sex_raw = str(row.get("sex", "")).lower()
    sex     = 1.0 if sex_raw == "male" else (0.0 if sex_raw == "female" else 0.5)
    site    = np.zeros(len(_SITE_CATS), dtype=np.float32)
    site_raw = str(row.get("anatom_site_general_challenge", ""))
    for i, cat in enumerate(_SITE_CATS):
        if site_raw == cat:
            site[i] = 1.0
    return np.concatenate([[age, sex], site]).astype(np.float32)


def _load_cudart():
    import ctypes, glob
    candidates = (
        glob.glob("/usr/local/cuda*/lib64/libcudart.so.12*")
        + glob.glob("/usr/local/cuda*/targets/aarch64-linux/lib/libcudart.so.12*")
        + ["/usr/lib/aarch64-linux-gnu/libcudart.so.12"]
    )
    for path in sorted(candidates):
        try:
            lib = ctypes.CDLL(path)
            ptr = ctypes.c_void_p()
            if lib.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(4)) == 0:
                lib.cudaFree(ctypes.c_void_p(ptr.value))
                return lib
        except OSError:
            continue
    raise RuntimeError("Cannot find libcudart.so.12")

_CUDART: Any = None
def _cudart():
    global _CUDART
    if _CUDART is None:
        _CUDART = _load_cudart()
    return _CUDART


class _TRTModel:
    """Single TRT engine — image-only or image+metadata inputs."""

    def __init__(self, engine_path: Path, with_meta: bool = False, input_size: int = 224):
        import tensorrt as trt, ctypes
        self._ct        = ctypes
        self.with_meta  = with_meta
        self.input_size = input_size
        _cudart()

        logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self._v10    = not hasattr(self.engine, "get_binding_index")

        if self._v10:
            n = self.engine.num_io_tensors
            self._in_names  = [self.engine.get_tensor_name(i) for i in range(n)
                               if self.engine.get_tensor_mode(self.engine.get_tensor_name(i))
                               == trt.TensorIOMode.INPUT]
            self._out_names = [self.engine.get_tensor_name(i) for i in range(n)
                               if self.engine.get_tensor_mode(self.engine.get_tensor_name(i))
                               == trt.TensorIOMode.OUTPUT]
            img_shape   = tuple(self.engine.get_tensor_shape(self._in_names[0]))
            out_shape   = tuple(self.engine.get_tensor_shape(self._out_names[0]))
            meta_shape  = (tuple(self.engine.get_tensor_shape(self._in_names[1]))
                           if with_meta and len(self._in_names) > 1 else None)
        else:
            img_shape  = tuple(self.engine.get_binding_shape(
                               self.engine.get_binding_index("image")))
            out_idx    = next(i for i in range(self.engine.num_bindings)
                              if not self.engine.binding_is_input(i))
            out_shape  = tuple(self.engine.get_binding_shape(out_idx))
            meta_shape = (tuple(self.engine.get_binding_shape(
                               self.engine.get_binding_index("metadata")))
                          if with_meta else None)

        self._h_img  = np.zeros(img_shape,  dtype=np.float32)
        self._h_out  = np.zeros(out_shape,  dtype=np.float32)
        self._h_meta = np.zeros(meta_shape, dtype=np.float32) if meta_shape else None
        self._d_img  = self._malloc(self._h_img.nbytes)
        self._d_out  = self._malloc(self._h_out.nbytes)
        self._d_meta = self._malloc(self._h_meta.nbytes) if self._h_meta is not None else None

    def _malloc(self, n):
        ptr = self._ct.c_void_p()
        _cudart().cudaMalloc(self._ct.byref(ptr), self._ct.c_size_t(n))
        return ptr.value

    def _h2d(self, d, h):
        _cudart().cudaMemcpy(self._ct.c_void_p(d),
                             h.ctypes.data_as(self._ct.c_void_p),
                             self._ct.c_size_t(h.nbytes), self._ct.c_int(1))

    def _d2h(self, h, d):
        _cudart().cudaMemcpy(h.ctypes.data_as(self._ct.c_void_p),
                             self._ct.c_void_p(d),
                             self._ct.c_size_t(h.nbytes), self._ct.c_int(2))

    def predict_prob(self, image_bgr: np.ndarray,
                     meta_np: Optional[np.ndarray] = None) -> float:
        chw = _preprocess(image_bgr, self.input_size)
        np.copyto(self._h_img, chw[np.newaxis])
        self._h2d(self._d_img, self._h_img)

        if self._h_meta is not None:
            m = meta_np if meta_np is not None else _NEUTRAL_META
            np.copyto(self._h_meta, m[np.newaxis])
            self._h2d(self._d_meta, self._h_meta)

        if self._v10:
            self.context.set_tensor_address(self._in_names[0],  self._d_img)
            if self._d_meta is not None and len(self._in_names) > 1:
                self.context.set_tensor_address(self._in_names[1], self._d_meta)
            self.context.set_tensor_address(self._out_names[0], self._d_out)
            self.context.execute_async_v3(stream_handle=0)
        else:
            n = self.engine.num_bindings
            bindings = [0] * n
            bindings[self.engine.get_binding_index("image")] = self._d_img
            if self._d_meta is not None:
                bindings[self.engine.get_binding_index("metadata")] = self._d_meta
            bindings[next(i for i in range(n)
                          if not self.engine.binding_is_input(i))] = self._d_out
            self.context.execute_async_v2(bindings, stream_handle=0)

        _cudart().cudaDeviceSynchronize()
        self._d2h(self._h_out, self._d_out)

        logits = self._h_out[0]
        exp    = np.exp(logits - logits.max())
        return float(exp[1] / exp.sum())

    def close(self):
        for ptr in [self._d_img, self._d_out, self._d_meta]:
            if ptr:
                _cudart().cudaFree(self._ct.c_void_p(ptr))


class MetaLearner:
    """Combines probabilities from two TRT models via a trained LogisticRegression."""

    def __init__(self, pkl_path: Path, engine_dir: Path, precision: str):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(pkl_path, "rb") as f:
                bundle = pickle.load(f)

        self.model1_key  = bundle["model1"]
        self.model2_key  = bundle["model2"]
        self.scaler      = bundle["scaler"]
        self.clf         = bundle["clf"]
        self.thresholds  = bundle["thresholds"]

        # older pickles were trained on an sklearn version without this attribute
        if not hasattr(self.clf, "multi_class"):
            self.clf.multi_class = "auto"

        self.m1 = _TRTModel(engine_dir / f"resnet50_none_sens_{precision}.engine",
                             with_meta=False)
        self.m2 = _TRTModel(engine_dir / f"medfusionnet_none_sens_{precision}.engine",
                             with_meta=True)

    def predict(self, image_bgr: np.ndarray,
                meta_np: Optional[np.ndarray] = None,
                dataset: str = "global") -> Tuple[float, float]:
        """Returns (meta_prob, prediction) using dataset-specific threshold."""
        p1 = self.m1.predict_prob(image_bgr)
        p2 = self.m2.predict_prob(image_bgr, meta_np)
        feat    = self.scaler.transform([[p1, p2]])
        prob    = float(self.clf.predict_proba(feat)[0, 1])
        thr     = self.thresholds.get(dataset, self.thresholds["global"])
        pred    = int(prob >= thr)
        return prob, pred

    def predict_probs_only(self, image_bgr: np.ndarray,
                           meta_np: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
        """Returns (prob_resnet, prob_medfusion, meta_prob) — no threshold."""
        p1   = self.m1.predict_prob(image_bgr)
        p2   = self.m2.predict_prob(image_bgr, meta_np)
        feat = self.scaler.transform([[p1, p2]])
        prob = float(self.clf.predict_proba(feat)[0, 1])
        return p1, p2, prob

    def close(self):
        self.m1.close()
        self.m2.close()


def _compute_metrics(probs, labels, threshold) -> Dict:
    preds = (np.array(probs) >= threshold).astype(int)
    labels = np.array(labels)
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
    return dict(threshold=round(threshold, 2),
                sensitivity=round(sens, 4), specificity=round(spec, 4),
                precision=round(prec, 4), f1=round(f1, 4), f2=round(f2, 4),
                accuracy=round(acc, 4),
                tp=tp, tn=tn, fp=fp, fn=fn,
                n_mel=int(labels.sum()), n_total=int(len(labels)))


def _best_f2_threshold(probs, labels) -> Tuple[float, Dict]:
    best_thr, best_f2 = _THRESHOLDS[0], -1.0
    for thr in _THRESHOLDS:
        m = _compute_metrics(probs, labels, thr)
        if m["f2"] > best_f2:
            best_f2, best_thr = m["f2"], thr
    return best_thr, _compute_metrics(probs, labels, best_thr)


def _auc(probs, labels) -> float:
    from sklearn.metrics import roc_auc_score
    try:
        return round(float(roc_auc_score(labels, probs)), 4)
    except Exception:
        return float("nan")


def run_accuracy(meta: MetaLearner) -> Dict[str, Dict]:
    import pandas as pd
    results = {}

    for ds in _ALL_DS:
        df = pd.read_csv(_SPLITS_DIR / "cls_test.csv")
        df = df[df["dataset_source"] == ds].copy()
        df["binary_label"] = (df["label_str"] == "mel").astype(int)
        df = resolve_dataset_paths(df, _MELANOMA_ROOT)
        df = df.reset_index(drop=True)

        print(f"  {_c(ds, _B)}  {len(df)} images  "
              f"(mel={int(df['binary_label'].sum())})")

        probs1, probs2, meta_probs, labels = [], [], [], []
        n_fail = 0

        for i, row in df.iterrows():
            img = cv2.imread(str(row["image_path"]))
            if img is None:
                n_fail += 1
                continue
            meta_np = _build_meta(row)
            try:
                p1, p2, pm = meta.predict_probs_only(img, meta_np)
                probs1.append(p1); probs2.append(p2)
                meta_probs.append(pm); labels.append(int(row["binary_label"]))
            except Exception:
                n_fail += 1

            if (i + 1) % 200 == 0:
                print(f"    {i+1}/{len(df)} processed ...", flush=True)

        if n_fail:
            print(f"    {_c(f'[WARN] {n_fail} images failed to load', _Y)}")

        if not labels:
            print(f"    {_c('No images loaded — skipping', _RE)}")
            continue

        ds_thr  = meta.thresholds.get(ds, meta.thresholds["global"])
        fixed   = _compute_metrics(meta_probs, labels, ds_thr)
        opt_thr, optimal = _best_f2_threshold(meta_probs, labels)
        auc     = _auc(meta_probs, labels)
        auc1    = _auc(probs1, labels)
        auc2    = _auc(probs2, labels)

        print(f"    AUC={auc:.4f}  sens={fixed['sensitivity']:.4f}  "
              f"spec={fixed['specificity']:.4f}  "
              f"@thr={ds_thr:.2f}  (opt_thr={opt_thr:.2f})")

        results[ds] = dict(
            dataset=ds, n_total=fixed["n_total"], n_mel=fixed["n_mel"],
            threshold_deployed=ds_thr, threshold_optimal=opt_thr,
            auc_meta=auc, auc_resnet50=auc1, auc_medfusionnet=auc2,
            **{f"deployed_{k}": v for k, v in fixed.items()
               if k not in ("threshold", "n_total", "n_mel")},
            **{f"optimal_{k}": v for k, v in optimal.items()
               if k not in ("threshold", "n_total", "n_mel")},
        )

    return results


def run_latency(meta: MetaLearner, images: List[np.ndarray],
                n_warmup: int, n_runs: int) -> Dict:
    dummy_meta = _NEUTRAL_META
    for img in images:
        for _ in range(n_warmup):
            meta.predict_probs_only(img, dummy_meta)

    t_r, t_m, t_meta, t_total = [], [], [], []
    for img in images:
        for _ in range(n_runs):
            t0 = time.perf_counter()
            p1 = meta.m1.predict_prob(img)
            t1 = time.perf_counter()
            p2 = meta.m2.predict_prob(img, dummy_meta)
            t2 = time.perf_counter()
            feat = meta.scaler.transform([[p1, p2]])
            meta.clf.predict_proba(feat)
            t3 = time.perf_counter()
            t_r.append((t1 - t0) * 1e3)
            t_m.append((t2 - t1) * 1e3)
            t_meta.append((t3 - t2) * 1e3)
            t_total.append((t3 - t0) * 1e3)

    def _s(vals):
        a = np.array(vals)
        return dict(median=round(float(np.median(a)), 2),
                    mean=round(float(np.mean(a)), 2),
                    std=round(float(np.std(a)), 2),
                    p95=round(float(np.percentile(a, 95)), 2))

    return dict(
        resnet50   = _s(t_r),
        medfusionnet = _s(t_m),
        meta_learner = _s(t_meta),
        total        = _s(t_total),
        fps          = round(1000.0 / float(np.median(t_total)), 1),
    )


def _make_plots(acc_results: Dict, lat: Dict, precision: str, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300,
                         "axes.spines.top": False, "axes.spines.right": False})
    out_dir.mkdir(parents=True, exist_ok=True)
    _DS_COL = {"ham10000": "#4C72B0", "isic2019": "#DD8452", "isic2020": "#55A868"}
    _DS_LBL = {"ham10000": "HAM10000", "isic2019": "ISIC-2019", "isic2020": "ISIC-2020"}

    def _save(fig, name):
        fig.tight_layout()
        fig.savefig(out_dir / name, bbox_inches="tight")
        plt.close(fig)
        print(f"  plot → {(out_dir / name).relative_to(_ROOT)}")

    ds_list = [d for d in _ALL_DS if d in acc_results]

    if ds_list:
        # AUC: meta vs. individual models
        fig, ax = plt.subplots(figsize=(8, 5))
        xs = np.arange(len(ds_list)); w = 0.25
        ax.bar(xs - w, [acc_results[d]["auc_resnet50"]    for d in ds_list], w,
               label="ResNet-50",    color="#55A868", zorder=3)
        ax.bar(xs,     [acc_results[d]["auc_medfusionnet"] for d in ds_list], w,
               label="MedFusionNet", color="#DD8452", zorder=3)
        ax.bar(xs + w, [acc_results[d]["auc_meta"]         for d in ds_list], w,
               label="Meta-Learner", color="#4C72B0", zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels([_DS_LBL.get(d, d) for d in ds_list])
        ax.set_ylim(0.5, 1.05); ax.set_ylabel("AUC-ROC")
        ax.set_title(f"AUC-ROC: ResNet-50 vs MedFusionNet vs Meta-Learner\n(TRT {precision.upper()})")
        ax.legend(); ax.grid(axis="y", alpha=0.35)
        _save(fig, "01_auc_comparison.png")

        # sensitivity & specificity at the deployed threshold
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, metric, title in [
            (axes[0], "deployed_sensitivity", "Sensitivity @ deployed threshold"),
            (axes[1], "deployed_specificity", "Specificity @ deployed threshold"),
        ]:
            vals = [acc_results[d].get(metric, 0) for d in ds_list]
            bars = ax.bar(np.arange(len(ds_list)), vals,
                          color=[_DS_COL[d] for d in ds_list], zorder=3)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=9)
            if "sensitivity" in metric:
                ax.axhline(0.85, color="red", ls="--", lw=1.2, alpha=0.7, label="Target ≥0.85")
                ax.legend()
            ax.set_xticks(np.arange(len(ds_list)))
            ax.set_xticklabels([_DS_LBL.get(d, d) for d in ds_list])
            ax.set_ylim(0, 1.1); ax.set_ylabel(metric.replace("deployed_", "").title())
            ax.set_title(title); ax.grid(axis="y", alpha=0.35)
        fig.suptitle(f"TRT {precision.upper()}: Meta-Learner Performance")
        _save(fig, "02_sensitivity_specificity.png")

        # deployed vs. optimal threshold
        fig, ax = plt.subplots(figsize=(9, 5))
        xs = np.arange(len(ds_list)); w = 0.3
        dep_sens = [acc_results[d]["deployed_sensitivity"] for d in ds_list]
        opt_sens = [acc_results[d]["optimal_sensitivity"]  for d in ds_list]
        dep_f2   = [acc_results[d]["deployed_f2"]          for d in ds_list]
        opt_f2   = [acc_results[d]["optimal_f2"]           for d in ds_list]
        ax.bar(xs - w/2, dep_sens, w, label="Sensitivity (deployed thr)", color="#4C72B0", zorder=3)
        ax.bar(xs + w/2, opt_sens, w, label="Sensitivity (optimal thr)", color="#4C72B0",
               alpha=0.45, hatch="//", zorder=3)
        ax2 = ax.twinx()
        ax2.plot(xs - w/2, dep_f2, "D--", color="#C44E52", ms=7, label="F2 (deployed)")
        ax2.plot(xs + w/2, opt_f2, "o--", color="#DD8452", ms=7, label="F2 (optimal)")
        ax2.set_ylabel("F2 score"); ax2.set_ylim(0, 1.1)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{_DS_LBL.get(d,d)}\n(thr={acc_results[d]['threshold_deployed']:.2f}→{acc_results[d]['threshold_optimal']:.2f})"
                            for d in ds_list])
        ax.set_ylim(0, 1.1); ax.set_ylabel("Sensitivity")
        ax.set_title(f"Deployed vs Optimal Threshold: {precision.upper()}")
        lines1, lbls1 = ax.get_legend_handles_labels()
        lines2, lbls2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, lbls1 + lbls2, loc="lower right", fontsize=8)
        ax.grid(axis="y", alpha=0.35)
        _save(fig, "03_threshold_comparison.png")

    if lat:
        components = ["resnet50", "medfusionnet", "meta_learner"]
        labels     = ["ResNet-50\n(TRT)", "MedFusionNet\n(TRT)", "Meta-Learner\n(CPU)"]
        medians    = [lat[c]["median"] for c in components]
        p95s       = [lat[c]["p95"]    for c in components]
        cols       = ["#4C72B0", "#DD8452", "#55A868"]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        axes[0].bar([0], [sum(medians)], 0.5,
                    color="#4C72B0", label=f"Total: {sum(medians):.1f}ms", zorder=3)
        bottom = 0
        for lbl, val, col in zip(labels, medians, cols):
            axes[0].bar([0], [val], 0.5, bottom=bottom, color=col,
                        label=f"{lbl.split(chr(10))[0]}: {val:.1f}ms", zorder=3, alpha=0.85)
            axes[0].text(0, bottom + val/2, f"{val:.1f}ms",
                         ha="center", va="center", fontsize=9, color="white", fontweight="bold")
            bottom += val
        axes[0].set_xlim(-0.5, 0.5); axes[0].set_xticks([])
        axes[0].set_ylabel("Latency (ms)")
        axes[0].set_title(f"Pipeline Latency Breakdown\n({lat['fps']:.1f} FPS end-to-end)")
        axes[0].legend(loc="upper right", fontsize=8)
        axes[0].grid(axis="y", alpha=0.35)

        xs = np.arange(len(components))
        spikes = [p - m for p, m in zip(p95s, medians)]
        axes[1].bar(xs, medians, 0.5, color=cols, zorder=3, label="Median")
        axes[1].bar(xs, spikes, 0.5, bottom=medians, color="#C44E52",
                    alpha=0.5, zorder=3, label="P95 tail")
        for i, (med, p95) in enumerate(zip(medians, p95s)):
            axes[1].text(i, p95 + 0.2, f"p95={p95:.1f}", ha="center", fontsize=8)
        axes[1].set_xticks(xs); axes[1].set_xticklabels(labels)
        axes[1].set_ylabel("Latency (ms)")
        axes[1].set_title("Per-Component Median + P95 Tail")
        axes[1].legend(); axes[1].grid(axis="y", alpha=0.35)

        fig.suptitle(f"Deployment Pipeline Latency: TRT {precision.upper()}")
        _save(fig, "04_latency_breakdown.png")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate deployment meta-learner system using TensorRT on Jetson.")
    parser.add_argument("--precision",     default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--runs",          type=int, default=100)
    parser.add_argument("--warmup",        type=int, default=20)
    parser.add_argument("--images",        default="scripts/test_images",
                        help="Directory of latency test images")
    parser.add_argument("--latency-only",  action="store_true")
    parser.add_argument("--accuracy-only", action="store_true")
    args = parser.parse_args()

    out_dir  = _TRT_DIR
    plot_dir = out_dir / "plots" / args.precision
    out_dir.mkdir(parents=True, exist_ok=True)

    print()
    print(_SEP)
    print(_c("  Deployment Evaluation — TRT Meta-Learner", _B, _W))
    print(_SEP2)
    print(f"  Precision  : {_c(args.precision.upper(), _B, _C)}")
    print(f"  Engines    : {_c(str(_TRT_DIR.relative_to(_ROOT)), _D)}")
    print(f"  Meta-model : {_c('scaler + LogisticRegression', _D)}")
    print(_SEP)
    print()

    print(f"  Loading engines ({args.precision.upper()}) ...")
    try:
        meta = MetaLearner(_DEPLOY_DIR / "meta_learner.pkl", _TRT_DIR, args.precision)
        print(f"  {_c('✓', _G)} ResNet-50 TRT loaded")
        print(f"  {_c('✓', _G)} MedFusionNet TRT loaded")
        print(f"  {_c('✓', _G)} Meta-learner loaded")
        print(f"  Thresholds : {meta.thresholds}")
    except Exception as e:
        print(f"  {_c('✗ Failed to load engines:', _RE)} {e}")
        traceback.print_exc()
        return
    print()

    acc_results: Dict = {}
    lat: Dict = {}

    if not args.latency_only:
        print(_c("  ── Accuracy Evaluation ──────────────────────────────────", _D))
        print()
        try:
            acc_results = run_accuracy(meta)
        except Exception as e:
            print(f"  {_c('✗ Accuracy eval failed:', _RE)} {e}")
            traceback.print_exc()
        print()

    if not args.accuracy_only:
        img_dir = _ROOT / args.images
        images  = [img for img in (cv2.imread(str(p))
                   for p in sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")))
                   if img is not None]
        if not images:
            print(f"  {_c('[WARN] No images in ' + str(img_dir) + ' — using synthetic', _Y)}")
            images = [np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)]

        print(_c(f"  ── Latency ({args.warmup} warmup + {args.runs} runs × {len(images)} images) ──", _D))
        print()
        try:
            lat = run_latency(meta, images, args.warmup, args.runs)
            print(f"  ResNet-50     : {lat['resnet50']['median']:.2f}ms  "
                  f"(p95={lat['resnet50']['p95']:.2f}ms)")
            print(f"  MedFusionNet  : {lat['medfusionnet']['median']:.2f}ms  "
                  f"(p95={lat['medfusionnet']['p95']:.2f}ms)")
            print(f"  Meta-learner  : {lat['meta_learner']['median']:.2f}ms  "
                  f"(CPU — scaler + logistic)")
            _tot_str = f'{lat["total"]["median"]:.2f}ms'
            _fps_str = f'{lat["fps"]:.1f} FPS'
            print(f"  {_c('Total pipeline', _B)}  : "
                  f"{_c(_tot_str, _B, _G)}  "
                  f"{_c(_fps_str, _B, _G)}")
        except Exception as e:
            print(f"  {_c('✗ Latency failed:', _RE)} {e}")
            traceback.print_exc()
        print()

    meta.close()

    import pandas as pd
    xlsx_path = out_dir / f"evaluation_trt_{args.precision}.xlsx"
    rows = []

    if acc_results:
        for ds, r in acc_results.items():
            row = {"precision": args.precision.upper(), **r}
            if lat:
                row["latency_total_ms"]  = lat["total"]["median"]
                row["fps"]               = lat["fps"]
                row["latency_r50_ms"]    = lat["resnet50"]["median"]
                row["latency_mfn_ms"]    = lat["medfusionnet"]["median"]
            rows.append(row)

    # append mode keeps sheets from earlier runs (e.g. the other precision) intact
    mode = "a" if xlsx_path.exists() else "w"
    if_exists = "replace"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl",
                        mode=mode, if_sheet_exists=if_exists if mode == "a" else None) as writer:
        if rows:
            pd.DataFrame(rows).to_excel(writer, sheet_name="Results", index=False)
        if lat:
            lat_rows = [
                {"component": k, **v}
                for k, v in lat.items()
                if isinstance(v, dict)
            ]
            lat_rows.append({"component": "fps_total", "median": lat["fps"]})
            pd.DataFrame(lat_rows).to_excel(writer, sheet_name="Latency", index=False)
    print(f"  {_c('Excel →', _G)} {xlsx_path.relative_to(_ROOT)}")

    _make_plots(acc_results, lat, args.precision, plot_dir)

    print()
    print(_SEP)
    print(f"  {_c('Done.', _G, _B)}  Output: {_c(str(out_dir.relative_to(_ROOT)), _D)}")
    print(_SEP)
    print()


if __name__ == "__main__":
    main()
