"""Benchmarks the full YOLO -> ResNet-50/MedFusionNet -> meta-learner pipeline on TRT FP16 engines.

Usage: python3 scripts/deployedTensorrt/benchmark_pipeline.py [--images DIR] [--runs N] [--warmup N]
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

_ROOT       = Path(__file__).resolve().parents[3]
_DEPLOY_DIR = _ROOT / "outputs" / "ablation_noseg" / "meta" / "deployment"
_TRT_DIR    = _DEPLOY_DIR / "tensorrt"
_DET_ENGINE = _ROOT / "outputs" / "detection" / "checkpoints" / "best_fp16.engine"
_OUT_DIR  = _TRT_DIR / "plots" / "fullPipeline"
_OUT_DIR.mkdir(parents=True, exist_ok=True)

_MEAN         = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD          = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_NEUTRAL_META = np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

_R="\033[0m"; _B="\033[1m"; _G="\033[32m"; _Y="\033[33m"
_RE="\033[31m"; _C="\033[36m"; _D="\033[2m"; _W="\033[97m"; _BL="\033[34m"
def _c(t, *codes): return "".join(codes) + t + _R
_SEP  = _c("═"*68, _BL)
_SEP2 = _c("─"*68, _D)


_CUDART_LIB: Any = None

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

def _cudart():
    global _CUDART_LIB
    if _CUDART_LIB is None:
        _CUDART_LIB = _load_cudart()
    return _CUDART_LIB


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
            img_shape  = tuple(self.engine.get_tensor_shape(self._in_names[0]))
            out_shape  = tuple(self.engine.get_tensor_shape(self._out_names[0]))
            meta_shape = (tuple(self.engine.get_tensor_shape(self._in_names[1]))
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
        chw = _preprocess_chw(image_bgr, self.input_size)
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


def _preprocess_chw(img_bgr: np.ndarray, size: int = 224) -> np.ndarray:
    big = int(size * 1.1)
    img = cv2.resize(img_bgr, (big, big))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    st  = (big - size) // 2
    img = img[st:st + size, st:st + size]
    return ((img - _MEAN) / _STD).transpose(2, 0, 1).astype(np.float32)


class _TRTYOLODetector:
    """YOLOv8 TRT FP16 engine via ctypes — no PyTorch CUDA required."""

    def __init__(self, engine_path: Path, conf_threshold: float = 0.35, imgsz: int = 640):
        import tensorrt as trt, ctypes
        self._ct = ctypes
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        _cudart()

        logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self._v10 = not hasattr(self.engine, "get_binding_index")

        if self._v10:
            n = self.engine.num_io_tensors
            self._in_names  = [self.engine.get_tensor_name(i) for i in range(n)
                               if self.engine.get_tensor_mode(self.engine.get_tensor_name(i))
                               == trt.TensorIOMode.INPUT]
            self._out_names = [self.engine.get_tensor_name(i) for i in range(n)
                               if self.engine.get_tensor_mode(self.engine.get_tensor_name(i))
                               == trt.TensorIOMode.OUTPUT]
            in_shape  = tuple(self.engine.get_tensor_shape(self._in_names[0]))
            out_shape = tuple(self.engine.get_tensor_shape(self._out_names[0]))
        else:
            in_shape  = tuple(self.engine.get_binding_shape(0))
            out_shape = tuple(self.engine.get_binding_shape(1))

        # pinned memory speeds up the host->device copy
        nbytes_in = int(np.prod(in_shape)) * 4
        self._h_in_ptr = ctypes.c_void_p()
        _cudart().cudaMallocHost(ctypes.byref(self._h_in_ptr), ctypes.c_size_t(nbytes_in))
        self._h_in = np.frombuffer(
            (ctypes.c_byte * nbytes_in).from_address(self._h_in_ptr.value),
            dtype=np.float32,
        ).reshape(in_shape)

        self._h_out = np.zeros(out_shape, dtype=np.float32)
        self._d_in  = self._malloc(nbytes_in)
        self._d_out = self._malloc(self._h_out.nbytes)

        self._canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
        self._lb_cache: Dict[Tuple[int, int], Tuple] = {}

        print("  {} YOLO FP16 loaded  in={}  out={}".format(_c("✓", _G), in_shape, out_shape))

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

    def _letterbox(self, img_bgr: np.ndarray):
        h, w = img_bgr.shape[:2]
        key = (h, w)
        if key not in self._lb_cache:
            scale = min(self.imgsz / h, self.imgsz / w)
            nh, nw = int(h * scale), int(w * scale)
            pad_y  = (self.imgsz - nh) // 2
            pad_x  = (self.imgsz - nw) // 2
            self._lb_cache[key] = (scale, nh, nw, pad_x, pad_y)
            # padding only needs to be redrawn if the resolution changes
            self._canvas[:pad_y, :]          = 114
            self._canvas[pad_y + nh:, :]     = 114
            self._canvas[:, :pad_x]          = 114
            self._canvas[:, pad_x + nw:]     = 114
        scale, nh, nw, pad_x, pad_y = self._lb_cache[key]

        cv2.resize(img_bgr, (nw, nh), dst=self._canvas[pad_y:pad_y + nh, pad_x:pad_x + nw],
                   interpolation=cv2.INTER_LINEAR)

        rgb = cv2.cvtColor(self._canvas, cv2.COLOR_BGR2RGB)
        chw = (rgb.astype(np.float32) * (1.0 / 255.0)).transpose(2, 0, 1)
        return chw, scale, pad_x, pad_y

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.45) -> list:
        bx = boxes[:, 0]; by = boxes[:, 1]
        bw = boxes[:, 2] - bx; bh = boxes[:, 3] - by
        indices = cv2.dnn.NMSBoxes(
            np.stack([bx, by, bw, bh], axis=1).tolist(),
            scores.tolist(), score_threshold=0.0, nms_threshold=iou_thr,
        )
        if isinstance(indices, np.ndarray):
            return indices.flatten().tolist()
        return [int(i[0]) for i in indices] if indices is not None else []

    def detect(self, img_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], float, bool]:
        """Returns (crop_or_None, latency_ms, detected_bool)."""
        t0 = time.perf_counter()
        h, w = img_bgr.shape[:2]

        chw, scale, pad_x, pad_y = self._letterbox(img_bgr)
        np.copyto(self._h_in, chw[np.newaxis])
        self._h2d(self._d_in, self._h_in)

        if self._v10:
            self.context.set_tensor_address(self._in_names[0],  self._d_in)
            self.context.set_tensor_address(self._out_names[0], self._d_out)
            self.context.execute_async_v3(stream_handle=0)
        else:
            self.context.execute_async_v2([self._d_in, self._d_out], stream_handle=0)

        _cudart().cudaDeviceSynchronize()
        self._d2h(self._h_out, self._d_out)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        out = self._h_out[0]
        if out.shape[0] < out.shape[1]:
            out = out.T

        scores = out[:, 4:].max(axis=1) if out.shape[1] > 5 else out[:, 4]
        mask = scores >= self.conf_threshold
        if not mask.any():
            return None, latency_ms, False

        sc = scores[mask]
        cx, cy = out[mask, 0], out[mask, 1]
        bw, bh = out[mask, 2], out[mask, 3]
        x1l, y1l = cx - bw / 2, cy - bh / 2
        x2l, y2l = cx + bw / 2, cy + bh / 2
        boxes = np.stack([x1l, y1l, x2l, y2l], axis=1)

        keep = self._nms(boxes, sc)
        if not keep:
            return None, latency_ms, False

        best = keep[int(np.argmax(sc[keep]))]
        bx1 = int(max(0, (boxes[best, 0] - pad_x) / scale))
        by1 = int(max(0, (boxes[best, 1] - pad_y) / scale))
        bx2 = int(min(w, (boxes[best, 2] - pad_x) / scale))
        by2 = int(min(h, (boxes[best, 3] - pad_y) / scale))

        if bx2 <= bx1 or by2 <= by1:
            return None, latency_ms, False

        return img_bgr[by1:by2, bx1:bx2], latency_ms, True

    def close(self):
        for ptr in [self._d_in, self._d_out]:
            if ptr:
                _cudart().cudaFree(self._ct.c_void_p(ptr))
        if self._h_in_ptr.value:
            _cudart().cudaFreeHost(self._ct.c_void_p(self._h_in_ptr.value))


class _MetaPipeline:
    def __init__(self, pkl_path: Path, engine_dir: Path, precision: str = "fp16"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(pkl_path, "rb") as f:
                bundle = pickle.load(f)
        self.scaler     = bundle["scaler"]
        self.clf        = bundle["clf"]
        self.thresholds = bundle["thresholds"]
        if not hasattr(self.clf, "multi_class"):
            self.clf.multi_class = "auto"
        self.r50 = _TRTModel(engine_dir / f"resnet50_none_sens_{precision}.engine",
                              with_meta=False)
        self.mfn = _TRTModel(engine_dir / f"medfusionnet_none_sens_{precision}.engine",
                              with_meta=True)
        print(f"  {_c('✓', _G)} ResNet-50 FP16 loaded")
        print(f"  {_c('✓', _G)} MedFusionNet FP16 loaded")
        print(f"  {_c('✓', _G)} Meta-learner loaded  thresholds={self.thresholds}")

    def infer_timed(self, crop: np.ndarray) -> Tuple[float, float, float, float, float]:
        """Returns (prob_r50, prob_mfn, meta_prob, r50_ms, mfn_ms, meta_ms)."""
        t0 = time.perf_counter()
        p1 = self.r50.predict_prob(crop)
        t1 = time.perf_counter()
        p2 = self.mfn.predict_prob(crop, _NEUTRAL_META)
        t2 = time.perf_counter()
        feat = self.scaler.transform([[p1, p2]])
        prob = float(self.clf.predict_proba(feat)[0, 1])
        t3 = time.perf_counter()
        return p1, p2, prob, (t1-t0)*1e3, (t2-t1)*1e3, (t3-t2)*1e3

    def close(self):
        self.r50.close()
        self.mfn.close()


def _stats(vals: List[float]) -> Dict:
    a = np.array(vals)
    return dict(
        median = round(float(np.median(a)), 2),
        mean   = round(float(np.mean(a)),   2),
        std    = round(float(np.std(a)),     2),
        p95    = round(float(np.percentile(a, 95)), 2),
        p99    = round(float(np.percentile(a, 99)), 2),
        min    = round(float(a.min()), 2),
        max    = round(float(a.max()), 2),
    )


def run_benchmark(yolo: _TRTYOLODetector, meta: _MetaPipeline,
                  images: List[np.ndarray],
                  n_warmup: int, n_runs: int) -> Dict:

    img0 = images[0]
    print(f"\n  Warming up ({n_warmup} passes) ...")
    for _ in range(n_warmup):
        crop, _, det = yolo.detect(img0)
        if det and crop is not None:
            meta.infer_timed(crop)
    print(f"  {_c('✓', _G)} Warmup done")

    t_yolo, t_r50, t_mfn, t_meta_clf, t_total = [], [], [], [], []
    t_classify, n_detected, n_total = [], 0, 0

    print(f"\n  Benchmarking ({n_runs} runs × {len(images)} images) ...")
    for img in images:
        for _ in range(n_runs):
            n_total += 1
            wall0 = time.perf_counter()

            crop, yolo_ms, detected = yolo.detect(img)
            t_yolo.append(yolo_ms)

            if detected and crop is not None:
                n_detected += 1
                _, _, _, r50_ms, mfn_ms, clf_ms = meta.infer_timed(crop)
                t_r50.append(r50_ms)
                t_mfn.append(mfn_ms)
                t_meta_clf.append(clf_ms)
                classify_ms = r50_ms + mfn_ms + clf_ms
                t_classify.append(classify_ms)

            wall_ms = (time.perf_counter() - wall0) * 1e3
            t_total.append(wall_ms)

    det_rate = n_detected / max(n_total, 1)
    print(f"  Detection rate: {det_rate*100:.1f}%  ({n_detected}/{n_total})")

    return dict(
        yolo        = _stats(t_yolo),
        r50         = _stats(t_r50)      if t_r50      else None,
        mfn         = _stats(t_mfn)      if t_mfn      else None,
        meta_clf    = _stats(t_meta_clf) if t_meta_clf else None,
        classify    = _stats(t_classify) if t_classify else None,
        total       = _stats(t_total),
        fps_total   = round(1000.0 / float(np.median(t_total)), 1),
        fps_det_only = round(1000.0 / float(np.median(t_yolo)), 1),
        det_rate    = round(det_rate, 4),
        n_detected  = n_detected,
        n_total     = n_total,
        total_raw   = t_total,
        yolo_raw    = t_yolo,
        classify_raw = t_classify if t_classify else [],
    )


def make_plots(res: Dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    plt.rcParams.update({
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "font.size":         11,
        "axes.grid":         True,
        "grid.alpha":        0.3,
    })

    C_YOLO  = "#E07B54"
    C_R50   = "#4C72B0"
    C_MFN   = "#DD8452"
    C_META  = "#55A868"
    C_TOTAL = "#8172B3"

    def _save(fig, name):
        p = out_dir / name
        fig.tight_layout()
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        print(f"  plot → {p.relative_to(_ROOT)}")

    # stacked latency breakdown
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    components = []
    vals_med   = []
    colors     = []

    components.append("YOLO\n(detect)");   vals_med.append(res["yolo"]["median"]);     colors.append(C_YOLO)
    if res["r50"]:
        components.append("ResNet-50\n(classify)"); vals_med.append(res["r50"]["median"]);  colors.append(C_R50)
        components.append("MedFusion\n(classify)"); vals_med.append(res["mfn"]["median"]);  colors.append(C_MFN)
        components.append("Meta\n(sklearn)");       vals_med.append(res["meta_clf"]["median"]); colors.append(C_META)

    bottoms = 0.0
    for comp, val, col in zip(components, vals_med, colors):
        bar = ax.bar([0], [val], 0.45, bottom=bottoms, color=col, label=comp, zorder=3)
        if val > 1.0:
            ax.text(0, bottoms + val / 2, f"{val:.1f}ms",
                    ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        bottoms += val

    total_med = res["total"]["median"]
    fps       = res["fps_total"]
    ax.text(0, bottoms + 1, f"{total_med:.1f}ms\n({fps:.0f} FPS)",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xlim(-0.5, 0.5); ax.set_xticks([])
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Full Pipeline\nLatency Breakdown")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, bottoms * 1.3)

    ax2 = axes[1]
    comp_keys = [("YOLO", res["yolo"], C_YOLO)]
    if res["r50"]:
        comp_keys += [
            ("ResNet-50", res["r50"], C_R50),
            ("MedFusion", res["mfn"], C_MFN),
            ("Meta-LR",   res["meta_clf"], C_META),
        ]
    xi = np.arange(len(comp_keys))
    meds   = [v["median"] for _, v, _ in comp_keys]
    p95s   = [v["p95"]    for _, v, _ in comp_keys]
    spikes = [p - m for p, m in zip(p95s, meds)]
    clrs   = [c for _, _, c in comp_keys]
    ax2.bar(xi, meds,   0.5, color=clrs, zorder=3, label="Median")
    ax2.bar(xi, spikes, 0.5, bottom=meds, color=clrs, alpha=0.35, zorder=3, label="P95 tail")
    for i, (p95, med) in enumerate(zip(p95s, meds)):
        ax2.text(i, p95 + 0.3, f"p95\n{p95:.1f}", ha="center", va="bottom", fontsize=7.5, color="gray")
    ax2.set_xticks(xi)
    ax2.set_xticklabels([k for k, _, _ in comp_keys])
    ax2.set_ylabel("Latency (ms)")
    ax2.set_title("Per-Component Median + P95 Tail")
    ax2.legend(fontsize=9)

    fig.suptitle(f"Full Pipeline Latency — TRT FP16  |  {fps:.0f} FPS end-to-end",
                 fontsize=13, y=1.01)
    _save(fig, "pipeline_benchmark_01_latency_breakdown.png")

    # FPS summary
    fig, ax = plt.subplots(figsize=(8, 5))
    labels_fps = ["Full pipeline\n(det + classify)", "YOLO only\n(detection)"]
    fps_vals   = [res["fps_total"], res["fps_det_only"]]
    bar_cols   = [C_TOTAL, C_YOLO]
    bars = ax.bar([0, 1], fps_vals, 0.5, color=bar_cols, zorder=3)
    for bar, val in zip(bars, fps_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels_fps)
    ax.set_ylabel("Frames per second")
    ax.set_title("End-to-End Throughput — TRT FP16")
    _save(fig, "pipeline_benchmark_02_fps.png")

    # total latency histograms
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    tot_med = res["total"]["median"]; tot_p95 = res["total"]["p95"]
    ax.hist(res["total_raw"], bins=40, color=C_TOTAL, edgecolor="white", linewidth=0.4, zorder=3)
    ax.axvline(tot_med, color="black",   lw=1.5, linestyle="--",
               label="Median {:.1f}ms".format(tot_med))
    ax.axvline(tot_p95, color="#C44E52", lw=1.2, linestyle=":",
               label="P95 {:.1f}ms".format(tot_p95))
    ax.set_xlabel("Total latency (ms)"); ax.set_ylabel("Count")
    ax.set_title("Total Pipeline Latency Distribution")
    ax.legend(fontsize=9)

    ax2 = axes[1]
    ax2.hist(res["yolo_raw"], bins=40, color=C_YOLO, edgecolor="white", linewidth=0.4, zorder=3)
    yolo_med = res["yolo"]["median"]; yolo_p95 = res["yolo"]["p95"]
    ax2.axvline(yolo_med, color="black",   lw=1.5, linestyle="--",
                label="Median {:.1f}ms".format(yolo_med))
    ax2.axvline(yolo_p95, color="#C44E52", lw=1.2, linestyle=":",
                label="P95 {:.1f}ms".format(yolo_p95))
    ax2.set_xlabel("YOLO detection latency (ms)"); ax2.set_ylabel("Count")
    ax2.set_title("YOLO Detection Latency Distribution")
    ax2.legend(fontsize=9)

    fig.suptitle("Latency Distributions — TRT FP16", fontsize=13, y=1.01)
    _save(fig, "pipeline_benchmark_03_histograms.png")

    # detection rate + classify latency (when detected)
    if res["classify_raw"]:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        ax = axes[0]
        det_pct  = res["det_rate"] * 100
        miss_pct = 100 - det_pct
        wedges, texts, autotexts = ax.pie(
            [det_pct, miss_pct],
            labels=["Lesion detected\n(classified)", "No detection\n(skipped)"],
            colors=[C_R50, "#CCCCCC"],
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        )
        for t in autotexts: t.set_fontsize(11)
        ax.set_title("Detection Rate\n({}/{} frames)".format(res["n_detected"], res["n_total"]))

        ax2 = axes[1]
        ax2.hist(res["classify_raw"], bins=40, color=C_R50, edgecolor="white",
                 linewidth=0.4, zorder=3)
        clf_med = res["classify"]["median"]; clf_p95 = res["classify"]["p95"]
        ax2.axvline(clf_med, color="black",   lw=1.5, linestyle="--",
                    label="Median {:.1f}ms".format(clf_med))
        ax2.axvline(clf_p95, color="#C44E52", lw=1.2, linestyle=":",
                    label="P95 {:.1f}ms".format(clf_p95))
        ax2.set_xlabel("Classify latency (ms)"); ax2.set_ylabel("Count")
        ax2.set_title("Classification Latency Distribution\n(detected frames only)")
        ax2.legend(fontsize=9)

        fig.suptitle("Detection Rate & Classification Latency — TRT FP16",
                     fontsize=13, y=1.01)
        _save(fig, "pipeline_benchmark_04_detection_classify.png")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the full YOLO FP16 + TRT FP16 meta-learner pipeline.")
    parser.add_argument("--images",  default="scripts/test_images",
                        help="Directory of test images (jpg/png)")
    parser.add_argument("--runs",    type=int, default=100,
                        help="Benchmark runs per image (default 100)")
    parser.add_argument("--warmup",  type=int, default=20,
                        help="Warmup passes before timing (default 20)")
    parser.add_argument("--conf",    type=float, default=0.35,
                        help="YOLO confidence threshold (default 0.35)")
    args = parser.parse_args()

    print()
    print(_SEP)
    print(_c("  Full Pipeline Benchmark — YOLO FP16 + TRT FP16 Meta-Learner", _B, _W))
    print(_SEP2)
    print(f"  YOLO engine : {_c(str(_DET_ENGINE.relative_to(_ROOT)), _D)}")
    print(f"  TRT engines : {_c(str(_TRT_DIR.relative_to(_ROOT)), _D)}")
    print(f"  Runs        : {_c(str(args.runs), _C)} per image  |  "
          f"Warmup: {_c(str(args.warmup), _C)}")
    print(_SEP)

    img_dir = _ROOT / args.images
    images  = [img for img in (cv2.imread(str(p))
               for p in sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")))
               if img is not None]
    if not images:
        print(f"  {_c('[WARN] No images in', _Y)} {img_dir} — using synthetic 640×480")
        images = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)]
    print(f"\n  {_c('✓', _G)} {len(images)} test image(s) loaded from {img_dir.relative_to(_ROOT)}")

    print(f"\n  Loading TRT engines ...")
    try:
        yolo = _TRTYOLODetector(_DET_ENGINE, conf_threshold=args.conf)
    except Exception as e:
        print(f"  {_c('✗ YOLO load failed:', _RE)} {e}")
        return

    try:
        meta = _MetaPipeline(_DEPLOY_DIR / "meta_learner.pkl", _TRT_DIR, "fp16")
    except Exception as e:
        print(f"  {_c('✗ Meta-pipeline load failed:', _RE)} {e}")
        yolo.close()
        return

    try:
        res = run_benchmark(yolo, meta, images, args.warmup, args.runs)
    finally:
        yolo.close()
        meta.close()

    print()
    print(_SEP)
    print(_c("  Results", _B, _W))
    print(_SEP2)

    def _row(label, s, fps=None):
        if s is None:
            return
        fps_str = ("  ->  " + "{:.1f} FPS".format(fps)) if fps else ""
        med = s["median"]; p95 = s["p95"]; p99 = s["p99"]
        med_str = "{:6.2f}ms".format(med)
        tail    = "p95={:6.2f}ms  p99={:6.2f}ms{}".format(p95, p99, fps_str)
        print("  {:<22} median={}  {}".format(label, _c(med_str, _G), tail))

    _row("YOLO detect",      res["yolo"],     res["fps_det_only"])
    _row("ResNet-50",         res["r50"])
    _row("MedFusionNet",      res["mfn"])
    _row("Meta-LR (sklearn)", res["meta_clf"])
    print(_SEP2)
    _row("Total pipeline",    res["total"],    res["fps_total"])
    print(_SEP2)
    det_pct  = res["det_rate"] * 100
    n_det    = res["n_detected"]
    n_tot    = res["n_total"]
    det_str  = "{:.1f}%".format(det_pct)
    print("  Detection rate   : {}  ({}/{} frames)".format(_c(det_str, _C), n_det, n_tot))
    print()

    import pandas as pd
    xlsx_path = _OUT_DIR / "benchmark_results.xlsx"

    summary_rows = []
    for label, key in [
        ("YOLO FP16 (detect)",       "yolo"),
        ("ResNet-50 FP16 (classify)", "r50"),
        ("MedFusionNet FP16 (classify)", "mfn"),
        ("Meta-LR sklearn (classify)", "meta_clf"),
        ("Classification total",       "classify"),
        ("Full pipeline total",        "total"),
    ]:
        s = res.get(key)
        if s is None:
            continue
        summary_rows.append({
            "component":    label,
            "median_ms":    s["median"],
            "mean_ms":      s["mean"],
            "std_ms":       s["std"],
            "p95_ms":       s["p95"],
            "p99_ms":       s["p99"],
            "min_ms":       s["min"],
            "max_ms":       s["max"],
        })

    fps_rows = [
        {"metric": "FPS (full pipeline)",  "value": res["fps_total"]},
        {"metric": "FPS (YOLO only)",      "value": res["fps_det_only"]},
        {"metric": "Detection rate (%)",   "value": round(res["det_rate"] * 100, 2)},
        {"metric": "Frames detected",      "value": res["n_detected"]},
        {"metric": "Frames total",         "value": res["n_total"]},
    ]

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Latency", index=False)
        pd.DataFrame(fps_rows).to_excel(writer, sheet_name="FPS", index=False)
        pd.DataFrame({"total_ms": res["total_raw"]}).to_excel(
            writer, sheet_name="RawTotal", index=False)
        pd.DataFrame({"yolo_ms": res["yolo_raw"]}).to_excel(
            writer, sheet_name="RawYOLO", index=False)
        if res["classify_raw"]:
            pd.DataFrame({"classify_ms": res["classify_raw"]}).to_excel(
                writer, sheet_name="RawClassify", index=False)

    print("  Excel -> {}".format(xlsx_path.relative_to(_ROOT)))

    print(_c("  Generating plots ...", _D))
    make_plots(res, _OUT_DIR)

    print()
    print(_SEP)
    print("  {}  Output -> {}".format(
        _c("Done.", _G, _B), _c(str(_OUT_DIR.relative_to(_ROOT)), _D)))
    print(_SEP)
    print()


if __name__ == "__main__":
    main()
