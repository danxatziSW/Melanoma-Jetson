"""Three-format benchmark for no-seg melanoma classification on Jetson Orin Nano.

Compares PyTorch (.pt) vs TFLite (.tflite) vs TensorRT FP16 (.engine)
for every classification model — the full table needed for a real-time edge paper.

Pipeline for all three formats:
    raw image  →  YOLO crop (TRT)  →  classifier (PT / TFLite / TRT)
NO segmentation step.

Measures per model / format:
  LATENCY  : median, mean, std, p95, p99  (ms)
  THROUGHPUT: FPS  (crop + classify)
  HARDWARE : GPU utilisation + GPU RAM  (via tegrastats / jtop)
  FILE SIZE : MB on disk
  PARAMS    : trainable parameters (M)  — PT only
  ACCURACY  : sensitivity, specificity, F1, F2, AUC
              per-dataset calibrated threshold

Output:
  outputs/ablation_noseg/benchmark_all_formats.xlsx
    sheets: PT_Results, TFLite_Results, TRT_Results, Comparison,
            All_By_Latency, All_By_AUC_HAM

Run ON the Jetson Orin Nano:
  # 1. Convert everything first
  python3 scripts/no_seg/convert_to_tensorrt.py --noseg --yolo-crop --skip-existing
  python3 scripts/no_seg/convert_to_tflite.py   --noseg --skip-existing

  # 2. Benchmark all three formats
  python3 scripts/no_seg/benchmark_tensorrt.py --images /path/to/test_images

Flags:
  --suffix none          evaluate original no-seg models (default)
  --suffix none_sens     evaluate sensitivity fine-tuned models
  --latency-only         skip accuracy evaluation
  --accuracy-only        skip latency benchmark (no test images needed)
  --no-plots             skip plot generation
  --runs  100            timed runs per image (default: 100)
  --warmup 20            warmup runs (default: 20)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from sklearn.metrics import roc_auc_score

from src.models.registry import build_model, uses_metadata
from src.utils.config import load_config
from src.utils.io import write_excel_sheet

# ── constants ─────────────────────────────────────────────────────────────────
ALL_DATASETS = ["ham10000", "isic2019", "isic2020"]
ALL_MODELS   = ["resnet50", "efficientnet_b2", "mobilenetv3_large",
                "convnext_tiny_se", "medfusionnet", "yolov8_cls"]

_MEAN       = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD        = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_THRESHOLDS = np.round(np.arange(0.20, 0.86, 0.01), 2)
_SITE_CATS  = ["head/neck", "upper extremity", "lower extremity",
               "torso", "palms/soles", "oral/genital"]
_NEUTRAL_META = np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ── image preprocessing (shared) ──────────────────────────────────────────────

def _preprocess_bgr(image_bgr: np.ndarray, size: int) -> np.ndarray:
    """BGR → normalised float32 CHW numpy array."""
    big = int(size * 1.1)
    img = cv2.resize(image_bgr, (big, big))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    st  = (big - size) // 2
    img = img[st:st + size, st:st + size]
    return ((img - _MEAN) / _STD).transpose(2, 0, 1).astype(np.float32)   # CHW


def _build_meta(row) -> np.ndarray:
    """Build 8-dim metadata vector from a DataFrame row."""
    age = float(row.get("age_approx", 50) or 50) / 100.0
    sex_raw = str(row.get("sex", "")).lower()
    sex = 1.0 if sex_raw == "male" else (0.0 if sex_raw == "female" else 0.5)
    site_raw = str(row.get("anatom_site_general_challenge", ""))
    site = np.zeros(len(_SITE_CATS), dtype=np.float32)
    for i, cat in enumerate(_SITE_CATS):
        if site_raw == cat:
            site[i] = 1.0
    return np.concatenate([[age, sex], site]).astype(np.float32)


# ── PyTorch classifier ────────────────────────────────────────────────────────

class ClassifierPT:
    """Wraps a PyTorch .pt checkpoint for classification inference."""

    def __init__(self, ckpt_path: Path, model_name: str, config,
                 with_meta: bool = False, input_size: int = 224):
        self.with_meta  = with_meta
        self.input_size = input_size
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = build_model(model_name, config, num_classes=2)
        self.model.load_state_dict(
            torch.load(ckpt_path, map_location=self.device, weights_only=True))
        self.model.to(self.device).eval()

        self.param_m = sum(p.numel() for p in self.model.parameters()
                           if p.requires_grad) / 1e6

    @torch.no_grad()
    def predict_prob(self, image_bgr: np.ndarray,
                     meta_np: Optional[np.ndarray] = None) -> float:
        chw   = _preprocess_bgr(image_bgr, self.input_size)
        img_t = torch.from_numpy(chw).unsqueeze(0).to(self.device)
        if self.with_meta:
            m      = meta_np if meta_np is not None else _NEUTRAL_META
            meta_t = torch.from_numpy(m).unsqueeze(0).to(self.device)
            logits = self.model(img_t, meta_t)
        else:
            logits = self.model(img_t)
        return float(torch.softmax(logits, dim=1)[0, 1].item())

    def close(self):
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ── TensorRT classifier ───────────────────────────────────────────────────────
# Uses ctypes + libcudart directly — no pycuda or torch.cuda required.
# This works on Jetson where PyTorch is compiled for a newer CUDA than the driver.

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
    raise RuntimeError(
        "Cannot find libcudart.so.12 — JetPack should provide it at "
        "/usr/local/cuda-12.6/lib64/"
    )

_CUDART: Any = None

def _cudart():
    global _CUDART
    if _CUDART is None:
        _CUDART = _load_cudart()
    return _CUDART


class ClassifierTRT:
    """Wraps a TRT engine for classification. Compatible with TRT 8.x and 10.x.
    Uses ctypes/libcudart for CUDA memory — no pycuda or torch.cuda needed."""

    def __init__(self, engine_path: Path, with_meta: bool = False,
                 input_size: int = 224):
        import tensorrt as trt
        import ctypes
        self._ct        = ctypes
        self.with_meta  = with_meta
        self.input_size = input_size

        _cudart()  # ensure libcudart is loaded and CUDA context exists

        logger = trt.Logger(trt.Logger.WARNING)
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
            n          = self.engine.num_bindings
            img_idx    = self.engine.get_binding_index("image")
            out_idx    = next(i for i in range(n) if not self.engine.binding_is_input(i))
            img_shape  = tuple(self.engine.get_binding_shape(img_idx))
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

    def _malloc(self, n: int) -> int:
        ptr = self._ct.c_void_p()
        err = _cudart().cudaMalloc(self._ct.byref(ptr), self._ct.c_size_t(n))
        if err:
            raise RuntimeError(f"cudaMalloc failed (err={err})")
        return ptr.value

    def _h2d(self, d: int, h: np.ndarray) -> None:
        err = _cudart().cudaMemcpy(
            self._ct.c_void_p(d), h.ctypes.data_as(self._ct.c_void_p),
            self._ct.c_size_t(h.nbytes), self._ct.c_int(1))
        if err:
            raise RuntimeError(f"cudaMemcpy H2D failed (err={err})")

    def _d2h(self, h: np.ndarray, d: int) -> None:
        err = _cudart().cudaMemcpy(
            h.ctypes.data_as(self._ct.c_void_p), self._ct.c_void_p(d),
            self._ct.c_size_t(h.nbytes), self._ct.c_int(2))
        if err:
            raise RuntimeError(f"cudaMemcpy D2H failed (err={err})")

    def _sync(self) -> None:
        _cudart().cudaDeviceSynchronize()

    def _infer(self) -> None:
        if self._v10:
            self.context.set_tensor_address(self._in_names[0], self._d_img)
            if self.with_meta and self._d_meta is not None:
                self.context.set_tensor_address(self._in_names[1], self._d_meta)
            self.context.set_tensor_address(self._out_names[0], self._d_out)
            self.context.execute_async_v3(stream_handle=0)
        else:
            n        = self.engine.num_bindings
            bindings = [0] * n
            bindings[self.engine.get_binding_index("image")] = self._d_img
            if self.with_meta and self._d_meta is not None:
                bindings[self.engine.get_binding_index("metadata")] = self._d_meta
            out_idx  = next(i for i in range(n) if not self.engine.binding_is_input(i))
            bindings[out_idx] = self._d_out
            self.context.execute_async_v2(bindings=bindings, stream_handle=0)
        self._sync()

    def predict_prob(self, image_bgr: np.ndarray,
                     meta_np: Optional[np.ndarray] = None) -> float:
        np.copyto(self._h_img,
                  _preprocess_bgr(image_bgr, self.input_size)[np.newaxis])
        self._h2d(self._d_img, self._h_img)
        if self.with_meta and self._h_meta is not None:
            m = meta_np if meta_np is not None else _NEUTRAL_META
            np.copyto(self._h_meta, m[np.newaxis])
            self._h2d(self._d_meta, self._h_meta)
        self._infer()
        self._d2h(self._h_out, self._d_out)
        logits = self._h_out[0]
        exp_l  = np.exp(logits - logits.max())
        return float((exp_l / exp_l.sum())[1])

    def close(self) -> None:
        for ptr in [self._d_img, self._d_out, self._d_meta]:
            if ptr is not None:
                _cudart().cudaFree(self._ct.c_void_p(ptr))
        del self.context, self.engine


# ── TFLite classifier ─────────────────────────────────────────────────────────

class ClassifierTFLite:
    """Wraps a TFLite interpreter for classification."""

    def __init__(self, tflite_path: Path, with_meta: bool = False,
                 input_size: int = 224):
        self.with_meta  = with_meta
        self.input_size = input_size

        try:
            import tflite_runtime.interpreter as _tfl
        except ImportError:
            try:
                import tensorflow as tf
                _tfl = tf.lite
            except ImportError:
                raise ImportError("Install tflite-runtime:  pip install tflite-runtime")

        self.interp = _tfl.Interpreter(model_path=str(tflite_path))
        self.interp.allocate_tensors()
        inp_det = self.interp.get_input_details()
        out_det = self.interp.get_output_details()

        self._out_idx = out_det[0]["index"]
        # image input: shape (1,H,W,3) NHWC or (1,3,H,W) NCHW
        img_info      = next(d for d in inp_det if len(d["shape"]) == 4)
        self._img_idx = img_info["index"]
        self._nhwc    = (img_info["shape"][-1] == 3)
        self._size    = int(img_info["shape"][1] if self._nhwc else img_info["shape"][2])
        # metadata input: shape (1, 8)
        self._meta_idx = next(
            (d["index"] for d in inp_det if len(d["shape"]) == 2), None)
        self._inp_det  = inp_det

    def predict_prob(self, image_bgr: np.ndarray,
                     meta_np: Optional[np.ndarray] = None) -> float:
        chw = _preprocess_bgr(image_bgr, self._size)
        img_b = chw[np.newaxis] if not self._nhwc else chw.transpose(1, 2, 0)[np.newaxis]
        self.interp.set_tensor(self._img_idx, img_b)
        if self.with_meta and self._meta_idx is not None:
            m = meta_np if meta_np is not None else _NEUTRAL_META
            self.interp.set_tensor(self._meta_idx, m.astype(np.float32)[np.newaxis])
        self.interp.invoke()
        logits = self.interp.get_tensor(self._out_idx)[0]
        exp_l  = np.exp(logits - logits.max())
        return float((exp_l / exp_l.sum())[1])

    def close(self):
        del self.interp


# ── YOLO crop ─────────────────────────────────────────────────────────────────

class YOLOCrop:
    """CPU fallback: runs ultralytics YOLO .pt on CPU."""
    def __init__(self, engine_path: Path, conf: float = 0.35):
        from ultralytics import YOLO
        self._model = YOLO(str(engine_path))
        self._conf  = conf

    def crop(self, image_bgr: np.ndarray) -> np.ndarray:
        h, w    = image_bgr.shape[:2]
        results = self._model(image_bgr, conf=self._conf, verbose=False)
        boxes   = results[0].boxes
        if boxes is None or len(boxes) == 0:
            m = int(min(h, w) * 0.05)
            return image_bgr[m:h - m, m:w - m]
        x1, y1, x2, y2 = map(int, boxes.xyxy[0].tolist())
        pad = max(4, int(0.05 * min(x2 - x1, y2 - y1)))
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
        return image_bgr[y1:y2, x1:x2]


class YOLOCropTRT:
    """YOLO lesion crop via TensorRT — no torch.cuda needed, uses ctypes + libcudart.

    Input:  [1, 3, 640, 640] float32 (letterboxed, RGB, /255)
    Output: [1, 5, 8400]     float32 (cx, cy, w, h, score) — YOLOv8 decoded format
    """
    INPUT = 640

    def __init__(self, engine_path: Path, conf: float = 0.35):
        import tensorrt as trt
        import ctypes
        self._ct   = ctypes
        self._conf = conf
        _cudart()  # ensure CUDA context

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self._h_img = np.zeros((1, 3, self.INPUT, self.INPUT), dtype=np.float32)
        self._h_out = np.zeros((1, 5, 8400), dtype=np.float32)
        self._d_img = self._malloc(self._h_img.nbytes)
        self._d_out = self._malloc(self._h_out.nbytes)

    def _malloc(self, n: int) -> int:
        ptr = self._ct.c_void_p()
        if _cudart().cudaMalloc(self._ct.byref(ptr), self._ct.c_size_t(n)):
            raise RuntimeError("cudaMalloc failed")
        return ptr.value

    def _h2d(self, d: int, h: np.ndarray) -> None:
        _cudart().cudaMemcpy(self._ct.c_void_p(d),
                             h.ctypes.data_as(self._ct.c_void_p),
                             self._ct.c_size_t(h.nbytes), self._ct.c_int(1))

    def _d2h(self, h: np.ndarray, d: int) -> None:
        _cudart().cudaMemcpy(h.ctypes.data_as(self._ct.c_void_p),
                             self._ct.c_void_p(d),
                             self._ct.c_size_t(h.nbytes), self._ct.c_int(2))

    def _letterbox(self, img: np.ndarray):
        h0, w0 = img.shape[:2]
        s   = self.INPUT / max(h0, w0)
        nh  = int(h0 * s + 0.5); nw = int(w0 * s + 0.5)
        rsz = cv2.resize(img, (nw, nh))
        ph  = (self.INPUT - nh) // 2; pw = (self.INPUT - nw) // 2
        canvas = np.full((self.INPUT, self.INPUT, 3), 114, dtype=np.uint8)
        canvas[ph:ph + nh, pw:pw + nw] = rsz
        return canvas, s, pw, ph

    def crop(self, image_bgr: np.ndarray) -> np.ndarray:
        h0, w0 = image_bgr.shape[:2]
        lb, s, pw, ph = self._letterbox(image_bgr)
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        np.copyto(self._h_img, rgb.transpose(2, 0, 1)[np.newaxis])

        self._h2d(self._d_img, self._h_img)
        self.context.set_tensor_address("images",  self._d_img)
        self.context.set_tensor_address("output0", self._d_out)
        self.context.execute_async_v3(stream_handle=0)
        _cudart().cudaDeviceSynchronize()
        self._d2h(self._h_out, self._d_out)

        preds  = self._h_out[0].T          # [8400, 5]
        scores = preds[:, 4]
        mask   = scores >= self._conf
        if not mask.any():
            m = int(min(h0, w0) * 0.05)
            return image_bgr[m:h0 - m, m:w0 - m]

        p = preds[mask]
        cx, cy, bw, bh, sc = p[:, 0], p[:, 1], p[:, 2], p[:, 3], p[:, 4]
        x1 = (cx - bw / 2 - pw) / s;  y1 = (cy - bh / 2 - ph) / s
        x2 = (cx + bw / 2 - pw) / s;  y2 = (cy + bh / 2 - ph) / s

        # greedy NMS
        order = sc.argsort()[::-1]; keep = []
        while order.size:
            i = order[0]; keep.append(i)
            ix1 = np.maximum(x1[i], x1[order[1:]])
            iy1 = np.maximum(y1[i], y1[order[1:]])
            ix2 = np.minimum(x2[i], x2[order[1:]])
            iy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
            a_i   = (x2[i] - x1[i]) * (y2[i] - y1[i])
            a_j   = (x2[order[1:]] - x1[order[1:]]) * (y2[order[1:]] - y1[order[1:]])
            iou   = inter / (a_i + a_j - inter + 1e-6)
            order = order[1:][iou < 0.45]

        b   = keep[0]
        bx1 = max(0,  int(x1[b])); by1 = max(0,  int(y1[b]))
        bx2 = min(w0, int(x2[b])); by2 = min(h0, int(y2[b]))
        pad = max(4, int(0.05 * min(bx2 - bx1, by2 - by1)))
        return image_bgr[max(0, by1 - pad):min(h0, by2 + pad),
                         max(0, bx1 - pad):min(w0, bx2 + pad)]

    def close(self) -> None:
        for ptr in [self._d_img, self._d_out]:
            if ptr is not None:
                _cudart().cudaFree(self._ct.c_void_p(ptr))
        del self.context, self.engine


# ── hardware sampler ──────────────────────────────────────────────────────────

def _build_gpu_fn() -> Optional[Callable]:
    try:
        from jtop import jtop as JTop
        j = JTop(); j.start()
        def _jtop():
            s = j.stats
            pct = s.get("GPU1", s.get("GPU", None))
            return float(pct) if pct is not None else None, None
        return _jtop
    except Exception:
        pass
    try:
        proc = subprocess.Popen(
            ["tegrastats", "--interval", "200"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        first = proc.stdout.readline()
        if "GR3D_FREQ" in first:
            import re as _re
            def _tegra():
                line = proc.stdout.readline()
                m  = _re.search(r"GR3D_FREQ\s+(\d+)%", line)
                m2 = _re.search(r"RAM\s+(\d+)/\d+MB", line)
                return (float(m.group(1)) if m else None,
                        float(m2.group(1)) if m2 else None)
            return _tegra
        proc.terminate()
    except FileNotFoundError:
        pass
    return None


class HWSampler:
    def __init__(self):
        self._samples: List = []
        self._running = False
        self._lock    = threading.Lock()
        self._gpu_fn  = _build_gpu_fn()

    def start(self):
        self._samples = []; self._running = True
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self._running = False; time.sleep(0.2)

    def _run(self):
        while self._running:
            gpu_pct, gpu_ram = None, None
            if self._gpu_fn:
                try:
                    gpu_pct, gpu_ram = self._gpu_fn()
                except Exception:
                    pass
            with self._lock:
                self._samples.append((gpu_pct, gpu_ram))
            time.sleep(0.1)

    def peak(self) -> Tuple[Optional[float], Optional[float]]:
        with self._lock:
            s = list(self._samples)
        gpus = [x[0] for x in s if x[0] is not None]
        rams = [x[1] for x in s if x[1] is not None]
        return (max(gpus) if gpus else None, max(rams) if rams else None)


# ── latency benchmark ─────────────────────────────────────────────────────────

def _bench_latency(
    images:     List[np.ndarray],
    crop_model: Optional[YOLOCrop],
    classifier,               # ClassifierTRT or ClassifierTFLite
    n_warmup:   int,
    n_runs:     int,
) -> Dict[str, Any]:
    for img in images:
        for _ in range(n_warmup):
            crop = crop_model.crop(img) if crop_model else img
            classifier.predict_prob(crop)

    crop_ms, cls_ms, total_ms = [], [], []
    sampler = HWSampler()
    sampler.start()

    for img in images:
        for _ in range(n_runs):
            t0 = time.perf_counter()
            crop = crop_model.crop(img) if crop_model else img
            t1   = time.perf_counter()
            classifier.predict_prob(crop)
            t2   = time.perf_counter()
            crop_ms.append((t1 - t0) * 1e3)
            cls_ms.append((t2 - t1) * 1e3)
            total_ms.append((t2 - t0) * 1e3)

    sampler.stop()
    gpu_pct_peak, gpu_ram_peak = sampler.peak()

    def _s(vals):
        a = np.asarray(vals)
        return dict(median=round(float(np.median(a)), 2),
                    mean=round(float(np.mean(a)), 2),
                    std=round(float(np.std(a)), 2),
                    p95=round(float(np.percentile(a, 95)), 2),
                    p99=round(float(np.percentile(a, 99)), 2))

    return dict(
        crop=_s(crop_ms) if crop_model else None,
        classify=_s(cls_ms),
        total=_s(total_ms),
        fps=round(1000.0 / float(np.median(total_ms)), 2),
        fps_classify=round(1000.0 / float(np.median(cls_ms)), 2),
        gpu_pct_peak=gpu_pct_peak,
        gpu_ram_peak_mb=gpu_ram_peak,
    )


# ── accuracy evaluation ────────────────────────────────────────────────────────

def _eval_accuracy(
    classifier,          # ClassifierTRT or ClassifierTFLite — needs predict_prob()
    model_name: str,
    test_dfs:   Dict,
) -> Dict[str, Any]:
    with_meta = uses_metadata(model_name)
    results   = {}

    for ds_name, df in test_dfs.items():
        all_probs, all_labels = [], []

        for _, row in df.iterrows():
            image = cv2.imread(str(row["image_path"]))
            if image is None:
                image = np.zeros((224, 224, 3), dtype=np.uint8)
            meta = _build_meta(row) if with_meta else None
            all_probs.append(classifier.predict_prob(image, meta))
            all_labels.append(int(row["binary_label"]))

        probs  = np.array(all_probs)
        labels = np.array(all_labels)

        try:
            auc = float(roc_auc_score(labels, probs))
        except Exception:
            auc = float("nan")

        # per-dataset F2-optimal threshold
        best_thr, best_f2 = _THRESHOLDS[0], -1.0
        for thr in _THRESHOLDS:
            preds = (probs >= thr).astype(int)
            tp = int(((preds == 1) & (labels == 1)).sum())
            fp = int(((preds == 1) & (labels == 0)).sum())
            fn = int(((preds == 0) & (labels == 1)).sum())
            s  = tp / max(tp + fn, 1)
            p  = tp / max(tp + fp, 1)
            f2 = (5 * p * s) / max(4 * p + s, 1e-9)
            if f2 > best_f2:
                best_f2, best_thr = f2, thr

        preds = (probs >= best_thr).astype(int)
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

        results[ds_name] = dict(
            threshold=round(best_thr, 2),
            auc=round(auc, 4),
            sensitivity=round(sens, 4),
            specificity=round(spec, 4),
            precision=round(prec, 4),
            f1=round(f1, 4), f2=round(f2, 4),
            accuracy=round(acc, 4),
            tp=tp, tn=tn, fp=fp, fn=fn,
            n_mel=int(labels.sum()), n_total=int(len(labels)),
        )
    return results


# ── scanners ─────────────────────────────────────────────────────────────────

def _scan_pt(ablation_dir: Path, suffix: str) -> List[Dict]:
    entries = []
    for ds in ALL_DATASETS:
        for mn in ALL_MODELS:
            run_id = f"{mn}_{suffix}"
            path   = ablation_dir / ds / run_id / "checkpoints" / f"{run_id}.pt"
            if path.exists():
                entries.append(dict(model=mn, train_dataset=ds,
                                    run_id=run_id, path=path, fmt="pt",
                                    file_mb=round(path.stat().st_size / 1e6, 2)))
    return entries


def _scan(ablation_dir: Path, suffix: str, fmt: str,
          precision: str = "fp16") -> List[Dict]:
    if fmt == "trt":
        ext = f"_{precision}.engine"
        sub = "tensorrt"
    else:
        ext = ".tflite"
        sub = "tflite"
    entries = []
    for ds in ALL_DATASETS:
        for mn in ALL_MODELS:
            run_id = f"{mn}_{suffix}"
            path   = ablation_dir / ds / run_id / sub / f"{run_id}{ext}"
            if path.exists():
                entries.append(dict(model=mn, train_dataset=ds,
                                    run_id=run_id, path=path, fmt=fmt,
                                    file_mb=round(path.stat().st_size / 1e6, 2)))
    return entries


# ── plotting ──────────────────────────────────────────────────────────────────

def _plot(pt_rows: List[Dict], tfl_rows: List[Dict],
          trt_rows: List[Dict], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plots] matplotlib not available — skipping"); return

    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300,
                          "axes.spines.top": False, "axes.spines.right": False})
    out_dir.mkdir(parents=True, exist_ok=True)
    _PT_COL  = "#55A868"
    _TFL_COL = "#DD8452"
    _TRT_COL = "#4C72B0"
    _DS_COLORS = {"ham10000": "#4C72B0", "isic2019": "#DD8452", "isic2020": "#55A868"}

    def _labels(rows):
        return [f"{r['model']}\n{r['train_dataset']}" for r in rows]

    def _save(fig, name):
        fig.tight_layout(); fig.savefig(out_dir / name, bbox_inches="tight")
        plt.close(fig)

    # build maps for 3-way comparison
    pt_map  = {(r["model"], r["train_dataset"]): r for r in pt_rows}
    tfl_map = {(r["model"], r["train_dataset"]): r for r in tfl_rows}
    trt_map = {(r["model"], r["train_dataset"]): r for r in trt_rows}
    all_keys   = sorted(set(pt_map) | set(tfl_map) | set(trt_map))
    common_keys = sorted(set(pt_map) & set(tfl_map) & set(trt_map))
    xs_all = np.arange(len(all_keys))
    xs_cmn = np.arange(len(common_keys))
    lbl_all = [f"{k[0]}\n{k[1]}" for k in all_keys]
    lbl_cmn = [f"{k[0]}\n{k[1]}" for k in common_keys]

    # 1. 3-way classify latency grouped bar
    if common_keys:
        pt_lat  = [pt_map[k].get("classify_ms_median",  0) or 0 for k in common_keys]
        tfl_lat = [tfl_map[k].get("classify_ms_median", 0) or 0 for k in common_keys]
        trt_lat = [trt_map[k].get("classify_ms_median", 0) or 0 for k in common_keys]
        fig, ax = plt.subplots(figsize=(max(10, len(common_keys) * 1.0), 5))
        ax.bar(xs_cmn - 0.25, pt_lat,  0.23, label="PyTorch (.pt)",   color=_PT_COL,  zorder=3)
        ax.bar(xs_cmn,         tfl_lat, 0.23, label="TFLite FP32",     color=_TFL_COL, zorder=3)
        ax.bar(xs_cmn + 0.25, trt_lat, 0.23, label="TensorRT FP16",   color=_TRT_COL, zorder=3)
        # annotate TRT speedup over PT
        for xi, pv, tv in zip(xs_cmn, pt_lat, trt_lat):
            if pv > 0 and tv > 0:
                ax.text(xi + 0.25, tv + 0.4, f"×{pv/tv:.1f}",
                        ha="center", va="bottom", fontsize=7, color=_TRT_COL, fontweight="bold")
        ax.set_xticks(xs_cmn); ax.set_xticklabels(lbl_cmn, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Classifier Latency: PyTorch vs TFLite vs TensorRT FP16\n"
                     "(×N above TRT bar = speedup over PyTorch)")
        ax.legend(); ax.grid(axis="y", alpha=0.4)
        _save(fig, "01_latency_3way.png")

    # 2. 3-way FPS
    if common_keys:
        pt_fps  = [pt_map[k].get("fps",  0) or 0 for k in common_keys]
        tfl_fps = [tfl_map[k].get("fps", 0) or 0 for k in common_keys]
        trt_fps = [trt_map[k].get("fps", 0) or 0 for k in common_keys]
        fig, ax = plt.subplots(figsize=(max(10, len(common_keys) * 1.0), 5))
        ax.bar(xs_cmn - 0.25, pt_fps,  0.23, label="PyTorch (.pt)",  color=_PT_COL,  zorder=3)
        ax.bar(xs_cmn,         tfl_fps, 0.23, label="TFLite FP32",    color=_TFL_COL, zorder=3)
        ax.bar(xs_cmn + 0.25, trt_fps, 0.23, label="TensorRT FP16",  color=_TRT_COL, zorder=3)
        ax.set_xticks(xs_cmn); ax.set_xticklabels(lbl_cmn, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("FPS"); ax.set_title("End-to-End FPS: PyTorch vs TFLite vs TensorRT")
        ax.legend(); ax.grid(axis="y", alpha=0.4)
        _save(fig, "02_fps_3way.png")

    # 3. File size
    if common_keys:
        pt_sz  = [pt_map[k].get("file_mb",  0) or 0 for k in common_keys]
        tfl_sz = [tfl_map[k].get("file_mb", 0) or 0 for k in common_keys]
        trt_sz = [trt_map[k].get("file_mb", 0) or 0 for k in common_keys]
        fig, ax = plt.subplots(figsize=(max(10, len(common_keys) * 1.0), 5))
        ax.bar(xs_cmn - 0.25, pt_sz,  0.23, label="PyTorch (.pt)",  color=_PT_COL,  zorder=3)
        ax.bar(xs_cmn,         tfl_sz, 0.23, label="TFLite FP32",    color=_TFL_COL, zorder=3)
        ax.bar(xs_cmn + 0.25, trt_sz, 0.23, label="TensorRT FP16",  color=_TRT_COL, zorder=3)
        ax.set_xticks(xs_cmn); ax.set_xticklabels(lbl_cmn, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("File size (MB)"); ax.set_title("Model File Size on Disk")
        ax.legend(); ax.grid(axis="y", alpha=0.4)
        _save(fig, "03_file_size_3way.png")

    # 4. Sensitivity per dataset — one chart per format
    n_ds = len(ALL_DATASETS)
    w    = 0.22
    for fmt_label, rows, slug in [("PyTorch", pt_rows, "pt"),
                                   ("TFLite",  tfl_rows, "tfl"),
                                   ("TensorRT FP16", trt_rows, "trt")]:
        if not rows:
            continue
        labels_f = _labels(rows); xs_f = np.arange(len(rows))
        fig, ax  = plt.subplots(figsize=(max(10, len(rows) * 0.9), 5))
        for i, ds in enumerate(ALL_DATASETS):
            vals   = [r.get(f"sensitivity_{ds}", 0) or 0 for r in rows]
            offset = (i - n_ds // 2) * w
            ax.bar(xs_f + offset, vals, w, label=ds, color=_DS_COLORS[ds], zorder=3)
        ax.axhline(0.85, color="red", linestyle="--", lw=1, alpha=0.7, label="Target ≥0.85")
        ax.set_xticks(xs_f); ax.set_xticklabels(labels_f, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05); ax.set_ylabel("Sensitivity")
        ax.set_title(f"Sensitivity per Dataset — {fmt_label}")
        ax.legend(); ax.grid(axis="y", alpha=0.4)
        _save(fig, f"04_sensitivity_{slug}.png")

    # 5. AUC per dataset — one chart per format
    for fmt_label, rows, slug in [("PyTorch", pt_rows, "pt"),
                                   ("TFLite",  tfl_rows, "tfl"),
                                   ("TensorRT FP16", trt_rows, "trt")]:
        if not rows:
            continue
        labels_f = _labels(rows); xs_f = np.arange(len(rows))
        fig, ax  = plt.subplots(figsize=(max(10, len(rows) * 0.9), 5))
        for i, ds in enumerate(ALL_DATASETS):
            vals   = [r.get(f"auc_{ds}", 0) or 0 for r in rows]
            offset = (i - n_ds // 2) * w
            ax.bar(xs_f + offset, vals, w, label=ds, color=_DS_COLORS[ds], zorder=3)
        ax.set_xticks(xs_f); ax.set_xticklabels(labels_f, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0.5, 1.05); ax.set_ylabel("AUC-ROC")
        ax.set_title(f"AUC-ROC per Dataset — {fmt_label}")
        ax.legend(); ax.grid(axis="y", alpha=0.4)
        _save(fig, f"05_auc_{slug}.png")

    # 6. Speed-accuracy scatter — all three formats, HAM10000
    fig, ax = plt.subplots(figsize=(9, 5))
    for rows, color, marker, fmt_label in [
        (pt_rows,  _PT_COL,  "^", "PT"),
        (tfl_rows, _TFL_COL, "s", "TFL"),
        (trt_rows, _TRT_COL, "o", "TRT"),
    ]:
        plotted = False
        for r in rows:
            lat = r.get("classify_ms_median")
            auc = r.get("auc_ham10000")
            if lat and auc:
                ax.scatter(lat, auc, color=color, marker=marker, s=70, zorder=3,
                           label=fmt_label if not plotted else "_")
                ax.annotate(f"{r['model'][:7]}",
                            (lat, auc), textcoords="offset points",
                            xytext=(4, 3), fontsize=6)
                plotted = True
    ax.set_xlabel("Classify latency (ms)"); ax.set_ylabel("AUC (HAM10000)")
    ax.set_title("Speed-Accuracy Trade-off (HAM10000)\n▲ PyTorch  ■ TFLite  ● TensorRT")
    ax.legend(); ax.grid(alpha=0.4)
    _save(fig, "06_speed_accuracy_3way.png")

    pngs = list(out_dir.glob("*.png"))
    print(f"  {len(pngs)} plots saved → {out_dir}/")


# ── row builder ───────────────────────────────────────────────────────────────

def _build_row(entry: Dict, lat: Optional[Dict],
               acc: Optional[Dict],
               param_m: Optional[float] = None) -> Dict[str, Any]:
    row: Dict[str, Any] = dict(
        model=entry["model"],
        train_dataset=entry["train_dataset"],
        format=entry["fmt"].upper(),
        file_mb=entry["file_mb"],
        param_m=param_m,
    )
    if lat:
        row["classify_ms_median"] = lat["classify"]["median"]
        row["classify_ms_mean"]   = lat["classify"]["mean"]
        row["classify_ms_std"]    = lat["classify"]["std"]
        row["classify_ms_p95"]    = lat["classify"]["p95"]
        row["classify_ms_p99"]    = lat["classify"]["p99"]
        row["total_ms_median"]    = lat["total"]["median"]
        row["fps"]                = lat["fps"]
        row["fps_classify"]       = lat["fps_classify"]
        row["gpu_pct_peak"]       = lat["gpu_pct_peak"]
        row["gpu_ram_peak_mb"]    = lat["gpu_ram_peak_mb"]
        if lat["crop"]:
            row["crop_ms_median"] = lat["crop"]["median"]
    if acc:
        for ds_name, m in acc.items():
            for k, v in m.items():
                row[f"{k}_{ds_name}"] = v
    return row


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TensorRT vs TFLite benchmark on Jetson Orin Nano (no-seg)")
    parser.add_argument("--images",        default="scripts/test_images")
    parser.add_argument("--runs",          type=int, default=200,
                        help="Timed inference runs per image (default: 200)")
    parser.add_argument("--warmup",        type=int, default=50,
                        help="Warmup runs before timing (default: 50)")
    parser.add_argument("--out-dir",       default="outputs/ablation_noseg/JetsonResults",
                        help="Root output directory for Excel and plots")
    parser.add_argument("--out",           default=None,
                        help="Excel output path (overrides --out-dir)")
    parser.add_argument("--plots",         default=None,
                        help="Plots output directory (overrides --out-dir)")
    parser.add_argument("--suffix",        default="none",
                        help="'none' or 'none_sens'")
    parser.add_argument("--precision",     default="fp16", choices=["fp16", "fp32"],
                        help="TRT engine precision to benchmark (default: fp16)")
    parser.add_argument("--no-plots",      action="store_true")
    parser.add_argument("--latency-only",  action="store_true")
    parser.add_argument("--accuracy-only", action="store_true")
    parser.add_argument("--trt-only",      action="store_true",
                        help="Skip PT and TFLite — benchmark TensorRT engines only")
    parser.add_argument("--no-crop",       action="store_true",
                        help="Disable YOLO lesion crop — use full image with centre-crop "
                             "(correct for no-seg models trained on full images)")
    args = parser.parse_args()

    # resolve output paths
    out_dir_base = _PROJECT_ROOT / args.out_dir
    out_path     = _PROJECT_ROOT / args.out if args.out else (
                       out_dir_base / f"benchmark_{args.suffix}_{args.precision}.xlsx")
    plots_dir    = _PROJECT_ROOT / args.plots if args.plots else (
                       out_dir_base / "plots" / f"{args.suffix}_{args.precision}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_cfg           = load_config()
    ablation_noseg_dir = Path(base_cfg.paths.outputs) / "ablation_noseg"
    device             = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _ckpt_dir   = Path(base_cfg.paths.outputs) / "detection" / "checkpoints"
    yolo_engine = _ckpt_dir / f"best_{args.precision}.engine"
    yolo_pt     = _ckpt_dir / "best.pt"

    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    pt_entries  = _scan_pt(ablation_noseg_dir, args.suffix)
    trt_entries = _scan(ablation_noseg_dir, args.suffix, "trt", args.precision)
    tfl_entries = _scan(ablation_noseg_dir, args.suffix, "tfl")

    print(f"\n  Device       : {gpu_name}")
    print(f"  Suffix       : _{args.suffix}")
    print(f"  PT checkpoints: {len(pt_entries)}")
    print(f"  TRT engines   : {len(trt_entries)}")
    print(f"  TFLite models : {len(tfl_entries)}")
    # For benchmark crop: always use .pt on CPU — TRT YOLO engine needs torch.cuda
    # which is unavailable on this Jetson (driver/PyTorch version mismatch).
    # The TRT engine is built and stored for actual deployment use.
    crop_model = None
    if args.no_crop:
        print(f"  YOLO crop    : disabled (--no-crop) — full image + centre-crop")
    else:
        _yolo_mode = ("TRT (GPU)" if yolo_engine.exists() else
                      "PT (CPU)"  if yolo_pt.exists()     else "centre-crop fallback")
        print(f"  YOLO crop    : {_yolo_mode}")
        if yolo_engine.exists():
            try:
                crop_model = YOLOCropTRT(yolo_engine)
                print("  YOLO crop TRT engine loaded (GPU — no torch.cuda).\n")
            except Exception as e:
                print(f"  [WARN] YOLO TRT load failed ({e}) — trying .pt CPU fallback\n")
        if crop_model is None and yolo_pt.exists():
            try:
                crop_model = YOLOCrop(yolo_pt)
                print("  YOLO crop .pt loaded (CPU fallback).\n")
            except Exception as e:
                print(f"  [WARN] YOLO .pt load failed ({e}) — centre-crop fallback\n")
        if crop_model is None:
            print("  YOLO crop not found — centre-crop fallback\n")
    print(f"  Runs         : {args.warmup} warmup + {args.runs} timed\n")

    # load test CSVs
    test_dfs = {}
    if not args.latency_only:
        import pandas as pd
        splits_dir = Path(base_cfg.paths.data_splits)
        for ds in ALL_DATASETS:
            try:
                df = pd.read_csv(splits_dir / "cls_test.csv")
                df = df[df["dataset_source"] == ds].copy()
                df["binary_label"] = (df["label_str"] == "mel").astype(int)
                # remap Windows absolute paths (c:\...) to the NoMachine mount
                _nm_root = "/home/dani/Desktop/C on Player (NoMachine)/"
                if df["image_path"].str.match(r"^[Cc]:[/\\]").any():
                    df["image_path"] = (
                        df["image_path"]
                        .str.replace(r"^[Cc]:[/\\]", _nm_root, regex=True)
                        .str.replace("\\", "/", regex=False)
                    )
                if not df.empty:
                    test_dfs[ds] = df
                    print(f"  {ds:<12} test={len(df)}  mel={int(df['binary_label'].sum())}")
            except Exception as e:
                print(f"  [SKIP] {ds}: {e}")
        print()

    # load latency images
    lat_images: List[np.ndarray] = []
    if not args.accuracy_only:
        img_dir = _PROJECT_ROOT / args.images
        if img_dir.exists():
            paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
            lat_images = [img for img in (cv2.imread(str(p)) for p in paths)
                          if img is not None]
        if not lat_images:
            print(f"  [WARN] No images in {img_dir} — skipping latency benchmark\n")

    pt_rows:  List[Dict] = []
    trt_rows: List[Dict] = []
    tfl_rows: List[Dict] = []

    # ── PyTorch ───────────────────────────────────────────────────────────────
    if args.trt_only:
        print("  ── PyTorch (.pt) ─────────────────────────── [SKIPPED --trt-only]")
    else:
        print("  ── PyTorch (.pt) ───────────────────────────────────────")
    if not args.trt_only:
      for entry in pt_entries:
        mn, tds = entry["model"], entry["train_dataset"]
        cfg      = load_config(mn)
        inp_size = getattr(cfg, "input_size", 224)
        print(f"  [{mn}/{tds}]", end="", flush=True)
        try:
            clf = ClassifierPT(entry["path"], mn, cfg, uses_metadata(mn), inp_size)
        except Exception as e:
            print(f"  [ERROR] {e}"); continue

        lat = None
        if lat_images and not args.accuracy_only:
            try:
                lat = _bench_latency(lat_images, crop_model, clf, args.warmup, args.runs)
                print(f"  classify={lat['classify']['median']}ms  fps={lat['fps']}", end="")
            except Exception as e:
                print(f"  [latency ERR] {e}", end="")

        acc = None
        if test_dfs and not args.latency_only:
            try:
                acc = _eval_accuracy(clf, mn, test_dfs)
                sens_h = acc.get("ham10000", {}).get("sensitivity", "—")
                auc_h  = acc.get("ham10000", {}).get("auc", "—")
                print(f"  sens_ham={sens_h}  auc_ham={auc_h}", end="")
            except Exception as e:
                print(f"  [accuracy ERR] {e}", end="")
                traceback.print_exc()

        param_m = getattr(clf, "param_m", None)
        clf.close()
        print()
        pt_rows.append(_build_row(entry, lat, acc, param_m=param_m))

    # ── TensorRT ──────────────────────────────────────────────────────────────
    print(f"\n  ── TensorRT {args.precision.upper()} ──────────────────────────────────────")
    for entry in trt_entries:
        mn, tds = entry["model"], entry["train_dataset"]
        cfg      = load_config(mn)
        inp_size = getattr(cfg, "input_size", 224)
        print(f"  [{mn}/{tds}]", end="", flush=True)
        try:
            clf = ClassifierTRT(entry["path"], uses_metadata(mn), inp_size)
        except Exception as e:
            print(f"  [ERROR] {e}"); continue

        lat = None
        if lat_images and not args.accuracy_only:
            try:
                lat = _bench_latency(lat_images, crop_model, clf, args.warmup, args.runs)
                print(f"  classify={lat['classify']['median']}ms  fps={lat['fps']}", end="")
            except Exception as e:
                print(f"  [latency ERR] {e}", end="")

        acc = None
        if test_dfs and not args.latency_only:
            try:
                acc = _eval_accuracy(clf, mn, test_dfs)
                sens_h = acc.get("ham10000", {}).get("sensitivity", "—")
                auc_h  = acc.get("ham10000", {}).get("auc", "—")
                print(f"  sens_ham={sens_h}  auc_ham={auc_h}", end="")
            except Exception as e:
                print(f"  [accuracy ERR] {e}", end="")
                traceback.print_exc()

        clf.close()
        print()
        trt_rows.append(_build_row(entry, lat, acc))

    # ── TFLite ────────────────────────────────────────────────────────────────
    if args.trt_only:
        print("\n  ── TFLite FP32 ──────────────────────────── [SKIPPED --trt-only]")
    else:
        print("\n  ── TFLite FP32 ─────────────────────────────────────────")
    for entry in ([] if args.trt_only else tfl_entries):
        mn, tds = entry["model"], entry["train_dataset"]
        cfg      = load_config(mn)
        inp_size = getattr(cfg, "input_size", 224)
        print(f"  [{mn}/{tds}]", end="", flush=True)
        try:
            clf = ClassifierTFLite(entry["path"], uses_metadata(mn), inp_size)
        except Exception as e:
            print(f"  [ERROR] {e}"); continue

        lat = None
        if lat_images and not args.accuracy_only:
            try:
                lat = _bench_latency(lat_images, crop_model, clf, args.warmup, args.runs)
                print(f"  classify={lat['classify']['median']}ms  fps={lat['fps']}", end="")
            except Exception as e:
                print(f"  [latency ERR] {e}", end="")

        acc = None
        if test_dfs and not args.latency_only:
            try:
                acc = _eval_accuracy(clf, mn, test_dfs)
                sens_h = acc.get("ham10000", {}).get("sensitivity", "—")
                auc_h  = acc.get("ham10000", {}).get("auc", "—")
                print(f"  sens_ham={sens_h}  auc_ham={auc_h}", end="")
            except Exception as e:
                print(f"  [accuracy ERR] {e}", end="")

        clf.close()
        print()
        tfl_rows.append(_build_row(entry, lat, acc))

    # ── Excel ─────────────────────────────────────────────────────────────────
    import pandas as pd

    id_cols  = ["model", "train_dataset", "format", "file_mb", "param_m"]
    lat_cols = ["classify_ms_median", "classify_ms_mean", "classify_ms_std",
                "classify_ms_p95", "classify_ms_p99", "crop_ms_median",
                "total_ms_median", "fps", "fps_classify",
                "gpu_pct_peak", "gpu_ram_peak_mb"]
    acc_cols = []
    for ds in ALL_DATASETS:
        acc_cols += [f"threshold_{ds}", f"auc_{ds}", f"sensitivity_{ds}",
                     f"specificity_{ds}", f"f1_{ds}", f"f2_{ds}",
                     f"accuracy_{ds}", f"tp_{ds}", f"tn_{ds}", f"fp_{ds}", f"fn_{ds}"]

    def _to_df(rows):
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        ordered = [c for c in id_cols + lat_cols + acc_cols if c in df.columns]
        extra   = [c for c in df.columns if c not in ordered]
        return df[ordered + extra].fillna("")

    df_trt = _to_df(trt_rows)
    df_tfl = _to_df(tfl_rows)

    df_pt  = _to_df(pt_rows)
    if not df_pt.empty:
        write_excel_sheet(out_path, "PT_Results", df_pt)
    if not df_trt.empty:
        write_excel_sheet(out_path, "TRT_Results", df_trt)
    if not df_tfl.empty:
        write_excel_sheet(out_path, "TFLite_Results", df_tfl)

    # 3-way comparison sheet
    all_map = {
        "pt":  {(r["model"], r["train_dataset"]): r for r in pt_rows},
        "trt": {(r["model"], r["train_dataset"]): r for r in trt_rows},
        "tfl": {(r["model"], r["train_dataset"]): r for r in tfl_rows},
    }
    all_keys_cmp = sorted(set(all_map["pt"]) | set(all_map["trt"]) | set(all_map["tfl"]))
    cmp_rows = []
    for key in all_keys_cmp:
        mn, tds = key
        p = all_map["pt"].get(key,  {})
        t = all_map["trt"].get(key, {})
        f = all_map["tfl"].get(key, {})
        cmp: Dict[str, Any] = dict(
            model=mn, train_dataset=tds,
            param_m=p.get("param_m"),
            pt_file_mb=p.get("file_mb"),  trt_file_mb=t.get("file_mb"),
            tfl_file_mb=f.get("file_mb"),
            pt_classify_ms=p.get("classify_ms_median"),
            trt_classify_ms=t.get("classify_ms_median"),
            tfl_classify_ms=f.get("classify_ms_median"),
            pt_fps=p.get("fps"),  trt_fps=t.get("fps"),  tfl_fps=f.get("fps"),
            pt_gpu_ram_mb=p.get("gpu_ram_peak_mb"),
            trt_gpu_ram_mb=t.get("gpu_ram_peak_mb"),
        )
        # speedups vs PyTorch
        pt_ms = p.get("classify_ms_median")
        if pt_ms:
            if t.get("classify_ms_median"):
                cmp["speedup_trt_vs_pt"] = round(pt_ms / t["classify_ms_median"], 2)
            if f.get("classify_ms_median"):
                cmp["speedup_tfl_vs_pt"] = round(pt_ms / f["classify_ms_median"], 2)
        for ds in ALL_DATASETS:
            for fmt, src in [("pt", p), ("trt", t), ("tfl", f)]:
                cmp[f"{fmt}_sens_{ds}"] = src.get(f"sensitivity_{ds}")
                cmp[f"{fmt}_auc_{ds}"]  = src.get(f"auc_{ds}")
                cmp[f"{fmt}_f2_{ds}"]   = src.get(f"f2_{ds}")
        cmp_rows.append(cmp)
    write_excel_sheet(out_path, "Comparison", pd.DataFrame(cmp_rows).fillna(""))

    all_rows = pt_rows + trt_rows + tfl_rows
    if all_rows:
        df_all = _to_df(all_rows)
        if "classify_ms_median" in df_all.columns:
            write_excel_sheet(out_path, "All_By_Latency",
                              df_all.sort_values("classify_ms_median",
                                                  ascending=True, na_position="last"))
        if "auc_ham10000" in df_all.columns:
            write_excel_sheet(out_path, "All_By_AUC_HAM",
                              df_all.sort_values("auc_ham10000",
                                                  ascending=False, na_position="last"))

    print(f"\n  Saved → {out_path}")

    # ── plots ─────────────────────────────────────────────────────────────────
    if not args.no_plots:
        _plot(pt_rows, tfl_rows, trt_rows, plots_dir)

    # ── console summary ───────────────────────────────────────────────────────
    sep = "=" * 70
    print(f"\n{sep}")
    for fmt_label, rows in [("PT ", pt_rows), ("TFL", tfl_rows), ("TRT", trt_rows)]:
        meds = [r["classify_ms_median"] for r in rows if r.get("classify_ms_median")]
        if meds:
            print(f"  {fmt_label} median classify : {np.median(meds):.1f} ms  "
                  f"(range {min(meds):.1f}–{max(meds):.1f} ms)")
    if pt_rows and trt_rows:
        pt_med  = np.median([r["classify_ms_median"] for r in pt_rows
                              if r.get("classify_ms_median")] or [1])
        trt_med = np.median([r["classify_ms_median"] for r in trt_rows
                              if r.get("classify_ms_median")] or [1])
        print(f"  TRT speedup vs PT    : ×{pt_med/trt_med:.1f}")
    print()
    for ds in ["ham10000", "isic2019"]:
        col = f"sensitivity_{ds}"
        for rows, fmt in [(pt_rows, "PT"), (tfl_rows, "TFL"), (trt_rows, "TRT")]:
            vals = [r[col] for r in rows if r.get(col)]
            if vals:
                print(f"  Best {fmt} sens {ds:<10}: {max(vals):.4f}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
