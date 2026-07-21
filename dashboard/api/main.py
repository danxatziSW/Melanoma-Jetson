from __future__ import annotations

import base64
import pickle
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from hardware import hw_monitor

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_DEPLOY_DIR = _PROJECT_ROOT / "outputs" / "ablation_noseg" / "meta" / "deployment"
_TRT_DIR    = _DEPLOY_DIR / "tensorrt"
_DET_ENGINE  = _PROJECT_ROOT / "outputs" / "detection" / "checkpoints" / "best_fp16.engine"

_MEAN  = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD   = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_NEUTRAL_META = np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
_QUALITY_THRESHOLD = 11.8


def _preprocess_chw(img_bgr: np.ndarray, size: int = 224) -> np.ndarray:
    big = int(size * 1.1)
    img = cv2.resize(img_bgr, (big, big))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    st  = (big - size) // 2
    img = img[st:st + size, st:st + size]
    return ((img - _MEAN) / _STD).transpose(2, 0, 1).astype(np.float32)


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
    """Single TensorRT engine — image-only or image+metadata inputs."""

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


class _MetaPipeline:
    """TRT FP16 ResNet-50 + MedFusionNet → sklearn meta-learner."""

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

    def infer(self, image_bgr: np.ndarray) -> Tuple[float, float, float]:
        """Returns (prob_resnet50, prob_medfusionnet, meta_prob)."""
        p1   = self.r50.predict_prob(image_bgr)
        p2   = self.mfn.predict_prob(image_bgr, _NEUTRAL_META)
        feat = self.scaler.transform([[p1, p2]])
        prob = float(self.clf.predict_proba(feat)[0, 1])
        return p1, p2, prob

    def close(self):
        self.r50.close()
        self.mfn.close()


_detector: Any = None
_pipeline: Optional[_MetaPipeline] = None


def _load_pipeline() -> None:
    global _pipeline
    try:
        _pipeline = _MetaPipeline(_DEPLOY_DIR / "meta_learner.pkl", _TRT_DIR, "fp16")
        print(f"[meta] TRT FP16 pipeline loaded — thresholds: {_pipeline.thresholds}")
    except Exception as exc:
        print(f"[meta] Load failed: {exc}")


class _TRTYOLODetector:
    """YOLOv8 TRT engine via ctypes — no PyTorch CUDA required."""

    _FALLBACK_FRAC = 0.70

    def __init__(self, engine_path: Path, conf_threshold: float = 0.35, imgsz: int = 640):
        import tensorrt as trt, ctypes
        self._ct = ctypes
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self._backend = "yolov8"
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

        self._h_in  = np.zeros(in_shape,  dtype=np.float32)
        self._h_out = np.zeros(out_shape, dtype=np.float32)
        self._d_in  = self._malloc(self._h_in.nbytes)
        self._d_out = self._malloc(self._h_out.nbytes)
        print(f"[det] TRT YOLO FP16 loaded  in={in_shape}  out={out_shape}")

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
        scale = min(self.imgsz / h, self.imgsz / w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(img_bgr, (nw, nh))
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        pad_y = (self.imgsz - nh) // 2
        pad_x = (self.imgsz - nw) // 2
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
        chw = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return chw.transpose(2, 0, 1), scale, pad_x, pad_y

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.45) -> list:
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while len(order):
            i = order[0]; keep.append(i)
            if len(order) == 1: break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou   = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[1:][iou < iou_thr]
        return keep

    def detect(self, img_bgr: np.ndarray):
        from src.detection.detector import DetectionResult
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

        # YOLOv8 output is [4+nc, na] e.g. [5, 8400] — transpose to [na, 4+nc]
        out = self._h_out[0]
        if out.shape[0] < out.shape[1]:   # features-first [5, 8400] → [8400, 5]
            out = out.T

        # out is now [na, 4+nc]: cols 0-3 = cx,cy,w,h; cols 4+ = class scores
        scores = out[:, 4:].max(axis=1) if out.shape[1] > 5 else out[:, 4]
        mask = scores >= self.conf_threshold
        if not mask.any():
            return self._fallback(h, w, latency_ms)

        scores = scores[mask]
        cx, cy = out[mask, 0], out[mask, 1]
        bw, bh = out[mask, 2], out[mask, 3]
        x1, y1 = cx - bw / 2, cy - bh / 2
        x2, y2 = cx + bw / 2, cy + bh / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        keep = self._nms(boxes, scores)
        if not keep:
            return self._fallback(h, w, latency_ms)

        best = keep[int(np.argmax(scores[keep]))]
        conf = float(scores[best])
        bx1 = int(max(0, (boxes[best, 0] - pad_x) / scale))
        by1 = int(max(0, (boxes[best, 1] - pad_y) / scale))
        bx2 = int(min(w, (boxes[best, 2] - pad_x) / scale))
        by2 = int(min(h, (boxes[best, 3] - pad_y) / scale))

        if bx2 <= bx1 or by2 <= by1:
            return self._fallback(h, w, latency_ms)

        return DetectionResult(
            found=True, bbox_xyxy=(bx1, by1, bx2, by2),
            confidence=conf, latency_ms=latency_ms, source="yolov8",
        )

    def _fallback(self, h, w, latency_ms=0.0):
        from src.detection.detector import DetectionResult
        frac = self._FALLBACK_FRAC
        ch, cw = int(h * frac), int(w * frac)
        y1 = (h - ch) // 2; x1 = (w - cw) // 2
        return DetectionResult(
            found=False, bbox_xyxy=(x1, y1, x1 + cw, y1 + ch),
            confidence=0.0, latency_ms=latency_ms, source="centre_crop",
        )

    def crop(self, img_bgr: np.ndarray, result) -> np.ndarray:
        x1, y1, x2, y2 = result.bbox_xyxy
        return img_bgr[y1:y2, x1:x2]

    def close(self):
        for ptr in [self._d_in, self._d_out]:
            if ptr:
                _cudart().cudaFree(self._ct.c_void_p(ptr))


def _load_detector() -> None:
    global _detector
    try:
        _detector = _TRTYOLODetector(_DET_ENGINE, conf_threshold=0.35)
    except Exception as exc:
        print(f"[det] TRT YOLO load failed: {exc} — falling back to centre crop")


def _quality_check(img_bgr: np.ndarray) -> Tuple[float, bool]:
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return round(lap_var, 2), lap_var >= _QUALITY_THRESHOLD


def _detect_and_crop(img_bgr: np.ndarray):
    if _detector is None:
        h, w = img_bgr.shape[:2]
        m    = int(min(h, w) * 0.1)
        crop = img_bgr[m:h - m, m:w - m]
        return crop, [m, m, w - m, h - m], "centre_crop", 0.0, 0.0
    det  = _detector.detect(img_bgr)
    crop = _detector.crop(img_bgr, det)
    return crop, list(det.bbox_xyxy), det.source, det.confidence, det.latency_ms


@asynccontextmanager
async def lifespan(app: FastAPI):
    hw_monitor.start()
    _load_pipeline()
    _load_detector()
    yield
    if _pipeline:
        _pipeline.close()
    if _detector:
        _detector.close()
    hw_monitor.stop()


app = FastAPI(title="Melanoma Detection API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/models")
def get_models() -> Dict[str, Any]:
    ready = _pipeline is not None
    return {
        "gpu":             [{"key": "trt_fp16", "model": "meta_learner", "train_dataset": "all"}] if ready else [],
        "tflite":          [],
        "gpu_ensemble":    [{"key": "resnet50|fp16"}, {"key": "medfusionnet|fp16"}] if ready else [],
        "tflite_ensemble": [],
        "mean_threshold":  _pipeline.thresholds.get("global", 0.20) if ready else 0.20,
    }


@app.get("/api/config")
def get_config() -> Dict[str, Any]:
    return {"mode": "trt_fp16", "model_key": "meta_learner_fp16"}


@app.post("/api/config")
def set_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {"mode": "trt_fp16", "model_key": "meta_learner_fp16"}


@app.get("/api/live/status")
def live_status() -> Dict[str, Any]:
    ready = _pipeline is not None
    return {
        "pipeline_loaded":  ready,
        "pipeline_error":   None,
        "det_loaded":       _detector is not None and _detector._backend == "yolov8",
        "available_gpu":    1 if ready else 0,
        "available_tflite": 0,
    }


@app.post("/api/live/infer-frame")
def infer_frame(payload: Dict[str, Any]) -> Dict[str, Any]:
    t_total  = time.perf_counter()
    stage_ms: Dict[str, float] = {}

    empty: Dict[str, Any] = {
        "predicted_label": None, "malignant_prob": None, "benign_prob": None,
        "confidence": None, "bbox_xyxy": None, "detection_source": None,
        "detection_confidence": None, "quality_score": None, "quality_ok": None,
        "crop_b64": None, "total_latency_ms": None, "stage_latencies_ms": {},
        "mode": "trt_fp16", "model_key": "meta_learner_fp16",
        "ensemble_details": [], "error": None,
    }

    if _pipeline is None:
        return {**empty, "error": "Meta-learner pipeline not loaded"}

    b64: str = payload.get("image", "")
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    try:
        img_bytes = base64.b64decode(b64)
        arr       = np.frombuffer(img_bytes, np.uint8)
        img_bgr   = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as exc:
        return {**empty, "error": f"Decode error: {exc}"}
    if img_bgr is None:
        return {**empty, "error": "Could not decode image"}

    detect_only: bool = payload.get("detect_only", False)

    t0 = time.perf_counter()
    quality_score, quality_ok = _quality_check(img_bgr)
    stage_ms["quality"] = round((time.perf_counter() - t0) * 1000, 2)

    if not quality_ok:
        total_ms = round((time.perf_counter() - t_total) * 1000, 2)
        return {
            **empty,
            "quality_score": quality_score, "quality_ok": False,
            "total_latency_ms": total_ms, "stage_latencies_ms": stage_ms,
        }

    t0 = time.perf_counter()
    try:
        crop, bbox, det_src, det_conf, det_ms = _detect_and_crop(img_bgr)
        stage_ms["detect"] = round(det_ms or (time.perf_counter() - t0) * 1000, 2)
    except Exception as exc:
        return {**empty, "error": f"Detection error: {exc}"}

    if detect_only or det_src != "yolov8":
        # No YOLO detection → skip classification, return detection result only
        total_ms = round((time.perf_counter() - t_total) * 1000, 2)
        return {
            **empty,
            "bbox_xyxy": bbox, "detection_source": det_src,
            "detection_confidence": round(det_conf, 4),
            "quality_score": quality_score, "quality_ok": True,
            "total_latency_ms": total_ms, "stage_latencies_ms": stage_ms,
        }

    _, jpg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    crop_b64 = "data:image/jpeg;base64," + base64.b64encode(jpg).decode()

    t0 = time.perf_counter()
    try:
        p_r50, p_mfn, meta_prob = _pipeline.infer(crop)
        stage_ms["classify"] = round((time.perf_counter() - t0) * 1000, 2)
    except Exception as exc:
        total_ms = round((time.perf_counter() - t_total) * 1000, 2)
        return {
            **empty,
            "bbox_xyxy": bbox, "detection_source": det_src,
            "quality_score": quality_score, "quality_ok": True,
            "crop_b64": crop_b64,
            "total_latency_ms": total_ms, "stage_latencies_ms": stage_ms,
            "error": f"Classify error: {exc}",
        }

    thr        = _pipeline.thresholds.get("global", 0.20)
    label      = "MALIGNANT" if meta_prob >= thr else "BENIGN"
    mal_prob   = meta_prob
    ben_prob   = 1.0 - meta_prob
    confidence = max(mal_prob, ben_prob)
    total_ms   = round((time.perf_counter() - t_total) * 1000, 2)

    return {
        "predicted_label":      label,
        "malignant_prob":       round(mal_prob, 4),
        "benign_prob":          round(ben_prob, 4),
        "confidence":           round(confidence, 4),
        "bbox_xyxy":            bbox,
        "detection_source":     det_src,
        "detection_confidence": round(det_conf, 4),
        "quality_score":        quality_score,
        "quality_ok":           True,
        "crop_b64":             crop_b64,
        "total_latency_ms":     total_ms,
        "stage_latencies_ms":   stage_ms,
        "mode":                 "trt_fp16",
        "model_key":            "meta_learner_fp16",
        "ensemble_details": [
            {"key": "resnet50|fp16",     "mal_prob": round(p_r50, 4)},
            {"key": "medfusionnet|fp16", "mal_prob": round(p_mfn, 4)},
            {"key": "meta_learner|fp16", "mal_prob": round(meta_prob, 4)},
        ],
        "error": None,
    }


@app.get("/api/live/hardware/current")
def hardware_current() -> Dict[str, Any]:
    return hw_monitor.current()


@app.get("/api/live/hardware/history")
def hardware_history(seconds: int = 120) -> Dict[str, Any]:
    return {"samples": hw_monitor.history(min(seconds, 120))}
