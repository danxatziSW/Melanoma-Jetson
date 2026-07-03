from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class DetectionResult:
    found: bool
    bbox_xyxy: Tuple[int, int, int, int]  # x1, y1, x2, y2 in pixel coords
    confidence: float
    latency_ms: float
    source: str  # "yolov8" | "centre_crop" | "full_image"


class LesionDetector:
    FALLBACK_CENTRE_FRACTION = 0.70

    # valid lesion bbox: between 2% and 65% of image area, aspect ratio ≤ 3.5
    MIN_AREA_FRACTION = 0.02
    MAX_AREA_FRACTION = 0.65
    MAX_ASPECT_RATIO  = 3.5

    def __init__(
        self,
        weights_path: Optional[str | Path] = None,
        conf_threshold: float = 0.35,
        imgsz: int = 640,
        device: str = "cpu",
    ) -> None:
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.device = device
        self._model = None
        self._backend: str = "none"

        if weights_path is not None:
            self._load(Path(weights_path))

    def _load(self, weights_path: Path) -> None:
        if not weights_path.exists():
            return
        try:
            from ultralytics import YOLO  # type: ignore
            self._model = YOLO(str(weights_path))
            self._model.to(self.device)
            self._backend = "yolov8"
        except ImportError:
            pass  # ultralytics not installed; remain in fallback mode

    def warmup(self) -> None:
        if self._backend != "yolov8":
            return
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self._model.predict(
            source=dummy, imgsz=self.imgsz, conf=self.conf_threshold,
            device=self.device, verbose=False, save=False,
        )

    @classmethod
    def from_pretrained(
        cls,
        weights_path: str | Path,
        conf_threshold: float = 0.35,
        imgsz: int = 640,
        device: str = "cpu",
    ) -> "LesionDetector":
        return cls(weights_path, conf_threshold, imgsz, device)

    def detect(self, image_bgr: np.ndarray) -> DetectionResult:
        h, w = image_bgr.shape[:2]

        if self._backend == "yolov8" and self._model is not None:
            return self._detect_yolov8(image_bgr, h, w)

        return self._centre_crop_fallback(h, w)

    def crop(self, image_bgr: np.ndarray, result: DetectionResult) -> np.ndarray:
        x1, y1, x2, y2 = result.bbox_xyxy
        return image_bgr[y1:y2, x1:x2]

    def _detect_yolov8(self, image_bgr: np.ndarray, h: int, w: int) -> DetectionResult:
        t0 = time.perf_counter()
        results = self._model.predict(
            source=image_bgr,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
            save=False,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        boxes = results[0].boxes if results and len(results) > 0 else None
        if boxes is None or len(boxes) == 0:
            fb = self._centre_crop_fallback(h, w)
            fb.latency_ms = latency_ms
            return fb

        confs = boxes.conf.cpu().numpy()
        best_idx = int(confs.argmax())
        best_conf = float(confs[best_idx])

        if best_conf < self.conf_threshold:
            fb = self._centre_crop_fallback(h, w)
            fb.latency_ms = latency_ms
            return fb

        xyxy = boxes.xyxy[best_idx].cpu().numpy().astype(int)
        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if not self._is_valid_bbox(x1, y1, x2, y2, h, w):
            fb = self._centre_crop_fallback(h, w)
            fb.latency_ms = latency_ms
            return fb

        return DetectionResult(
            found=True,
            bbox_xyxy=(x1, y1, x2, y2),
            confidence=best_conf,
            latency_ms=latency_ms,
            source="yolov8",
        )

    def _is_valid_bbox(self, x1: int, y1: int, x2: int, y2: int, h: int, w: int) -> bool:
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            return False
        area_frac = (bw * bh) / (w * h)
        if area_frac < self.MIN_AREA_FRACTION or area_frac > self.MAX_AREA_FRACTION:
            return False
        aspect = max(bw, bh) / min(bw, bh)
        return aspect <= self.MAX_ASPECT_RATIO

    def _centre_crop_fallback(self, h: int, w: int) -> DetectionResult:
        frac = self.FALLBACK_CENTRE_FRACTION
        ch, cw = int(h * frac), int(w * frac)
        y1 = (h - ch) // 2
        x1 = (w - cw) // 2
        return DetectionResult(
            found=False,
            bbox_xyxy=(x1, y1, x1 + cw, y1 + ch),
            confidence=0.0,
            latency_ms=0.0,
            source="centre_crop",
        )

    def detect_batch(self, images_bgr: list[np.ndarray]) -> list[DetectionResult]:
        return [self.detect(img) for img in images_bgr]
