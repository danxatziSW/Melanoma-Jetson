from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from .detector import DetectionResult, LesionDetector


@dataclass
class PipelineResult:
    detection: DetectionResult
    seg_mask: Optional[np.ndarray]          # H×W uint8 binary mask (on crop coords)
    classifier_logits: Optional[torch.Tensor]
    classifier_proba: Optional[np.ndarray]  # shape (num_classes,)
    predicted_class: int
    predicted_label: str
    confidence: float
    total_latency_ms: float
    stage_latencies_ms: Dict[str, float] = field(default_factory=dict)


class TwoStagePipeline:
    def __init__(
        self,
        detector: LesionDetector,
        seg_model: Optional[torch.nn.Module],
        classifier: Optional[torch.nn.Module],
        class_names: List[str],
        seg_input_size: int = 256,
        cls_input_size: int = 224,
        device: str = "cpu",
    ) -> None:
        self.detector = detector
        self.seg_model = seg_model
        self.classifier = classifier
        self.class_names = class_names
        self.seg_input_size = seg_input_size
        self.cls_input_size = cls_input_size
        self.device = device

        if self.seg_model is not None:
            self.seg_model = self.seg_model.to(device).eval()
        if self.classifier is not None:
            self.classifier = self.classifier.to(device).eval()

    @classmethod
    def build(cls, config, classifier_name: Optional[str] = None) -> "TwoStagePipeline":
        from src.models.registry import build_model
        from src.segmentation.model import build_unet
        from src.utils.config import load_config, load_segmentation_config

        seg_cfg = load_segmentation_config()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        det_weights = Path(config.paths.outputs) / "detection" / "checkpoints" / "best.pt"
        detector = LesionDetector(
            weights_path=det_weights if det_weights.exists() else None,
            conf_threshold=0.35,
            device=device,
        )

        seg_model = None
        seg_ckpt = Path(config.paths.outputs) / "segmentation" / "checkpoints" / "best_unet.pt"
        if seg_ckpt.exists():
            seg_model = build_unet(seg_cfg)
            seg_model.load_state_dict(
                torch.load(str(seg_ckpt), map_location="cpu", weights_only=True)
            )

        clf_model = None
        if classifier_name is not None:
            clf_cfg = load_config(classifier_name)
            clf_ckpt = (
                Path(clf_cfg.paths.outputs) / "classifiers" / classifier_name
                / "checkpoints" / "best_model.pt"
            )
            if clf_ckpt.exists():
                clf_model = build_model(classifier_name, clf_cfg, clf_cfg.num_classes)
                clf_model.load_state_dict(
                    torch.load(str(clf_ckpt), map_location="cpu", weights_only=True)
                )

        class_names = getattr(config, "class_names", [str(i) for i in range(config.num_classes)])
        return cls(
            detector=detector,
            seg_model=seg_model,
            classifier=clf_model,
            class_names=class_names,
            seg_input_size=getattr(seg_cfg, "input_size", 256),
            cls_input_size=getattr(config, "input_size", 224),
            device=device,
        )

    def run(self, image_bgr: np.ndarray) -> PipelineResult:
        t_total = time.perf_counter()
        stage_ms: Dict[str, float] = {}

        detection = self.detector.detect(image_bgr)
        stage_ms["detect"] = detection.latency_ms
        crop_bgr = self.detector.crop(image_bgr, detection)

        seg_mask = None
        if self.seg_model is not None:
            t0 = time.perf_counter()
            seg_mask = self._segment(crop_bgr)
            stage_ms["segment"] = (time.perf_counter() - t0) * 1000.0
            focused = self._soft_focus(crop_bgr, seg_mask)
        else:
            focused = crop_bgr

        logits = None
        proba = None
        pred_class = 0
        pred_label = self.class_names[0] if self.class_names else "0"
        confidence = 0.0

        if self.classifier is not None:
            t0 = time.perf_counter()
            logits, proba = self._classify(focused)
            stage_ms["classify"] = (time.perf_counter() - t0) * 1000.0
            pred_class = int(proba.argmax())
            pred_label = self.class_names[pred_class] if pred_class < len(self.class_names) else str(pred_class)
            confidence = float(proba[pred_class])

        total_ms = (time.perf_counter() - t_total) * 1000.0

        return PipelineResult(
            detection=detection,
            seg_mask=seg_mask,
            classifier_logits=logits,
            classifier_proba=proba,
            predicted_class=pred_class,
            predicted_label=pred_label,
            confidence=confidence,
            total_latency_ms=round(total_ms, 2),
            stage_latencies_ms={k: round(v, 2) for k, v in stage_ms.items()},
        )

    def _segment(self, crop_bgr: np.ndarray) -> np.ndarray:
        h, w = crop_bgr.shape[:2]
        resized = cv2.resize(crop_bgr, (self.seg_input_size, self.seg_input_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        rgb = (rgb - mean) / std
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            logit = self.seg_model(tensor)
            prob = torch.sigmoid(logit).squeeze().cpu().numpy()

        mask = (prob > 0.5).astype(np.uint8) * 255
        # resize mask back to original crop size
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return mask

    @staticmethod
    def _soft_focus(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        mask_f = mask.astype(np.float32) / 255.0
        blurred = cv2.GaussianBlur(mask_f, (51, 51), 0)
        weight = np.clip(0.3 + 0.7 * blurred, 0.0, 1.0)[:, :, np.newaxis]
        return np.clip(image_bgr.astype(np.float32) * weight, 0, 255).astype(np.uint8)

    def _classify(self, image_bgr: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        h, w = self.cls_input_size, self.cls_input_size
        resized = cv2.resize(image_bgr, (w, h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        rgb = (rgb - mean) / std
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            logits = self.classifier(tensor)
            proba = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        return logits, proba
