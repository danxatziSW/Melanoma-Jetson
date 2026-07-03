from __future__ import annotations

import os
import time
import tracemalloc
import types
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)

from src.utils.io import write_excel_sheet


def compute_specificity(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    specificities = []
    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        denom = tn + fp
        specificities.append(tn / denom if denom > 0 else 0.0)
    return float(np.mean(specificities))


def find_optimal_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    num_classes: int,
    class_names: list[str] | None = None,
) -> dict[str, float]:
    thresholds = {}
    for i in range(num_classes):
        label = class_names[i] if class_names else str(i)
        binary_true = (y_true == i).astype(int)
        scores = y_proba[:, i]

        best_thresh, best_j = 0.5, -1.0
        for t in np.linspace(0.05, 0.95, 91):
            pred = (scores >= t).astype(int)
            tp = ((pred == 1) & (binary_true == 1)).sum()
            fn = ((pred == 0) & (binary_true == 1)).sum()
            fp = ((pred == 1) & (binary_true == 0)).sum()
            tn = ((pred == 0) & (binary_true == 0)).sum()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            j = sens + spec - 1.0
            if j > best_j:
                best_j, best_thresh = j, t
        thresholds[label] = round(float(best_thresh), 4)
    return thresholds


class DiagnosticMetrics:
    @staticmethod
    def compute(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        y_pred: np.ndarray,
        num_classes: int,
        class_names: list[str] | None = None,
    ) -> dict:
        accuracy = float(accuracy_score(y_true, y_pred))
        sensitivity = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        specificity = compute_specificity(y_true, y_pred, num_classes)
        f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        try:
            auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
        except Exception:
            auc = float("nan")
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).tolist()
        thresholds = find_optimal_thresholds(y_true, y_proba, num_classes, class_names)
        return {
            "accuracy": accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "f1_score": f1,
            "auc_roc": auc,
            "confusion_matrix": cm,
            "thresholds": thresholds,
        }

    @staticmethod
    def save(metrics: dict, output_dir: Path | str, model_name: str, class_names: list[str] | None = None) -> None:
        output_dir = Path(output_dir) / "classifiers" / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        xlsx_path = output_dir / f"{model_name}_metrics.xlsx"

        scalar = {k: v for k, v in metrics.items() if k not in ("confusion_matrix", "thresholds")}
        write_excel_sheet(xlsx_path, "EvalMetrics", scalar)

        cm = np.array(metrics["confusion_matrix"])
        labels = class_names if class_names else [str(i) for i in range(cm.shape[0])]
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        cm_df.index.name = "Actual \\ Predicted"
        write_excel_sheet(xlsx_path, "ConfusionMatrix", cm_df.reset_index())

        thresh_df = pd.DataFrame(
            [{"class": k, "optimal_threshold": v} for k, v in metrics["thresholds"].items()]
        )
        write_excel_sheet(xlsx_path, "Thresholds", thresh_df)


class EdgeMetrics:
    @staticmethod
    def measure_load_time(model_path: str | Path, build_fn: Callable, config: types.SimpleNamespace) -> float:
        t0 = time.perf_counter()
        model = build_fn(config, config.num_classes)
        state = torch.load(str(model_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        t1 = time.perf_counter()
        return (t1 - t0) * 1000.0

    @staticmethod
    def measure_inference_latency(
        model: nn.Module,
        input_tensor: torch.Tensor,
        n_warmup: int = 10,
        n_runs: int = 100,
    ) -> float:
        model.eval()
        with torch.no_grad():
            for _ in range(n_warmup):
                model(input_tensor)
            times = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                model(input_tensor)
                times.append((time.perf_counter() - t0) * 1000.0)
        return float(np.median(times))

    @staticmethod
    def measure_memory_footprint(model: nn.Module, input_tensor: torch.Tensor) -> float:
        model.eval()
        tracemalloc.start()
        with torch.no_grad():
            model(input_tensor)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak / (1024 ** 2)

    @staticmethod
    def measure_model_size(model_path: str | Path) -> float:
        return os.path.getsize(str(model_path)) / (1024 ** 2)

    @staticmethod
    def measure_flops(model: nn.Module, input_tensor: torch.Tensor) -> dict:
        total_macs = [0]
        total_params = [0]
        hooks = []

        def _conv_hook(module, inp, out):
            b = inp[0].size(0)
            out_h, out_w = out.shape[-2], out.shape[-1]
            in_c = module.in_channels // (module.groups or 1)
            k_h, k_w = module.kernel_size
            macs = b * out_h * out_w * in_c * k_h * k_w * out.shape[1]
            total_macs[0] += macs

        def _linear_hook(module, inp, out):
            b = inp[0].size(0)
            total_macs[0] += b * module.in_features * module.out_features

        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                hooks.append(m.register_forward_hook(_conv_hook))
            elif isinstance(m, nn.Linear):
                hooks.append(m.register_forward_hook(_linear_hook))

        for p in model.parameters():
            total_params[0] += p.numel()

        model.eval()
        with torch.no_grad():
            model(input_tensor)

        for h in hooks:
            h.remove()

        return {
            "GMACs": round(total_macs[0] / 1e9, 4),
            "params_M": round(total_params[0] / 1e6, 4),
        }

    @staticmethod
    def save(metrics: dict, output_dir: Path | str, model_name: str) -> None:
        output_dir = Path(output_dir) / "classifiers" / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        xlsx_path = output_dir / f"{model_name}_metrics.xlsx"
        write_excel_sheet(xlsx_path, "EdgeMetrics", metrics)
