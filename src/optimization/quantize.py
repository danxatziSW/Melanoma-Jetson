from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def export_to_onnx(
    model: nn.Module,
    input_shape: tuple,
    output_path: str | Path,
    opset: int = 17,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.zeros(input_shape)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version=opset,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )
    return output_path


def quantize_onnx_int8(
    onnx_model_path: str | Path,
    output_path: str | Path,
    calibration_loader: DataLoader | None = None,
) -> Path:
    # dynamic quantization — no calibration data needed
    from onnxruntime.quantization import quantize_dynamic, QuantType

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(onnx_model_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
    )
    return output_path


def quantize_pytorch_dynamic(model: nn.Module) -> nn.Module:
    return torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8,
    )


def quantize_pytorch_fp16(model: nn.Module) -> nn.Module:
    return model.half()


def run_onnx_inference(onnx_path: str | Path, input_np):
    import onnxruntime as ort
    import numpy as np

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    result = sess.run(None, {input_name: input_np.astype(np.float32)})
    return result[0]
