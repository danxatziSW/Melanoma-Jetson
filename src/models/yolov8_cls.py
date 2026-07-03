import types
from pathlib import Path


def build_yolov8_cls(config: types.SimpleNamespace, num_classes: int):
    try:
        from ultralytics import YOLO
        import torch
        import torch.nn as nn

        class _YOLOClsWrapper(nn.Module):
            # Ultralytics forward returns a tuple in train mode; unwrap to logits
            def __init__(self, inner: nn.Module):
                super().__init__()
                self.inner = inner

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = self.inner(x)
                return out[0] if isinstance(out, tuple) else out

        model = YOLO("yolov8n-cls.pt")
        head = model.model.model[-1]
        if hasattr(head, "linear"):
            in_features = head.linear.in_features
            head.linear = nn.Linear(in_features, num_classes)
        return _YOLOClsWrapper(model.model)

    except ImportError:
        import torch.nn as nn
        import timm

        fallback_note = (
            "ultralytics not installed — YOLOv8-cls replaced by MobileNetV3-RW "
            "(comparable parameter count ~4M). Install ultralytics for the real model."
        )
        note_path = Path(config.paths.outputs) / "yolov8_fallback.txt"
        note_path.parent.mkdir(parents=True, exist_ok=True)
        note_path.write_text(fallback_note)

        dropout = getattr(config, "dropout", 0.2)
        model = timm.create_model(
            "mobilenetv3_rw",
            pretrained=getattr(config, "pretrained", True),
            num_classes=num_classes,
            drop_rate=dropout,
        )
        return model
