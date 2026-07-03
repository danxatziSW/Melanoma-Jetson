from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from src.utils.io import append_excel_row


class CSVEpochLogger:
    def __init__(self, xlsx_path: str | Path):
        self.xlsx_path = Path(xlsx_path)
        self.xlsx_path.parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch: int, **metrics) -> None:
        row = {"epoch": epoch, **{k: round(float(v), 6) for k, v in metrics.items()}}
        append_excel_row(self.xlsx_path, "TrainLog", row)


class ModelCheckpoint:
    def __init__(
        self,
        save_dir: str | Path,
        monitor: str = "val_auc",
        mode: str = "max",
        filename: str = "best_model",
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor  = monitor
        self.mode     = mode
        self.filename = filename
        self.best = float("-inf") if mode == "max" else float("inf")

    def on_epoch_end(self, model: nn.Module, **metrics) -> bool:
        value = metrics.get(self.monitor, None)
        if value is None:
            return False
        improved = (self.mode == "max" and value > self.best) or \
                   (self.mode == "min" and value < self.best)
        if improved:
            self.best = value
            torch.save(model.state_dict(), self.save_dir / f"{self.filename}.pt")
        return improved


class EarlyStopping:
    def __init__(self, patience: int = 10, monitor: str = "val_auc", mode: str = "max"):
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.best = float("-inf") if mode == "max" else float("inf")
        self.counter = 0

    def on_epoch_end(self, **metrics) -> bool:
        value = metrics.get(self.monitor, None)
        if value is None:
            return False
        improved = (self.mode == "max" and value > self.best) or \
                   (self.mode == "min" and value < self.best)
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience
