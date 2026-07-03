from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.callbacks import CSVEpochLogger, EarlyStopping, ModelCheckpoint


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        loss_fn: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: types.SimpleNamespace,
        model_name: str,
        uses_metadata: bool = False,
        run_name: str | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.model_name = model_name
        self.uses_metadata = uses_metadata
        self.num_classes = config.num_classes

        name    = run_name if run_name else model_name
        out_dir = Path(config.paths.outputs) / "classifiers" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        xlsx_path = out_dir / f"{name}_metrics.xlsx"
        ckpt_dir  = out_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.logger     = CSVEpochLogger(xlsx_path)
        self.checkpoint = ModelCheckpoint(ckpt_dir, monitor="val_auc", mode="max",
                                          filename=name)
        self.early_stop = EarlyStopping(
            patience=getattr(config, "early_stopping_patience", 10),
            monitor="val_auc", mode="max",
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if hasattr(self.loss_fn, "alpha") and self.loss_fn.alpha is not None:
            self.loss_fn.alpha = self.loss_fn.alpha.to(self.device)

    def _unpack_batch(self, batch):
        if self.uses_metadata:
            imgs, meta, labels = batch
            return imgs.to(self.device), meta.to(self.device), labels.to(self.device)
        imgs, labels = batch
        return imgs.to(self.device), None, labels.to(self.device)

    def _forward(self, imgs, meta):
        if self.uses_metadata and meta is not None:
            return self.model(imgs, meta)
        return self.model(imgs)

    def train_epoch(self) -> tuple[float, float]:
        self.model.train()
        total_loss, all_preds, all_labels = 0.0, [], []
        bar = tqdm(self.train_loader, desc="  train", leave=False,
                   unit="batch", dynamic_ncols=True, file=sys.stdout)
        for batch in bar:
            imgs, meta, labels = self._unpack_batch(batch)
            self.optimizer.zero_grad()
            logits = self._forward(imgs, meta)
            loss   = self.loss_fn(logits, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            bar.set_postfix(loss=f"{loss.item():.4f}")
        return total_loss / len(self.train_loader), float(accuracy_score(all_labels, all_preds))

    def val_epoch(self) -> tuple[float, float, float]:
        self.model.eval()
        total_loss, all_preds, all_labels, all_probas = 0.0, [], [], []
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="  val  ", leave=False,
                              unit="batch", dynamic_ncols=True, file=sys.stdout):
                imgs, meta, labels = self._unpack_batch(batch)
                logits  = self._forward(imgs, meta)
                loss    = self.loss_fn(logits, labels)
                total_loss += loss.item()
                probas  = torch.softmax(logits, dim=1).cpu().numpy()
                all_probas.append(probas)
                all_preds.extend(probas.argmax(axis=1))
                all_labels.extend(labels.cpu().numpy())
        avg_loss   = total_loss / len(self.val_loader)
        acc        = float(accuracy_score(all_labels, all_preds))
        all_probas = np.vstack(all_probas)
        try:
            auc = float(roc_auc_score(all_labels, all_probas, multi_class="ovr", average="macro"))
        except Exception:
            auc = float("nan")
        return avg_loss, acc, auc

    def fit(self) -> None:
        patience  = getattr(self.config, "early_stopping_patience", 10)
        gpu_label = torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU"
        print(f"\n  Model   : {self.model_name}", flush=True)
        print(f"  Device  : {gpu_label}", flush=True)
        print(f"  Batches : train={len(self.train_loader)}  val={len(self.val_loader)}\n", flush=True)

        epoch_bar = tqdm(
            range(1, self.config.epochs + 1),
            desc="Epochs",
            unit="ep",
            dynamic_ncols=True,
            file=sys.stdout,
        )

        for epoch in epoch_bar:
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, val_auc = self.val_epoch()

            if self.scheduler is not None:
                self.scheduler.step()

            metrics = dict(
                train_loss=train_loss, val_loss=val_loss,
                train_acc=train_acc,   val_acc=val_acc,
                val_auc=val_auc,
            )
            self.logger.on_epoch_end(epoch, **metrics)
            improved = self.checkpoint.on_epoch_end(self.model, **metrics)

            epoch_bar.set_postfix(
                loss=f"{train_loss:.4f}/{val_loss:.4f}",
                acc=f"{val_acc:.3f}",
                auc=f"{val_auc:.4f}",
            )

            star = " * saved" if improved else ""
            tqdm.write(
                f"  ep {epoch:02d}/{self.config.epochs}  "
                f"loss={train_loss:.4f}/{val_loss:.4f}  "
                f"acc={train_acc:.3f}/{val_acc:.3f}  "
                f"auc={val_auc:.4f}{star}"
            )

            if self.early_stop.on_epoch_end(**metrics):
                tqdm.write(f"  Early stop at epoch {epoch}")
                break
