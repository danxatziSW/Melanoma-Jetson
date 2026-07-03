from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from src.utils.io import write_excel_sheet


@dataclass
class EnsemblePrediction:
    predicted_class: int
    predicted_label: str
    ensemble_probabilities: list[float]
    per_model_predictions: dict[str, int] = field(default_factory=dict)
    per_model_confidences: dict[str, float] = field(default_factory=dict)
    agreement_score: float = 0.0
    requires_review: bool = False

    def to_dict(self) -> dict:
        return {
            "predicted_class": self.predicted_class,
            "predicted_label": self.predicted_label,
            "ensemble_probabilities": self.ensemble_probabilities,
            "agreement_score": self.agreement_score,
            "requires_review": self.requires_review,
            **{f"conf_{k}": v for k, v in self.per_model_confidences.items()},
            **{f"pred_{k}": v for k, v in self.per_model_predictions.items()},
        }


class EnsembleAgreement:
    def __init__(
        self,
        class_names: list[str],
        confidence_threshold: float = 0.60,
        uncertain_weight_factor: float = 0.25,
        min_agreement_fraction: float = 0.50,
        model_weights: dict[str, float] | None = None,
    ):
        self.class_names = class_names
        self.confidence_threshold = confidence_threshold
        self.uncertain_weight_factor = uncertain_weight_factor
        self.min_agreement_fraction = min_agreement_fraction
        self.model_weights = model_weights or {}

    def _get_probabilities(
        self,
        model_name: str,
        model: nn.Module,
        image_tensor: torch.Tensor,
        metadata_tensor: torch.Tensor | None,
    ) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            if metadata_tensor is not None:
                logits = model(image_tensor, metadata_tensor)
            else:
                logits = model(image_tensor)
        return torch.softmax(logits.squeeze(0), dim=0).cpu().numpy()

    def predict(
        self,
        image_tensor: torch.Tensor,
        models: dict[str, nn.Module],
        metadata_tensors: dict[str, torch.Tensor] | None = None,
    ) -> EnsemblePrediction:
        per_model_probas: dict[str, np.ndarray] = {}
        per_model_preds: dict[str, int] = {}
        per_model_confs: dict[str, float] = {}

        for name, model in models.items():
            meta = (metadata_tensors or {}).get(name)
            probas = self._get_probabilities(name, model, image_tensor, meta)
            per_model_probas[name] = probas
            per_model_preds[name] = int(np.argmax(probas))
            per_model_confs[name] = float(np.max(probas))

        # low-confidence models are down-weighted in the soft vote
        weighted_sum = np.zeros(len(self.class_names))
        weight_total = 0.0
        for name, probas in per_model_probas.items():
            conf = per_model_confs[name]
            base_w = self.model_weights.get(name, 1.0)
            w = base_w * (1.0 if conf >= self.confidence_threshold else self.uncertain_weight_factor)
            weighted_sum += w * probas
            weight_total += w

        ensemble_probas = (weighted_sum / weight_total).tolist()
        final_class = int(np.argmax(ensemble_probas))
        final_label = self.class_names[final_class]

        n_agree = sum(1 for p in per_model_preds.values() if p == final_class)
        agreement = n_agree / len(models)
        requires_review = agreement < self.min_agreement_fraction

        return EnsemblePrediction(
            predicted_class=final_class,
            predicted_label=final_label,
            ensemble_probabilities=ensemble_probas,
            per_model_predictions=per_model_preds,
            per_model_confidences=per_model_confs,
            agreement_score=round(agreement, 4),
            requires_review=requires_review,
        )

    def run_on_dataset(
        self,
        image_tensors: list[torch.Tensor],
        models: dict[str, nn.Module],
        metadata_tensors: list[dict[str, torch.Tensor]] | None = None,
        image_ids: list[str] | None = None,
    ) -> list[dict]:
        results = []
        for i, img in enumerate(image_tensors):
            meta = metadata_tensors[i] if metadata_tensors else None
            pred = self.predict(img.unsqueeze(0), models, meta)
            row = {"image_id": image_ids[i] if image_ids else i}
            row.update(pred.to_dict())
            results.append(row)
        return results

    @staticmethod
    def save_results(results: list[dict], output_dir: str | Path) -> None:
        output_dir = Path(output_dir) / "ensemble"
        output_dir.mkdir(parents=True, exist_ok=True)
        xlsx_path = output_dir / "ensemble_results.xlsx"
        write_excel_sheet(xlsx_path, "Results", results)

        import pandas as pd
        df = pd.DataFrame(results)
        summary = {
            "total_cases": len(df),
            "requires_review_count": int(df["requires_review"].sum()),
            "mean_agreement_score": round(float(df["agreement_score"].mean()), 4),
            "min_agreement_score": round(float(df["agreement_score"].min()), 4),
        }
        write_excel_sheet(xlsx_path, "ModelAgreement", [summary])
