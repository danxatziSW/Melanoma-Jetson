"""Why meta-learner sensitivity is low on PH2 at 0.85.

Sweeps thresholds to see how much sensitivity recovers lower down, then lists the
missed melanomas with their probabilities to separate near-misses from confident errors.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.models.registry import build_model, uses_metadata
from src.utils.config import load_config
from evaluate_ph2_external import EvalDataset, _load_ph2, _run_inference, _metrics, _auc, MODEL1, DATASET1, MODEL2, DATASET2, AUG_MODE

base_cfg      = load_config()
device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
melanoma_root = Path(base_cfg.paths.melanoma_data)
ablation_dir  = Path(base_cfg.paths.outputs) / "ablation_noseg"

with open(ablation_dir / "meta" / "deployment" / "meta_learner.pkl", "rb") as f:
    bundle = pickle.load(f)
scaler, clf, thresholds = bundle["scaler"], bundle["clf"], bundle["thresholds"]

df = _load_ph2(melanoma_root)
labels = df["binary_label"].values

probs = {}
for model_name, dataset_name in [(MODEL1, DATASET1), (MODEL2, DATASET2)]:
    run_id = f"{model_name}_{AUG_MODE}"
    ckpt_f = ablation_dir / dataset_name / run_id / "checkpoints" / f"{run_id}.pt"
    config = load_config(model_name)
    wm     = uses_metadata(model_name)
    model  = build_model(model_name, config, num_classes=2)
    model.load_state_dict(torch.load(ckpt_f, map_location=device, weights_only=True))
    model.to(device).eval().requires_grad_(False)
    inp, nw = getattr(config, "input_size", 224), getattr(config, "num_workers", 0)
    loader = DataLoader(EvalDataset(df, inp, with_meta=wm), batch_size=config.batch_size,
                        shuffle=False, num_workers=nw)
    probs[f"{model_name}/{dataset_name}"] = _run_inference(model, loader, device, wm)
    del model

k1, k2 = f"{MODEL1}/{DATASET1}", f"{MODEL2}/{DATASET2}"
meta_prob = clf.predict_proba(scaler.transform(np.column_stack([probs[k1], probs[k2]])))[:, 1]

print(f"\n  Per-dataset thresholds stored in meta_learner.pkl: {thresholds}\n")
print(f"  {'Threshold (source)':<28}  {'Sens':>6}  {'Spec':>6}  {'F2':>6}")
print(f"  {'-'*54}")
for label, thr in [("0.85 (global / ham10000)", thresholds["ham10000"]),
                   ("0.74 (isic2019)",          thresholds["isic2019"]),
                   ("0.20 (isic2020)",          thresholds["isic2020"])]:
    m = _metrics(meta_prob, labels, thr)
    print(f"  {label:<28}  {m['sensitivity']:>6.4f}  {m['specificity']:>6.4f}  {m['f2']:>6.4f}")

# Post-hoc PH2-optimal threshold, diagnostic only
THR_RANGE = np.round(np.arange(0.20, 0.86, 0.01), 2)
best_thr, best_f2 = None, -1
for thr in THR_RANGE:
    f2 = _metrics(meta_prob, labels, thr)["f2"]
    if f2 > best_f2:
        best_f2, best_thr = f2, thr
m = _metrics(meta_prob, labels, best_thr)
print(f"  {'PH2-optimal (post-hoc): '+str(best_thr):<28}  {m['sensitivity']:>6.4f}  {m['specificity']:>6.4f}  {m['f2']:>6.4f}")

print(f"\n  AUC (threshold-independent): {_auc(meta_prob, labels)}\n")

# Missed melanomas at 0.85
mel_mask = labels == 1
mel_probs = meta_prob[mel_mask]
mel_names = df.loc[mel_mask, "image_name"].values
order = np.argsort(mel_probs)
print("  All 40 melanoma cases, meta_prob sorted ascending (missed at thr=0.85 marked):")
for name, p in zip(mel_names[order], mel_probs[order]):
    flag = "  <-- MISSED @0.85" if p < 0.85 else ""
    print(f"    {name}  prob={p:.4f}{flag}")
