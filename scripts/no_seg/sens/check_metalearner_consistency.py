"""Checks a refitted meta-learner against the deployed meta_learner.pkl.

The evaluation scripts each refit their own scaler and logistic regression rather
than loading the pickle. This recomputes that fit from the validation splits and
diffs the coefficients to confirm they are the same model.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.models.registry import build_model, uses_metadata
from src.utils.config import load_config
from src.utils.io import resolve_dataset_paths
from evaluate_ph2_external import EvalDataset, _run_inference, MODEL1, DATASET1, MODEL2, DATASET2, AUG_MODE

ALL_DATASETS = ["ham10000", "isic2019", "isic2020"]

base_cfg      = load_config()
device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
splits_dir    = Path(base_cfg.paths.data_splits)
melanoma_root = Path(base_cfg.paths.melanoma_data)
ablation_dir  = Path(base_cfg.paths.outputs) / "ablation_noseg"


def _load_df(csv_name: str, dataset_source: str) -> pd.DataFrame:
    df = pd.read_csv(splits_dir / csv_name)
    df = df[df["dataset_source"] == dataset_source].copy()
    df["binary_label"] = (df["label_str"] == "mel").astype(int)
    df = resolve_dataset_paths(df, melanoma_root)
    return df.reset_index(drop=True)


val_dfs = {ds: _load_df("cls_val.csv", ds) for ds in ALL_DATASETS}

val_probs: dict[str, dict[str, np.ndarray]] = {}
for model_name, dataset_name in [(MODEL1, DATASET1), (MODEL2, DATASET2)]:
    key    = f"{model_name}/{dataset_name}"
    run_id = f"{model_name}_{AUG_MODE}"
    ckpt_f = ablation_dir / dataset_name / run_id / "checkpoints" / f"{run_id}.pt"
    print(f"  Loading {key} for val-split re-inference ...")
    config = load_config(model_name)
    wm     = uses_metadata(model_name)
    model  = build_model(model_name, config, num_classes=2)
    model.load_state_dict(torch.load(ckpt_f, map_location=device, weights_only=True))
    model.to(device).eval().requires_grad_(False)
    inp, nw = getattr(config, "input_size", 224), getattr(config, "num_workers", 0)

    vp = {}
    for ds in ALL_DATASETS:
        loader = DataLoader(EvalDataset(val_dfs[ds], inp, with_meta=wm),
                            batch_size=config.batch_size, shuffle=False, num_workers=nw)
        vp[ds], _ = _run_inference(model, loader, device, wm), None
    val_probs[key] = vp
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

k1, k2 = f"{MODEL1}/{DATASET1}", f"{MODEL2}/{DATASET2}"
X_parts, y_parts = [], []
for ds in ALL_DATASETS:
    X_parts.append(np.column_stack([val_probs[k1][ds], val_probs[k2][ds]]))
    y_parts.append(val_dfs[ds]["binary_label"].values)
X_meta = np.vstack(X_parts)
y_meta = np.concatenate(y_parts)

scaler_refit = StandardScaler().fit(X_meta)
clf_refit    = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced")
clf_refit.fit(scaler_refit.transform(X_meta), y_meta)

with open(ablation_dir / "meta" / "deployment" / "meta_learner.pkl", "rb") as f:
    deployed = pickle.load(f)

print("\n  === Comparison ===")
print(f"  Deployed pkl  coef={deployed['clf'].coef_}   intercept={deployed['clf'].intercept_}")
print(f"  Refit (here)  coef={clf_refit.coef_}          intercept={clf_refit.intercept_}")
print(f"  Deployed pkl  scaler.mean_={deployed['scaler'].mean_}  scale_={deployed['scaler'].scale_}")
print(f"  Refit (here)  scaler.mean_={scaler_refit.mean_}  scale_={scaler_refit.scale_}")
same_clf    = np.allclose(deployed["clf"].coef_, clf_refit.coef_) and np.allclose(deployed["clf"].intercept_, clf_refit.intercept_)
same_scaler = np.allclose(deployed["scaler"].mean_, scaler_refit.mean_) and np.allclose(deployed["scaler"].scale_, scaler_refit.scale_)
print(f"\n  IDENTICAL fitted classifier : {same_clf}")
print(f"  IDENTICAL fitted scaler     : {same_scaler}")
