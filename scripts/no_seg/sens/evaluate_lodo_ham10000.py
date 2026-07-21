"""Leave-one-dataset-out check for the deployed pair on HAM10000.

The shipped meta_learner.pkl pools all three validation sets to fit its scaler,
logistic regression and threshold, so HAM10000 shaped the meta-learner even though
it never touched either base classifier. This refits on ISIC 2019 + ISIC 2020
validation only, then evaluates on the HAM10000 test set.

  python scripts/no_seg/sens/evaluate_lodo_ham10000.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.models.registry import build_model, uses_metadata
from src.utils.config import load_config
from src.utils.io import resolve_dataset_paths, write_excel_sheet
from evaluate_ph2_external import _run_inference
from evaluate_3models_mean_sens import EvalDataset, _load_test_df, _metrics_at_threshold

MODEL1, DATASET1 = "resnet50",     "isic2019"
MODEL2, DATASET2 = "medfusionnet", "isic2020"
AUG_MODE          = "none_sens"
HELD_OUT          = "ham10000"
FIT_DATASETS      = ["isic2019", "isic2020"]
THRESHOLDS        = np.round(np.arange(0.20, 0.86, 0.01), 2)
BETA              = 2.0

base_cfg      = load_config()
device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
splits_dir    = Path(base_cfg.paths.data_splits)
melanoma_root = Path(base_cfg.paths.melanoma_data)
ablation_dir  = Path(base_cfg.paths.outputs) / "ablation_noseg"
out_xlsx      = ablation_dir / "meta" / "external_validation" / "evaluation_lodo_ham10000.xlsx"


def _load_val_df(dataset_source: str) -> pd.DataFrame:
    df = pd.read_csv(splits_dir / "cls_val.csv")
    df = df[df["dataset_source"] == dataset_source].copy()
    df["binary_label"] = (df["label_str"] == "mel").astype(int)
    df = resolve_dataset_paths(df, melanoma_root)
    return df.reset_index(drop=True)


def _load_model(model_name: str, dataset_name: str):
    run_id = f"{model_name}_{AUG_MODE}"
    ckpt_f = ablation_dir / dataset_name / run_id / "checkpoints" / f"{run_id}.pt"
    config = load_config(model_name)
    wm     = uses_metadata(model_name)
    model  = build_model(model_name, config, num_classes=2)
    model.load_state_dict(torch.load(ckpt_f, map_location=device, weights_only=True))
    model.to(device).eval().requires_grad_(False)
    return model, config, wm


def _infer(model, config, wm, df: pd.DataFrame) -> np.ndarray:
    inp, nw = getattr(config, "input_size", 224), getattr(config, "num_workers", 0)
    loader  = DataLoader(EvalDataset(df, inp, with_meta=wm), batch_size=config.batch_size,
                          shuffle=False, num_workers=nw, pin_memory=(nw > 0))
    return _run_inference(model, loader, device, wm)


def _best_f2_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    best_thr, best_f2 = THRESHOLDS[0], -1.0
    for thr in THRESHOLDS:
        f2 = _metrics_at_threshold(probs, labels, thr)["f2"]
        if f2 > best_f2:
            best_f2, best_thr = f2, thr
    return float(best_thr)


def main() -> None:
    print(f"\n  LODO check: fit meta-learner on {FIT_DATASETS} only, test on '{HELD_OUT}' TEST set")
    print(f"  Model 1 : {MODEL1}/{DATASET1}   Model 2 : {MODEL2}/{DATASET2}\n")

    m1, cfg1, wm1 = _load_model(MODEL1, DATASET1)
    m2, cfg2, wm2 = _load_model(MODEL2, DATASET2)

    # ISIC2019 + ISIC2020 validation only, HAM10000 excluded
    X_parts, y_parts = [], []
    for ds in FIT_DATASETS:
        val_df = _load_val_df(ds)
        p1 = _infer(m1, cfg1, wm1, val_df)
        p2 = _infer(m2, cfg2, wm2, val_df)
        X_parts.append(np.column_stack([p1, p2]))
        y_parts.append(val_df["binary_label"].values)
        print(f"  fit-val {ds:<10} n={len(val_df):<6} mel={int(val_df['binary_label'].sum())}")

    X_fit = np.vstack(X_parts)
    y_fit = np.concatenate(y_parts)

    scaler = StandardScaler().fit(X_fit)
    clf    = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced")
    clf.fit(scaler.transform(X_fit), y_fit)

    fit_meta_prob = clf.predict_proba(scaler.transform(X_fit))[:, 1]
    thr = _best_f2_threshold(fit_meta_prob, y_fit)
    print(f"\n  LODO threshold (F2-optimal on ISIC2019+ISIC2020 val only): {thr}\n")

    # HAM10000 test, held out from every fit above
    test_df = _load_test_df(splits_dir, HELD_OUT, melanoma_root)
    labels  = test_df["binary_label"].values
    p1_test = _infer(m1, cfg1, wm1, test_df)
    p2_test = _infer(m2, cfg2, wm2, test_df)
    meta_prob = clf.predict_proba(scaler.transform(np.column_stack([p1_test, p2_test])))[:, 1]

    m   = _metrics_at_threshold(meta_prob, labels, thr)
    auc = float(roc_auc_score(labels, meta_prob))

    print(f"  HAM10000 TEST (n={len(test_df)}, mel={int(labels.sum())}), LODO meta-learner, thr={thr}")
    print(f"    sensitivity={m['sensitivity']:.4f}  specificity={m['specificity']:.4f}  "
          f"precision={m['precision']:.4f}  f2={m['f2']:.4f}  f1={m['f1']:.4f}  auc={auc:.4f}")
    print(f"    TP={m['tp']}  TN={m['tn']}  FP={m['fp']}  FN={m['fn']}\n")

    row = dict(protocol="LODO (fit on ISIC2019+ISIC2020 val only)", held_out_test=HELD_OUT,
               threshold=thr, sensitivity=m["sensitivity"], specificity=m["specificity"],
               precision=m["precision"], f2=m["f2"], f1=m["f1"], auc=round(auc, 4),
               tp=m["tp"], tn=m["tn"], fp=m["fp"], fn=m["fn"], n=len(test_df),
               n_melanoma=int(labels.sum()))
    write_excel_sheet(out_xlsx, "LODO_HAM10000", pd.DataFrame([row]))
    print(f"  Saved -> {out_xlsx}\n")


if __name__ == "__main__":
    main()
