from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.models.registry import build_model, uses_metadata
from src.utils.config import load_config
from src.utils.io import resolve_dataset_paths, write_excel_sheet

MODEL1, DATASET1 = "resnet50",     "isic2019"
MODEL2, DATASET2 = "medfusionnet", "isic2020"
AUG_MODE         = "none_sens"
ALL_DATASETS     = ["ham10000", "isic2019", "isic2020"]
BETA             = 2.0
THR_RANGE        = np.round(np.arange(0.20, 0.86, 0.01), 2)

_MEAN      = (0.485, 0.456, 0.406)
_STD       = (0.229, 0.224, 0.225)
_SITE_CATS = ["head/neck", "upper extremity", "lower extremity",
              "torso", "palms/soles", "oral/genital"]


class EvalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, input_size: int, with_meta: bool = False):
        self.df        = df.reset_index(drop=True)
        self.with_meta = with_meta
        self.transform = A.Compose([
            A.Resize(height=int(input_size * 1.1), width=int(input_size * 1.1)),
            A.CenterCrop(height=input_size, width=input_size),
            A.Normalize(mean=_MEAN, std=_STD),
            ToTensorV2(),
        ])
        if with_meta:
            self._encode_metadata()

    def _encode_metadata(self) -> None:
        df = self.df
        age_col = "age_approx" if "age_approx" in df.columns else None
        self.age = (
            (df[age_col].fillna(df[age_col].median()) / 100.0).values.astype(np.float32)
            if age_col else np.zeros(len(df), dtype=np.float32)
        )
        sex_col = "sex" if "sex" in df.columns else None
        self.sex = (
            df[sex_col].map({"male": 1.0, "female": 0.0}).fillna(0.5).values.astype(np.float32)
            if sex_col else np.full(len(df), 0.5, dtype=np.float32)
        )
        site_col = "anatom_site_general_challenge" if "anatom_site_general_challenge" in df.columns else None
        self.site_ohe = np.zeros((len(df), len(_SITE_CATS)), dtype=np.float32)
        if site_col:
            site_s = df[site_col].fillna("unknown")
            for i, cat in enumerate(_SITE_CATS):
                self.site_ohe[:, i] = (site_s == cat).astype(np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        image = cv2.imread(str(row["image_path"]))
        image = (
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if image is not None
            else np.zeros((224, 224, 3), dtype=np.uint8)
        )
        img_t = self.transform(image=image)["image"]
        label = int(row["binary_label"])
        if self.with_meta:
            meta = np.concatenate([[self.age[idx], self.sex[idx]], self.site_ohe[idx]])
            return img_t, torch.from_numpy(meta), label
        return img_t, label


def _load_df(splits_dir: Path, csv_name: str, dataset_source: str, melanoma_root: Path) -> pd.DataFrame:
    df = pd.read_csv(splits_dir / csv_name)
    df = df[df["dataset_source"] == dataset_source].copy()
    df["binary_label"] = (df["label_str"] == "mel").astype(int)
    df = resolve_dataset_paths(df, melanoma_root)
    return df


def _run_inference(model: nn.Module, loader: DataLoader,
                   device: torch.device, with_meta: bool) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, leave=False, unit="batch",
                          dynamic_ncols=True, file=sys.stdout):
            if with_meta:
                imgs, mdata, labels = batch
                logits = model(imgs.to(device), mdata.to(device))
            else:
                imgs, labels = batch
                logits = model(imgs.to(device))
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    return np.array(all_probs), np.array(all_labels)


def _get_loader(df: pd.DataFrame, model_name: str, config, with_meta: bool) -> DataLoader:
    inp  = getattr(config, "input_size", 224)
    nw   = getattr(config, "num_workers", 0)
    return DataLoader(EvalDataset(df, inp, with_meta=with_meta),
                      batch_size=config.batch_size, shuffle=False,
                      num_workers=nw, pin_memory=(nw > 0))


def _metrics(probs: np.ndarray, labels: np.ndarray, thr: float) -> dict:
    preds = (probs >= thr).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    b2   = BETA ** 2
    f2   = (1 + b2) * prec * sens / max(b2 * prec + sens, 1e-9)
    f1   = 2 * prec * sens / max(prec + sens, 1e-9)
    return dict(sensitivity=round(sens, 4), specificity=round(spec, 4),
                precision=round(prec, 4), f2=round(f2, 4), f1=round(f1, 4),
                tp=tp, tn=tn, fp=fp, fn=fn, threshold=round(thr, 2))


def _best_thr(probs: np.ndarray, labels: np.ndarray) -> float:
    best_thr, best = THR_RANGE[0], -1.0
    for thr in THR_RANGE:
        preds = (probs >= thr).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        sens = tp / max(tp + fn, 1)
        prec = tp / max(tp + fp, 1)
        b2 = BETA ** 2
        f2 = (1 + b2) * prec * sens / max(b2 * prec + sens, 1e-9)
        if f2 > best:
            best, best_thr = f2, thr
    return float(best_thr)


def _auc(probs, labels):
    try:
        return round(float(roc_auc_score(labels, probs)), 4)
    except Exception:
        return float("nan")


def main() -> None:
    base_cfg     = load_config()
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits_dir   = Path(base_cfg.paths.data_splits)
    melanoma_root = Path(base_cfg.paths.melanoma_data)
    ablation_dir = Path(base_cfg.paths.outputs) / "ablation_noseg"
    out_dir      = ablation_dir / "meta" / "deployment"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_xlsx = out_dir / "evaluation_deployment_pair.xlsx"

    print(f"\n  Deployment pair evaluation")
    print(f"  Model 1 : {MODEL1} / {DATASET1}")
    print(f"  Model 2 : {MODEL2} / {DATASET2}\n")

    val_dfs:  dict[str, pd.DataFrame] = {}
    test_dfs: dict[str, pd.DataFrame] = {}
    for ds in ALL_DATASETS:
        val_dfs[ds]  = _load_df(splits_dir, "cls_val.csv",  ds, melanoma_root)
        test_dfs[ds] = _load_df(splits_dir, "cls_test.csv", ds, melanoma_root)
        print(f"  {ds.upper():<12}  val={len(val_dfs[ds])}  test={len(test_dfs[ds])}"
              f"  mel_val={int(val_dfs[ds]['binary_label'].sum())}"
              f"  mel_test={int(test_dfs[ds]['binary_label'].sum())}")
    print()

    val_probs:  dict[str, dict[str, np.ndarray]] = {}
    test_probs: dict[str, dict[str, np.ndarray]] = {}

    for model_name, dataset_name in [(MODEL1, DATASET1), (MODEL2, DATASET2)]:
        key    = f"{model_name}/{dataset_name}"
        run_id = f"{model_name}_{AUG_MODE}"
        ckpt_f = ablation_dir / dataset_name / run_id / "checkpoints" / f"{run_id}.pt"

        if not ckpt_f.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_f}")

        print(f"  Loading {key} ...")
        config = load_config(model_name)
        wm     = uses_metadata(model_name)
        model  = build_model(model_name, config, num_classes=2)
        model.load_state_dict(torch.load(ckpt_f, map_location=device))
        model.to(device).eval().requires_grad_(False)

        vp, tp = {}, {}
        for ds in ALL_DATASETS:
            v_loader = _get_loader(val_dfs[ds],  model_name, config, wm)
            t_loader = _get_loader(test_dfs[ds], model_name, config, wm)
            vp[ds], _ = _run_inference(model, v_loader, device, wm)
            tp[ds], _ = _run_inference(model, t_loader, device, wm)
        val_probs[key]  = vp
        test_probs[key] = tp

        del model
        torch.cuda.empty_cache()

    k1 = f"{MODEL1}/{DATASET1}"
    k2 = f"{MODEL2}/{DATASET2}"

    print("\n  Fitting meta-learner on val set ...")
    X_parts, y_parts = [], []
    for ds in ALL_DATASETS:
        X_parts.append(np.column_stack([val_probs[k1][ds], val_probs[k2][ds]]))
        y_parts.append(val_dfs[ds]["binary_label"].values)
    X_meta   = np.vstack(X_parts)
    y_meta   = np.concatenate(y_parts)
    scaler   = StandardScaler()
    X_meta_s = scaler.fit_transform(X_meta)
    clf      = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced")
    clf.fit(X_meta_s, y_meta)
    print(f"  Meta weights: [{clf.coef_[0][0]:.4f}, {clf.coef_[0][1]:.4f}]  "
          f"bias={clf.intercept_[0]:.4f}\n")

    mean_val_pairs = [((val_probs[k1][ds] + val_probs[k2][ds]) / 2,
                       val_dfs[ds]["binary_label"].values) for ds in ALL_DATASETS]
    mean_global_thr_list = []
    for thr in THR_RANGE:
        score = float(np.mean([_metrics(p, l, thr)["f2"] for p, l in mean_val_pairs]))
        mean_global_thr_list.append((score, thr))
    mean_global_thr = max(mean_global_thr_list)[1]

    rows = []
    sep  = "-" * 72

    print(f"  {'Dataset':<12}  {'Approach':<22}  {'Sens':>6}  {'Spec':>6}  "
          f"{'F2':>6}  {'AUC':>6}  {'Thr':>5}")
    print(f"  {sep}")

    for ds in ALL_DATASETS:
        labels = test_dfs[ds]["binary_label"].values
        p1     = test_probs[k1][ds]
        p2     = test_probs[k2][ds]

        thr1 = _best_thr(val_probs[k1][ds], val_dfs[ds]["binary_label"].values)
        thr2 = _best_thr(val_probs[k2][ds], val_dfs[ds]["binary_label"].values)
        m_m1 = _metrics(p1, labels, thr1)
        m_m2 = _metrics(p2, labels, thr2)

        mean_prob = (p1 + p2) / 2
        m_mean    = _metrics(mean_prob, labels, mean_global_thr)

        X_test    = scaler.transform(np.column_stack([p1, p2]))
        meta_prob = clf.predict_proba(X_test)[:, 1]
        meta_thr  = _best_thr(meta_prob, labels)
        m_meta    = _metrics(meta_prob, labels, meta_thr)

        for approach, m, prob in [
            (f"{MODEL1}/{DATASET1}",  m_m1,  p1),
            (f"{MODEL2}/{DATASET2}",  m_m2,  p2),
            ("mean ensemble",         m_mean, mean_prob),
            ("meta-learner",          m_meta, meta_prob),
        ]:
            auc = _auc(prob, labels)
            print(f"  {ds.upper():<12}  {approach:<22}  "
                  f"{m['sensitivity']:>6.4f}  {m['specificity']:>6.4f}  "
                  f"{m['f2']:>6.4f}  {auc:>6.4f}  {m['threshold']:>5.2f}")
            rows.append(dict(
                dataset=ds, approach=approach,
                sensitivity=m["sensitivity"], specificity=m["specificity"],
                precision=m["precision"], f2=m["f2"], f1=m["f1"],
                auc=auc, threshold=m["threshold"],
                tp=m["tp"], tn=m["tn"], fp=m["fp"], fn=m["fn"],
            ))
        print(f"  {sep}")

    df_all = pd.DataFrame(rows)
    summary_rows = []
    print(f"\n  {'Approach':<22}  {'MinSens':>8}  {'AvgSens':>8}  {'AvgF2':>8}  {'AvgAUC':>8}")
    print(f"  {'-' * 60}")
    for approach in [f"{MODEL1}/{DATASET1}", f"{MODEL2}/{DATASET2}",
                     "mean ensemble", "meta-learner"]:
        sub = df_all[df_all["approach"] == approach]
        min_sens = sub["sensitivity"].min()
        avg_sens = sub["sensitivity"].mean()
        avg_f2   = sub["f2"].mean()
        avg_auc  = sub["auc"].mean()
        print(f"  {approach:<22}  {min_sens:>8.4f}  {avg_sens:>8.4f}  "
              f"{avg_f2:>8.4f}  {avg_auc:>8.4f}")
        summary_rows.append(dict(approach=approach,
                                 min_sensitivity=round(min_sens, 4),
                                 avg_sensitivity=round(avg_sens, 4),
                                 avg_f2=round(avg_f2, 4),
                                 avg_auc=round(avg_auc, 4)))
    print()

    df_summary = pd.DataFrame(summary_rows)
    write_excel_sheet(out_xlsx, "Summary",  df_summary)
    for ds in ALL_DATASETS:
        sub = df_all[df_all["dataset"] == ds].copy()
        write_excel_sheet(out_xlsx, ds.upper(), sub)

    print(f"  Results saved -> {out_xlsx}\n")


if __name__ == "__main__":
    main()
