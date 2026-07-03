from __future__ import annotations

import itertools
import pickle
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

ALL_DATASETS = ["ham10000", "isic2019", "isic2020"]
ALL_MODELS   = [
    "resnet50", "efficientnet_b2", "mobilenetv3_large", "convnext_tiny_se",
    "medfusionnet", "yolov8_cls",
]

AUG_MODE  = "none_sens"
BETA      = 2.0
THR_RANGE = np.round(np.arange(0.20, 0.86, 0.01), 2)

# set to None to try ALL triplets; or list specific ones to focus on
FOCUS_TRIPLETS: list[tuple] | None = None

_MEAN      = (0.485, 0.456, 0.406)
_STD       = (0.229, 0.224, 0.225)
_SITE_CATS = [
    "head/neck", "upper extremity", "lower extremity",
    "torso", "palms/soles", "oral/genital",
]


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


def _load_df(splits_dir: Path, csv_name: str, melanoma_root: Path,
            dataset_source: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(splits_dir / csv_name)
    if dataset_source:
        df = df[df["dataset_source"] == dataset_source].copy()
    df["binary_label"] = (df["label_str"] == "mel").astype(int)
    df = resolve_dataset_paths(df, melanoma_root)
    return df


def _run_inference(
    model: nn.Module, loader: DataLoader,
    device: torch.device, with_meta: bool,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="    infer", leave=False,
                          unit="batch", dynamic_ncols=True, file=sys.stdout):
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
    inp_size = getattr(config, "input_size", 224)
    nw       = getattr(config, "num_workers", 0)
    ds       = EvalDataset(df, inp_size, with_meta=with_meta)
    return DataLoader(ds, batch_size=config.batch_size, shuffle=False,
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
    acc  = (tp + tn) / max(tp + tn + fp + fn, 1)
    return dict(sensitivity=round(sens, 4), specificity=round(spec, 4),
                precision=round(prec, 4), f2=round(f2, 4), f1=round(f1, 4),
                accuracy=round(acc, 4), tp=tp, tn=tn, fp=fp, fn=fn)


def _majority_vote(p1, p2, p3, thr1, thr2, thr3) -> np.ndarray:
    votes = (p1 >= thr1).astype(int) + (p2 >= thr2).astype(int) + (p3 >= thr3).astype(int)
    return (votes >= 2).astype(int)


def _meta_thr_sweep(meta_probs: np.ndarray, labels: np.ndarray) -> float:
    best_thr, best = THR_RANGE[0], -1.0
    for thr in THR_RANGE:
        f2 = _metrics(meta_probs, labels, thr)["f2"]
        if f2 > best:
            best, best_thr = f2, thr
    return float(best_thr)


def _save_progress(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def main() -> None:
    base_cfg     = load_config()
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits_dir   = Path(base_cfg.paths.data_splits)
    melanoma_root = Path(base_cfg.paths.melanoma_data)
    ablation_dir = Path(base_cfg.paths.outputs) / "ablation_noseg"
    out_xlsx     = ablation_dir / "meta" / "evaluation_meta_stacking.xlsx"
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    _detail_csv  = ablation_dir / "meta" / "_stacking_progress_detail.csv"
    _summary_csv = ablation_dir / "meta" / "_stacking_progress_summary.csv"
    _probs_cache = ablation_dir / "meta" / "_stacking_probs_cache.pkl"

    gpu_label = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"\n  Meta-learner stacking — {AUG_MODE} checkpoints")
    print(f"  Device : {gpu_label}\n")

    val_dfs:  dict[str, pd.DataFrame] = {}
    test_dfs: dict[str, pd.DataFrame] = {}
    for ds in ALL_DATASETS:
        try:
            val_dfs[ds]  = _load_df(splits_dir, "cls_val.csv",  melanoma_root, ds)
            test_dfs[ds] = _load_df(splits_dir, "cls_test.csv", melanoma_root, ds)
            print(f"  {ds.upper():<12}  val={len(val_dfs[ds])}  test={len(test_dfs[ds])}"
                  f"  mel_val={int(val_dfs[ds]['binary_label'].sum())}"
                  f"  mel_test={int(test_dfs[ds]['binary_label'].sum())}")
        except Exception as exc:
            print(f"  [SKIP] {ds}: {exc}")
    print()

    val_labels  = {ds: val_dfs[ds]["binary_label"].values  for ds in ALL_DATASETS if ds in val_dfs}
    test_labels = {ds: test_dfs[ds]["binary_label"].values for ds in ALL_DATASETS if ds in test_dfs}

    if FOCUS_TRIPLETS:
        triplets = FOCUS_TRIPLETS
    else:
        candidates = [(m, ds) for ds in ALL_DATASETS for m in ALL_MODELS]
        triplets = [
            (m1, d1, m2, d2, m3, d3)
            for (m1, d1), (m2, d2), (m3, d3)
            in itertools.combinations(candidates, 3)
        ]
    print(f"  Total triplets : {len(triplets)}\n")

    if _probs_cache.exists():
        print("  [CACHE] loading precomputed probabilities from disk ...")
        with open(_probs_cache, "rb") as f:
            _c = pickle.load(f)
        val_probs:  dict[str, dict[str, np.ndarray]] = _c["val_probs"]
        test_probs: dict[str, dict[str, np.ndarray]] = _c["test_probs"]
        model_thrs: dict[str, float]                 = _c["model_thrs"]
        print(f"  [CACHE] {len(val_probs)} candidates loaded — skipping GPU inference.\n")
    else:
        all_candidates = [(m, ds) for ds in ALL_DATASETS for m in ALL_MODELS]
        val_probs:  dict[str, dict[str, np.ndarray]] = {}
        test_probs: dict[str, dict[str, np.ndarray]] = {}
        model_thrs: dict[str, float]                 = {}

        print(f"  Precomputing probabilities for {len(all_candidates)} candidates ...\n")
        for ci, (mn, dn) in enumerate(all_candidates, 1):
            key    = f"{mn}/{dn}"
            run_id = f"{mn}_{AUG_MODE}"
            ckpt_f = ablation_dir / dn / run_id / "checkpoints" / f"{run_id}.pt"
            if not ckpt_f.exists():
                print(f"  [{ci}/{len(all_candidates)}] [SKIP] missing checkpoint: {key}")
                continue

            print(f"  [{ci}/{len(all_candidates)}] {key}")
            config = load_config(mn)
            wm     = uses_metadata(mn)
            model  = build_model(mn, config, num_classes=2)
            model.load_state_dict(torch.load(ckpt_f, map_location=device))
            model.to(device).eval().requires_grad_(False)

            vp: dict[str, np.ndarray] = {}
            all_vp, all_vl = [], []
            for ds in ALL_DATASETS:
                if ds not in val_dfs:
                    continue
                loader = _get_loader(val_dfs[ds], mn, config, wm)
                p, l   = _run_inference(model, loader, device, wm)
                vp[ds] = p
                all_vp.append(p)
                all_vl.append(l)
            val_probs[key]  = vp
            model_thrs[key] = _meta_thr_sweep(np.concatenate(all_vp), np.concatenate(all_vl))

            tp: dict[str, np.ndarray] = {}
            for ds in ALL_DATASETS:
                if ds not in test_dfs:
                    continue
                loader = _get_loader(test_dfs[ds], mn, config, wm)
                p, _   = _run_inference(model, loader, device, wm)
                tp[ds] = p
            test_probs[key] = tp

            del model
            torch.cuda.empty_cache()

        with open(_probs_cache, "wb") as f:
            pickle.dump({"val_probs": val_probs, "test_probs": test_probs, "model_thrs": model_thrs}, f)
        print(f"\n  [CACHE] saved → {_probs_cache.name}  (GPU no longer needed)\n")

    meta_rows:     list[dict] = []
    majority_rows: list[dict] = []
    detail_rows:   list[dict] = []

    # resume: reload previously completed triplets
    done_triplets: set[str] = set()
    if _detail_csv.exists() and _summary_csv.exists():
        try:
            prev_detail   = pd.read_csv(_detail_csv)
            prev_summary  = pd.read_csv(_summary_csv)
            done_triplets = set(prev_detail["triplet"].unique())
            detail_rows   = prev_detail.to_dict("records")
            meta_rows     = prev_summary[prev_summary["mode"] == "meta"].to_dict("records")
            majority_rows = prev_summary[prev_summary["mode"] == "majority"].to_dict("records")
            print(f"  [RESUME] {len(done_triplets)} triplets already done — skipping.\n")
        except Exception as exc:
            print(f"  [WARN] could not load progress: {exc} — starting fresh.\n")

    for tri_idx, (m1, d1, m2, d2, m3, d3) in enumerate(triplets, 1):
        label = f"{m1}/{d1} + {m2}/{d2} + {m3}/{d3}"

        if label in done_triplets:
            print(f"  [{tri_idx}/{len(triplets)}] [SKIP] {label}")
            continue

        k1, k2, k3 = f"{m1}/{d1}", f"{m2}/{d2}", f"{m3}/{d3}"
        missing = [k for k in (k1, k2, k3) if k not in val_probs]
        if missing:
            print(f"  [{tri_idx}/{len(triplets)}] [SKIP] missing probs: {missing}")
            continue

        print(f"  [{tri_idx}/{len(triplets)}] {label}")

        X_parts, y_parts = [], []
        for ds in ALL_DATASETS:
            if ds not in val_labels:
                continue
            p1 = val_probs[k1].get(ds)
            p2 = val_probs[k2].get(ds)
            p3 = val_probs[k3].get(ds)
            if p1 is None or p2 is None or p3 is None:
                continue
            X_parts.append(np.column_stack([p1, p2, p3]))
            y_parts.append(val_labels[ds])

        X_meta   = np.vstack(X_parts)
        y_meta   = np.concatenate(y_parts)
        scaler   = StandardScaler()
        X_meta_s = scaler.fit_transform(X_meta)
        clf      = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced")
        clf.fit(X_meta_s, y_meta)
        print(f"    meta weights: {clf.coef_[0].round(3)}  bias={clf.intercept_[0]:.3f}")

        tri_detail_rows: list[dict] = []
        meta_sens, meta_f2 = [], []
        maj_sens,  maj_f2  = [], []

        for ds in ALL_DATASETS:
            if ds not in test_labels:
                continue
            labels = test_labels[ds]
            p1 = test_probs[k1].get(ds)
            p2 = test_probs[k2].get(ds)
            p3 = test_probs[k3].get(ds)
            if p1 is None or p2 is None or p3 is None:
                continue

            X_test    = np.column_stack([p1, p2, p3])
            X_test_s  = scaler.transform(X_test)
            meta_prob = clf.predict_proba(X_test_s)[:, 1]
            meta_thr  = _meta_thr_sweep(meta_prob, labels)
            m_meta    = _metrics(meta_prob, labels, meta_thr)
            try:
                meta_auc = float(roc_auc_score(labels, meta_prob))
            except Exception:
                meta_auc = float("nan")

            thr1, thr2, thr3 = model_thrs[k1], model_thrs[k2], model_thrs[k3]
            maj_preds = _majority_vote(p1, p2, p3, thr1, thr2, thr3)
            maj_prob  = (p1 + p2 + p3) / 3
            tp = int(((maj_preds == 1) & (labels == 1)).sum())
            tn = int(((maj_preds == 0) & (labels == 0)).sum())
            fp = int(((maj_preds == 1) & (labels == 0)).sum())
            fn = int(((maj_preds == 0) & (labels == 1)).sum())
            sens_maj = tp / max(tp + fn, 1)
            spec_maj = tn / max(tn + fp, 1)
            prec_maj = tp / max(tp + fp, 1)
            b2 = BETA ** 2
            f2_maj = (1 + b2) * prec_maj * sens_maj / max(b2 * prec_maj + sens_maj, 1e-9)
            try:
                maj_auc = float(roc_auc_score(labels, maj_prob))
            except Exception:
                maj_auc = float("nan")

            meta_sens.append(m_meta["sensitivity"])
            meta_f2.append(m_meta["f2"])
            maj_sens.append(sens_maj)
            maj_f2.append(f2_maj)

            tri_detail_rows.append(dict(
                triplet=label, eval_dataset=ds,
                meta_sensitivity=m_meta["sensitivity"], meta_specificity=m_meta["specificity"],
                meta_f2=m_meta["f2"], meta_f1=m_meta["f1"], meta_auc=round(meta_auc, 4),
                meta_threshold=meta_thr,
                maj_sensitivity=round(sens_maj, 4), maj_specificity=round(spec_maj, 4),
                maj_f2=round(f2_maj, 4), maj_auc=round(maj_auc, 4),
                model1=f"{m1}/{d1}", model2=f"{m2}/{d2}", model3=f"{m3}/{d3}",
            ))
            print(f"    {ds.upper():<12}  meta: sens={m_meta['sensitivity']:.3f} f2={m_meta['f2']:.3f}"
                  f"  | majority: sens={sens_maj:.3f} f2={f2_maj:.3f}")

        def _summary(label, m1, d1, m2, d2, m3, d3, sens_list, f2_list, mode):
            ham = sens_list[0] if len(sens_list) > 0 else 0.0
            i19 = sens_list[1] if len(sens_list) > 1 else 0.0
            i20 = sens_list[2] if len(sens_list) > 2 else 0.0
            return dict(
                triplet=label, mode=mode,
                model1=f"{m1}/{d1}", model2=f"{m2}/{d2}", model3=f"{m3}/{d3}",
                sensitivity_ham10000=ham, sensitivity_isic2019=i19, sensitivity_isic2020=i20,
                f2_ham10000=f2_list[0] if len(f2_list) > 0 else 0.0,
                f2_isic2019=f2_list[1] if len(f2_list) > 1 else 0.0,
                f2_isic2020=f2_list[2] if len(f2_list) > 2 else 0.0,
                min_sensitivity=round(float(np.min(sens_list)), 4) if sens_list else 0.0,
                mean_sens_no2020=round((ham + i19) / 2, 4),
                avg_sensitivity=round(float(np.mean(sens_list)), 4) if sens_list else 0.0,
                avg_f2=round(float(np.mean(f2_list)), 4) if f2_list else 0.0,
            )

        tri_meta_row     = _summary(label, m1, d1, m2, d2, m3, d3, meta_sens, meta_f2, "meta")
        tri_majority_row = _summary(label, m1, d1, m2, d2, m3, d3, maj_sens, maj_f2, "majority")

        detail_rows.extend(tri_detail_rows)
        meta_rows.append(tri_meta_row)
        majority_rows.append(tri_majority_row)

        _save_progress(tri_detail_rows, _detail_csv)
        _save_progress([tri_meta_row, tri_majority_row], _summary_csv)

    if not meta_rows:
        print("\n  No results.\n")
        return

    df_meta = pd.DataFrame(meta_rows).sort_values("min_sensitivity", ascending=False)
    df_maj  = pd.DataFrame(majority_rows).sort_values("min_sensitivity", ascending=False)
    df_det  = pd.DataFrame(detail_rows)

    comp_cols = ["triplet", "min_sensitivity", "mean_sens_no2020", "avg_f2"]
    df_comp = df_meta[comp_cols].copy().rename(columns={
        "min_sensitivity":  "meta_min_sens",
        "mean_sens_no2020": "meta_mean_sens_no2020",
        "avg_f2":           "meta_avg_f2",
    })
    df_comp["maj_min_sens"]           = df_maj["min_sensitivity"].values
    df_comp["maj_mean_sens_no2020"]   = df_maj["mean_sens_no2020"].values
    df_comp["maj_avg_f2"]             = df_maj["avg_f2"].values
    df_comp["delta_min_sens"]         = (df_comp["meta_min_sens"] - df_comp["maj_min_sens"]).round(4)
    df_comp["delta_mean_sens_no2020"] = (df_comp["meta_mean_sens_no2020"] - df_comp["maj_mean_sens_no2020"]).round(4)
    df_comp = df_comp.sort_values("delta_min_sens", ascending=False)

    write_excel_sheet(out_xlsx, "Summary_Meta",     df_meta)
    write_excel_sheet(out_xlsx, "Summary_Majority", df_maj)
    write_excel_sheet(out_xlsx, "Comparison",       df_comp)
    for ds in ALL_DATASETS:
        sub = df_det[df_det["eval_dataset"] == ds].copy()
        if not sub.empty:
            write_excel_sheet(out_xlsx, ds.upper(), sub.sort_values("meta_sensitivity", ascending=False))

    for _f in (_detail_csv, _summary_csv, _probs_cache):
        try:
            _f.unlink(missing_ok=True)
        except Exception:
            pass

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  Results : {out_xlsx}")
    print(f"  Sheets  : Summary_Meta | Summary_Majority | Comparison | HAM10000 | ISIC2019 | ISIC2020")
    print(f"{sep}\n")

    print("  Top results by min_sensitivity (meta-learner):\n")
    top_cols = ["triplet", "sensitivity_ham10000", "sensitivity_isic2019",
                "sensitivity_isic2020", "min_sensitivity", "mean_sens_no2020"]
    print(df_meta[top_cols].head(5).to_string(index=False))
    print("\n  Delta vs majority vote (positive = meta better):\n")
    print(df_comp[["triplet", "delta_min_sens", "delta_mean_sens_no2020"]].head(5).to_string(index=False))
    print()


if __name__ == "__main__":
    main()
