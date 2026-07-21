"""Calibrates the two hand-set quality gates against real data.

Laplacian blur threshold (dashboard/api/main.py) and YOLO confidence
(src/detection/detector.py). Writes to outputs/quality_thresholds/.

  python scripts/no_seg/evaluate_quality_thresholds.py [--skip-yolo]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import pandas as pd

from src.detection.detector import LesionDetector
from src.utils.config import load_config
from src.utils.io import resolve_dataset_paths, write_excel_sheet

ALL_DATASETS = ["ham10000", "isic2019", "isic2020"]

SHARPNESS_THRESHOLD = 80.0            # dashboard/api/main.py:_QUALITY_THRESHOLD
YOLO_CONF_THRESHOLD = 0.35            # src/detection/detector.py default
BLUR_SIGMAS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
YOLO_CONF_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]

RNG = np.random.default_rng(42)


def get_paths():
    cfg = load_config()
    return (Path(cfg.paths.data_splits), Path(cfg.paths.melanoma_data), Path(cfg.paths.outputs))


def load_test_data(splits_dir: Path, melanoma_root: Path) -> dict[str, pd.DataFrame]:
    """Per-dataset test frames with binary_label (1 = melanoma)."""
    df = pd.read_csv(splits_dir / "cls_test.csv")
    df["binary_label"] = (df["label_str"] == "mel").astype(int)
    df = resolve_dataset_paths(df, melanoma_root)
    return {ds: df[df["dataset_source"] == ds].reset_index(drop=True) for ds in ALL_DATASETS}


def sample(paths: list[str], n: int) -> list[str]:
    if len(paths) <= n:
        return paths
    return list(RNG.choice(paths, n, replace=False))


def sample_df(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if len(df) <= n:
        return df
    return df.sample(n, random_state=42)


def lap_var(path: str) -> float | None:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def sharpness_distribution(test_data: dict[str, pd.DataFrame], n_per_ds: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-dataset sharpness stats plus a pooled long-form frame."""
    rows, pooled = [], []
    for ds in ALL_DATASETS:
        df = sample_df(test_data[ds], n_per_ds)
        lv = df["image_path"].map(lap_var)
        df = df.assign(lapvar=lv).dropna(subset=["lapvar"])
        pooled.append(df[["binary_label", "lapvar"]].assign(dataset=ds))

        vals = df["lapvar"].to_numpy()
        pct = np.percentile(vals, [0.5, 1, 5, 25, 50, 75, 95, 99])
        mel = df.loc[df["binary_label"] == 1, "lapvar"]
        rows.append(dict(
            dataset=ds, n=len(vals), n_mel=int(len(mel)),
            min=round(float(vals.min()), 1),
            p0_5=round(float(pct[0]), 1), p1=round(float(pct[1]), 1),
            p5=round(float(pct[2]), 1), p25=round(float(pct[3]), 1),
            median=round(float(pct[4]), 1), p75=round(float(pct[5]), 1),
            p95=round(float(pct[6]), 1), p99=round(float(pct[7]), 1),
            max=round(float(vals.max()), 1),
            frac_rejected_at_80=round(float((vals < SHARPNESS_THRESHOLD).mean()), 4),
            mel_median=round(float(mel.median()), 1) if len(mel) else None,
            mel_frac_rejected_at_80=round(float((mel < SHARPNESS_THRESHOLD).mean()), 4) if len(mel) else None,
        ))
    return pd.DataFrame(rows), pd.concat(pooled, ignore_index=True)


def sharpness_threshold_sweep(pooled: pd.DataFrame) -> pd.DataFrame:
    """Rejection rate by threshold, melanoma vs. non-melanoma kept separate."""
    grid = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40, 50, 60, 80, 100, 150, 200]
    lv = pooled["lapvar"].to_numpy()
    mel = pooled.loc[pooled["binary_label"] == 1, "lapvar"].to_numpy()
    nonmel = pooled.loc[pooled["binary_label"] == 0, "lapvar"].to_numpy()
    return pd.DataFrame([
        dict(
            threshold=t,
            frac_rejected_overall=round(float((lv < t).mean()), 4),
            frac_rejected_melanoma=round(float((mel < t).mean()), 4) if mel.size else None,
            frac_rejected_nonmelanoma=round(float((nonmel < t).mean()), 4) if nonmel.size else None,
            melanoma_pass_rate=round(float((mel >= t).mean()), 4) if mel.size else None,
        )
        for t in grid
    ])


def blur_injection(test_data: dict[str, pd.DataFrame], n: int) -> tuple[pd.DataFrame, dict[float, np.ndarray]]:
    """Blurs images at increasing sigma. Returns the summary and the raw per-sigma arrays."""
    pool = []
    for ds in ALL_DATASETS:
        pool += test_data[ds]["image_path"].tolist()
    pool = sample(pool, n)

    curves = {s: [] for s in BLUR_SIGMAS}
    for p in pool:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        for s in BLUR_SIGMAS:
            blurred = img if s == 0 else cv2.GaussianBlur(img, (0, 0), s)
            curves[s].append(float(cv2.Laplacian(blurred, cv2.CV_64F).var()))

    curves = {s: np.array(v) for s, v in curves.items()}
    rows = []
    for s in BLUR_SIGMAS:
        v = curves[s]
        rows.append(dict(
            gaussian_sigma=s,
            median_lapvar=round(float(np.median(v)), 1),
            mean_lapvar=round(float(v.mean()), 1),
            frac_rejected_at_80=round(float((v < SHARPNESS_THRESHOLD).mean()), 4),
        ))
    return pd.DataFrame(rows), curves


def blur_separation_calibration(
    curves: dict[float, np.ndarray],
    pooled: pd.DataFrame,
    check_sigmas=(0.5, 1.0),
    target_catch_rates=(0.90, 0.95, 0.99),
) -> pd.DataFrame:
    """Per blur level: the catching threshold, and its cost in sharp images and melanomas."""
    native = curves[0.0]
    mel = pooled.loc[pooled["binary_label"] == 1, "lapvar"].to_numpy()
    n_mel = int(mel.size)

    rows = []
    for sigma in check_sigmas:
        blurred = curves[sigma]
        for target in target_catch_rates:
            thr = float(np.percentile(blurred, target * 100))
            mel_rejected = int((mel < thr).sum()) if n_mel else None
            rows.append(dict(
                blur_sigma=sigma,
                target_blur_catch_rate=target,
                threshold=round(thr, 1),
                achieved_blur_catch_rate=round(float((blurred < thr).mean()), 4),
                native_wrongly_rejected=round(float((native < thr).mean()), 4),
                melanoma_pass_rate=round(float((mel >= thr).mean()), 4) if n_mel else None,
                melanoma_rejected=mel_rejected,
                n_melanoma=n_mel,
            ))
    return pd.DataFrame(rows)


def calibrate_sharpness_threshold(pooled: pd.DataFrame, pass_rates=(0.99, 0.95, 0.90)) -> dict:
    """Percentile threshold needed to pass a given fraction of all images."""
    lv = pooled["lapvar"].to_numpy()
    return {
        f"threshold_for_{int(pr * 100)}pct_pass": round(float(np.percentile(lv, (1 - pr) * 100)), 1)
        for pr in pass_rates
    }


def calibrate_sensitivity_first(pooled: pd.DataFrame, mel_pass_targets=(1.0, 0.995, 0.99, 0.95)) -> dict:
    """Highest threshold that still passes a given fraction of melanoma images."""
    mel = pooled.loc[pooled["binary_label"] == 1, "lapvar"].to_numpy()
    out: dict = {"n_melanoma": int(mel.size)}
    for target in mel_pass_targets:
        thr = float(np.percentile(mel, (1 - target) * 100)) if mel.size else float("nan")
        out[f"threshold_for_{target * 100:.1f}pct_melanoma_pass".replace(".0pct", "pct")] = round(thr, 1)
    return out


def mask_bbox(mask_path: Path) -> tuple[int, int, int, int] | None:
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    coords = np.argwhere(m > 127)
    if coords.size == 0:
        return None
    ys, xs = coords[:, 0], coords[:, 1]
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def iou(a: tuple, b: tuple) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def gather_gt_pairs(outputs_dir: Path, melanoma_root: Path, n: int) -> dict[str, list[tuple[str, tuple]]]:
    """(image, gt_bbox) pairs. ISIC2018 val is held out, HAM10000 is in-domain and optimistic."""
    det_dir = outputs_dir / "detection"
    isic_mask_dir = melanoma_root / "isic2018-challenge-task1-data-segmentation" / "versions" / "1" / "ISIC2018_Task1_Training_GroundTruth"
    ham_mask_dir = melanoma_root / "HAM10000_segmentations_lesion_tschandl"

    pairs: dict[str, list[tuple[str, tuple]]] = {"isic2018_val": [], "ham10000": []}

    val_txt = det_dir / "val_resolved.txt"
    if val_txt.exists():
        val_imgs = [l.strip() for l in val_txt.read_text().splitlines() if l.strip()]
        val_imgs = sample(val_imgs, n)
        for p in val_imgs:
            gt = mask_bbox(isic_mask_dir / f"{Path(p).stem}_segmentation.png")
            if gt is not None:
                pairs["isic2018_val"].append((p, gt))

    train_csv = Path(load_config().paths.data_splits) / "cls_test.csv"
    if train_csv.exists():
        df = pd.read_csv(train_csv)
        df = df[df["dataset_source"] == "ham10000"].copy()
        df = resolve_dataset_paths(df, melanoma_root)
        if "mask_path" in df.columns:
            df = df[df["mask_path"].notna()]
        rows = df.sample(min(n, len(df)), random_state=42).to_dict("records") if len(df) else []
        for row in rows:
            mp = row.get("mask_path")
            if not mp:
                mp = ham_mask_dir / f"{Path(row['image_path']).stem}_segmentation.png"
            gt = mask_bbox(Path(mp))
            if gt is not None:
                pairs["ham10000"].append((row["image_path"], gt))

    return pairs


def yolo_confidence_sweep(pairs: list[tuple[str, tuple]], ckpt: Path) -> tuple[pd.DataFrame, dict]:
    """One YOLO pass per image, then the LesionDetector decision rule per threshold."""
    from ultralytics import YOLO
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(str(ckpt))
    model.to(device)

    validator = LesionDetector()  # unloaded, only for its bbox-validity rule

    records = []
    for path, gt in pairs:
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        res = model.predict(source=img, imgsz=640, conf=0.05, device=device, verbose=False, save=False)
        boxes = res[0].boxes
        if boxes is None or len(boxes) == 0:
            records.append((0.0, None, gt, h, w))
            continue
        confs = boxes.conf.cpu().numpy()
        j = int(confs.argmax())
        xyxy = boxes.xyxy[j].cpu().numpy()
        x1, y1 = max(0, int(xyxy[0])), max(0, int(xyxy[1]))
        x2, y2 = min(w, int(xyxy[2])), min(h, int(xyxy[3]))
        records.append((float(confs[j]), (x1, y1, x2, y2), gt, h, w))

    rows = []
    for thr in YOLO_CONF_GRID:
        n = len(records)
        accepted = 0
        iou_vals = []
        for conf, box, gt, h, w in records:
            if box is None or conf < thr:
                continue
            x1, y1, x2, y2 = box
            if not validator._is_valid_bbox(x1, y1, x2, y2, h, w):
                continue
            accepted += 1
            iou_vals.append(iou(box, gt))
        rows.append(dict(
            conf_threshold=thr,
            detection_rate=round(accepted / n, 4) if n else 0.0,
            fallback_rate=round(1 - accepted / n, 4) if n else 1.0,
            mean_iou_accepted=round(float(np.mean(iou_vals)), 4) if iou_vals else 0.0,
            median_iou_accepted=round(float(np.median(iou_vals)), 4) if iou_vals else 0.0,
            recall_iou_ge_0_5=round(float(np.sum(np.array(iou_vals) >= 0.5) / n), 4) if n else 0.0,
            recall_iou_ge_0_75=round(float(np.sum(np.array(iou_vals) >= 0.75) / n), 4) if n else 0.0,
        ))

    conf_all = np.array([c for c, b, *_ in records if b is not None])
    dist = dict(
        n=len(records), n_with_any_box=int(len(conf_all)),
        conf_median=round(float(np.median(conf_all)), 4) if conf_all.size else None,
        conf_p10=round(float(np.percentile(conf_all, 10)), 4) if conf_all.size else None,
        conf_p90=round(float(np.percentile(conf_all, 90)), 4) if conf_all.size else None,
    )
    return pd.DataFrame(rows), dist


def calibrate_yolo_threshold(sweep: pd.DataFrame, target_detection=(0.99, 0.95)) -> dict:
    out = {}
    for target in target_detection:
        ok = sweep[sweep["detection_rate"] >= target]
        rec = float(ok["conf_threshold"].max()) if not ok.empty else float(sweep["conf_threshold"].min())
        out[f"threshold_for_{int(target * 100)}pct_detection"] = rec
    return out


def print_recommendation(sharp_calib: dict, sens_calib: dict, sep_df: pd.DataFrame, sharp_sweep_at_80, blur_df, yolo_calib: dict | None, yolo_sweep: pd.DataFrame | None):
    print("\n" + "=" * 78)
    print("RECOMMENDATION")
    print("=" * 78)
    print(f"\nSharpness gate (shipped: Laplacian variance < {SHARPNESS_THRESHOLD:.0f}):")
    print(f"  On this benchmark population, threshold=80 rejects "
          f"{sharp_sweep_at_80:.1%} of real clinical test images.")
    print(f"  Zero-cost ceiling (never reject a melanoma image on this data): "
          f"{sens_calib['threshold_for_100pct_melanoma_pass']} "
          f"-- but this barely rejects anything (see [A2]), so it doesn't actually filter blur.")
    print(f"  Blur-separation trade-off ([A6]): the threshold needed to reliably catch real blur")
    print(f"  costs melanoma cases. Pick the row matching how much blur you want to catch:")
    print(sep_df.to_string(index=False))
    print("  -> 80 is calibrated for high-resolution live-camera frames, not these")
    print("     (compressed, resized) benchmark JPEGs; it is resolution-dependent")
    print("     and must be recalibrated per acquisition device before deployment.")
    rising = blur_df["frac_rejected_at_80"].is_monotonic_increasing
    print(f"  Blur-injection monotonicity check: {'PASS' if rising else 'FAIL'} "
          f"(rejection rate rises monotonically with injected blur = {list(blur_df['frac_rejected_at_80'])})")

    print(f"\nDetection gate (shipped: YOLO confidence < {YOLO_CONF_THRESHOLD}):")
    if yolo_calib is not None:
        print(f"  Highest threshold keeping detection-rate >= 99%: {yolo_calib['threshold_for_99pct_detection']}")
        print(f"  Highest threshold keeping detection-rate >= 95%: {yolo_calib['threshold_for_95pct_detection']}")
        row = yolo_sweep[yolo_sweep["conf_threshold"] == YOLO_CONF_THRESHOLD].iloc[0]
        print(f"  At 0.35 (held-out ISIC2018): detection_rate={row['detection_rate']:.3f}, "
              f"mean_IoU={row['mean_iou_accepted']:.3f}, recall@IoU0.5={row['recall_iou_ge_0_5']:.3f}")
    else:
        print("  (skipped, --skip-yolo)")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-sharpness", type=int, default=5000, help="images per dataset for the sharpness distribution (default covers full test sets)")
    ap.add_argument("--n-blur", type=int, default=800, help="images for the blur-injection sweep")
    ap.add_argument("--n-yolo", type=int, default=800, help="images per mask population for the YOLO sweep")
    ap.add_argument("--skip-yolo", action="store_true", help="skip the GPU-bound YOLO confidence sweep")
    args = ap.parse_args()

    splits_dir, melanoma_root, outputs_dir = get_paths()
    out_dir = outputs_dir / "quality_thresholds"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Quality-threshold calibration check")
    print(f"  Sharpness gate : Laplacian variance < {SHARPNESS_THRESHOLD}  (dashboard/api/main.py)")
    print(f"  Detection gate : YOLO confidence < {YOLO_CONF_THRESHOLD}  (src/detection/detector.py)")
    print("=" * 78)

    test_data = load_test_data(splits_dir, melanoma_root)

    print("\n[A1] Laplacian sharpness distribution (native resolution, per dataset, melanoma vs. overall):")
    sharp_df, pooled = sharpness_distribution(test_data, args.n_sharpness)
    print(sharp_df.to_string(index=False))
    overall_rejected_80 = float((pooled["lapvar"] < SHARPNESS_THRESHOLD).mean())

    print("\n[A2] Rejection rate across candidate thresholds (melanoma vs. non-melanoma):")
    sweep_df = sharpness_threshold_sweep(pooled)
    print(sweep_df.to_string(index=False))

    print(f"\n[A3] Blur-injection sweep (does rejection rate rise monotonically with blur?):")
    blur_df, blur_curves = blur_injection(test_data, args.n_blur)
    print(blur_df.to_string(index=False))

    sharp_calib = calibrate_sharpness_threshold(pooled)
    print(f"\n[A4] Population-level percentile calibration (label-blind): {sharp_calib}")

    sens_calib = calibrate_sensitivity_first(pooled)
    print(f"\n[A5] Sensitivity-first calibration (zero-cost ceiling, protects melanoma recall): {sens_calib}")

    print(f"\n[A6] Blur-separation trade-off (threshold that actually catches injected blur, vs. its melanoma cost):")
    sep_df = blur_separation_calibration(blur_curves, pooled)
    print(sep_df.to_string(index=False))

    yolo_sweep_df = None
    yolo_calib = None
    yolo_dist = {}
    yolo_pairs_n = {}
    if not args.skip_yolo:
        ckpt = outputs_dir / "detection" / "checkpoints" / "best.pt"
        if not ckpt.exists():
            print(f"\n[B] SKIPPED: no trained YOLO checkpoint at {ckpt}")
        else:
            pairs = gather_gt_pairs(outputs_dir, melanoma_root, args.n_yolo)
            yolo_pairs_n = {k: len(v) for k, v in pairs.items()}
            print(f"\n[B] YOLO confidence sweep - held-out ISIC2018 val (n={yolo_pairs_n.get('isic2018_val', 0)}, "
                  f"unbiased) and HAM10000 (n={yolo_pairs_n.get('ham10000', 0)}, in-domain/optimistic):")

            print("\n  ISIC2018 held-out val (unbiased)")
            iou_sweep_isic, dist_isic = yolo_confidence_sweep(pairs["isic2018_val"], ckpt)
            print(iou_sweep_isic.to_string(index=False))
            print("  best-box confidence distribution:", dist_isic)

            print("\n  HAM10000 (in-domain, detector trained on this source)")
            iou_sweep_ham, dist_ham = yolo_confidence_sweep(pairs["ham10000"], ckpt)
            print(iou_sweep_ham.to_string(index=False))
            print("  best-box confidence distribution:", dist_ham)

            yolo_sweep_df = iou_sweep_isic
            yolo_calib = calibrate_yolo_threshold(iou_sweep_isic)
            yolo_dist = dict(isic2018_val=dist_isic, ham10000=dist_ham)

    print_recommendation(sharp_calib, sens_calib, sep_df, overall_rejected_80, blur_df, yolo_calib, yolo_sweep_df)

    report = dict(
        sharpness_threshold_shipped=SHARPNESS_THRESHOLD,
        yolo_conf_threshold_shipped=YOLO_CONF_THRESHOLD,
        sharpness=dict(
            per_dataset=sharp_df.to_dict("records"),
            overall_frac_rejected_at_80=round(overall_rejected_80, 4),
            threshold_sweep=sweep_df.to_dict("records"),
            blur_injection=blur_df.to_dict("records"),
            blur_monotonic=bool(blur_df["frac_rejected_at_80"].is_monotonic_increasing),
            population_calibration=sharp_calib,
            sensitivity_first_calibration=sens_calib,
            blur_separation_tradeoff=sep_df.to_dict("records"),
        ),
        yolo=dict(
            n_pairs=yolo_pairs_n,
            confidence_sweep_isic2018_val=(yolo_sweep_df.to_dict("records") if yolo_sweep_df is not None else None),
            confidence_distribution=yolo_dist,
            calibrated_recommendation=yolo_calib,
        ) if not args.skip_yolo else None,
    )
    json_path = out_dir / "evaluate_quality_thresholds.json"
    json_path.write_text(json.dumps(report, indent=2, default=str))

    xlsx_path = out_dir / "evaluate_quality_thresholds.xlsx"
    write_excel_sheet(xlsx_path, "Laplacian_By_Dataset", sharp_df)
    write_excel_sheet(xlsx_path, "Laplacian_Threshold_Sweep", sweep_df)
    write_excel_sheet(xlsx_path, "Laplacian_Blur_Injection", blur_df)
    write_excel_sheet(xlsx_path, "Laplacian_Blur_Separation", sep_df)
    if yolo_sweep_df is not None:
        write_excel_sheet(xlsx_path, "YOLO_Conf_Sweep_ISIC2018", yolo_sweep_df)
        write_excel_sheet(xlsx_path, "YOLO_Conf_Sweep_HAM10000", iou_sweep_ham)

    print(f"Saved -> {json_path}")
    print(f"Saved -> {xlsx_path}")


if __name__ == "__main__":
    main()
