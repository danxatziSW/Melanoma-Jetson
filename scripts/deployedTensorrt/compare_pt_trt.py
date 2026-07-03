"""Compares PyTorch CPU vs TRT FP16 vs TRT FP32 deployment pipelines and plots the results.

Usage: python3 scripts/deployedTensorrt/compare_pt_trt.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

_ROOT      = Path(__file__).resolve().parents[3]
_DEPLOY    = _ROOT / "outputs" / "ablation_noseg" / "meta" / "deployment"
_PLOT_DIR  = _DEPLOY / "JetsonPT" / "plots" / "comparison"
_PLOT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.size":         11,
})

_COL_PT   = "#9467BD"   # purple
_COL_F16  = "#4C72B0"   # blue
_COL_F32  = "#DD8452"   # orange
_COL_RED  = "#C44E52"

_DS_ORDER = ["ham10000", "isic2019", "isic2020"]
_DS_LABEL = {"ham10000": "HAM10000", "isic2019": "ISIC-2019", "isic2020": "ISIC-2020"}

_BACKENDS = [
    ("PT CPU",   _COL_PT,  _DEPLOY / "JetsonPT" / "evaluation_pt.xlsx",       "Results"),
    ("TRT FP16", _COL_F16, _DEPLOY / "tensorrt"  / "evaluation_trt_fp16.xlsx", "Results"),
    ("TRT FP32", _COL_F32, _DEPLOY / "tensorrt"  / "evaluation_trt_fp32.xlsx", "Results"),
]

_LAT_SRC = [
    ("PT CPU",   _COL_PT,  _DEPLOY / "JetsonPT" / "evaluation_pt.xlsx",       "Latency"),
    ("TRT FP16", _COL_F16, _DEPLOY / "tensorrt"  / "evaluation_trt_fp16.xlsx", "Latency"),
    ("TRT FP32", _COL_F32, _DEPLOY / "tensorrt"  / "evaluation_trt_fp32.xlsx", "Latency"),
]


def _load_results() -> list[tuple[str, str, pd.DataFrame]]:
    out = []
    for label, col, path, sheet in _BACKENDS:
        if not path.exists():
            print(f"  [WARN] missing: {path.name} — skipping {label}")
            continue
        xl = pd.ExcelFile(path)
        if sheet not in xl.sheet_names:
            print(f"  [WARN] no '{sheet}' sheet in {path.name} — skipping {label}")
            continue
        df = xl.parse(sheet)
        out.append((label, col, df))
    return out


def _load_latency() -> list[tuple[str, str, pd.DataFrame]]:
    out = []
    for label, col, path, sheet in _LAT_SRC:
        if not path.exists():
            continue
        xl = pd.ExcelFile(path)
        if sheet not in xl.sheet_names:
            continue
        df = xl.parse(sheet).set_index("component")
        out.append((label, col, df))
    return out


def _gcol(df: pd.DataFrame, col: str) -> list[float]:
    return [float(df[df["dataset"] == d][col].iloc[0])
            if (df["dataset"] == d).any() else 0.0
            for d in _DS_ORDER]


def _save(fig: plt.Figure, name: str) -> None:
    p = _PLOT_DIR / name
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {p.relative_to(_ROOT)}")


backends = _load_results()
lat_data  = _load_latency()

if not backends and not lat_data:
    print("No data found. Run evaluate_pt.py first.")
    raise SystemExit(1)

xs  = np.arange(len(_DS_ORDER))
ds_labels = [_DS_LABEL[d] for d in _DS_ORDER]
n   = len(backends)
W   = 0.22
offsets = np.linspace(-(n-1)*W/2, (n-1)*W/2, n)

leg_patches = [mpatches.Patch(color=col, label=lbl) for lbl, col, _ in backends]


if backends:
    fig, ax = plt.subplots(figsize=(11, 5))
    for (lbl, col, df), off in zip(backends, offsets):
        vals = _gcol(df, "auc_meta")
        bars = ax.bar(xs + off, vals, W, label=lbl, color=col, zorder=3)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.003,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(xs); ax.set_xticklabels(ds_labels)
    ax.set_ylim(0.8, 1.06)
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Meta-Learner AUC-ROC: PT CPU vs TRT FP16 vs TRT FP32")
    ax.legend(handles=leg_patches, fontsize=10)
    ax.grid(axis="y", alpha=0.35)
    _save(fig, "01_auc_comparison.png")

if backends:
    fig, ax = plt.subplots(figsize=(11, 5))
    for (lbl, col, df), off in zip(backends, offsets):
        vals = _gcol(df, "deployed_sensitivity")
        bars = ax.bar(xs + off, vals, W, label=lbl, color=col, zorder=3)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.004,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7)
    ax.axhline(0.85, color=_COL_RED, linestyle="--", lw=1.2, label="Target >=0.85")
    ax.set_xticks(xs); ax.set_xticklabels(ds_labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Sensitivity @ deployed threshold")
    ax.set_title("Sensitivity: PT CPU vs TRT FP16 vs TRT FP32")
    ax.legend(handles=leg_patches + [mpatches.Patch(color=_COL_RED, label="Target >=0.85")],
              fontsize=10)
    ax.grid(axis="y", alpha=0.35)
    _save(fig, "02_sensitivity_comparison.png")

if backends:
    fig, ax = plt.subplots(figsize=(11, 5))
    for (lbl, col, df), off in zip(backends, offsets):
        vals = _gcol(df, "deployed_specificity")
        bars = ax.bar(xs + off, vals, W, label=lbl, color=col, zorder=3)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.004,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(xs); ax.set_xticklabels(ds_labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Specificity @ deployed threshold")
    ax.set_title("Specificity: PT CPU vs TRT FP16 vs TRT FP32")
    ax.legend(handles=leg_patches, fontsize=10)
    ax.grid(axis="y", alpha=0.35)
    _save(fig, "03_specificity_comparison.png")

if backends:
    fig, ax = plt.subplots(figsize=(11, 5))
    for (lbl, col, df), off in zip(backends, offsets):
        vals = _gcol(df, "deployed_f2")
        bars = ax.bar(xs + off, vals, W, label=lbl, color=col, zorder=3)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.004,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(xs); ax.set_xticklabels(ds_labels)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("F2 Score @ deployed threshold")
    ax.set_title("F2 Score: PT CPU vs TRT FP16 vs TRT FP32")
    ax.legend(handles=leg_patches, fontsize=10)
    ax.grid(axis="y", alpha=0.35)
    _save(fig, "04_f2_comparison.png")

if lat_data:
    components  = ["resnet50", "medfusionnet", "meta_learner"]
    comp_labels = ["ResNet-50", "MedFusionNet", "Meta-Learner"]
    comp_cols   = ["#4C72B0", "#DD8452", "#55A868"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # left: stacked bars per backend
    ax = axes[0]
    x2 = np.arange(len(lat_data))
    bottoms = np.zeros(len(lat_data))
    for comp, label, col in zip(components, comp_labels, comp_cols):
        vals = np.array([float(df.loc[comp, "median"]) if comp in df.index else 0
                         for _, _, df in lat_data])
        ax.bar(x2, vals, 0.5, bottom=bottoms, label=label, color=col, zorder=3)
        for i, (val, bot) in enumerate(zip(vals, bottoms)):
            if val > 5:
                ax.text(x2[i], bot + val/2, f"{val:.1f}ms",
                        ha="center", va="center", fontsize=8,
                        color="white", fontweight="bold")
        bottoms += vals
    for i, (lbl, _, df) in enumerate(lat_data):
        fps = float(df.loc["fps_total", "median"]) if "fps_total" in df.index else 0
        ax.text(x2[i], bottoms[i]+2, f"{bottoms[i]:.1f}ms\n({fps:.0f} FPS)",
                ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(x2)
    ax.set_xticklabels([lbl for lbl, _, _ in lat_data])
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Pipeline Latency Breakdown")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.35)

    # right: FPS bar
    ax2 = axes[1]
    fps_vals = []
    for lbl, col, df in lat_data:
        fps = float(df.loc["fps_total", "median"]) if "fps_total" in df.index else 0
        fps_vals.append(fps)
    cols2 = [col for _, col, _ in lat_data]
    bars = ax2.bar(x2, fps_vals, 0.5, color=cols2, zorder=3)
    for bar, val in zip(bars, fps_vals):
        ax2.text(bar.get_x()+bar.get_width()/2, val+0.3,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels([lbl for lbl, _, _ in lat_data])
    ax2.set_ylabel("Frames per second (FPS)")
    ax2.set_title("End-to-End FPS")
    ax2.grid(axis="y", alpha=0.35)

    fig.suptitle("Deployment Latency: PT CPU vs TRT FP16 vs TRT FP32", fontsize=13, y=1.01)
    _save(fig, "05_latency_comparison.png")

if lat_data:
    components  = ["resnet50", "medfusionnet", "meta_learner"]
    comp_labels = ["ResNet-50", "MedFusionNet", "Meta-Learner"]
    n_comp = len(components)
    n_back = len(lat_data)
    W2     = 0.22
    offs   = np.linspace(-(n_back-1)*W2/2, (n_back-1)*W2/2, n_back)
    xi     = np.arange(n_comp)

    fig, ax = plt.subplots(figsize=(12, 5))
    for (lbl, col, df), off in zip(lat_data, offs):
        meds   = [float(df.loc[c, "median"]) if c in df.index else 0 for c in components]
        p95s   = [float(df.loc[c, "p95"])    if c in df.index else 0 for c in components]
        spikes = [p - m for p, m in zip(p95s, meds)]
        ax.bar(xi + off, meds,   W2, label=f"{lbl} median", color=col, zorder=3)
        ax.bar(xi + off, spikes, W2, bottom=meds, color=col, alpha=0.35, zorder=3)
        for i, p95 in enumerate(p95s):
            ax.text(xi[i]+off, p95+0.5, f"{p95:.1f}",
                    ha="center", va="bottom", fontsize=6.5, color="gray")
    ax.set_xticks(xi); ax.set_xticklabels(comp_labels)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Per-Component Latency (median + P95 tail): PT CPU vs TRT FP16 vs TRT FP32")
    lat_patches = [mpatches.Patch(color=col, label=lbl) for lbl, col, _ in lat_data]
    ax.legend(handles=lat_patches, fontsize=10)
    ax.grid(axis="y", alpha=0.35)
    _save(fig, "06_per_component_latency.png")

if lat_data and len(lat_data) >= 2:
    pt_lat = float(lat_data[0][2].loc["total", "median"]) if "total" in lat_data[0][2].index else None
    if pt_lat:
        fig, ax = plt.subplots(figsize=(8, 5))
        trt_lbls, speedups, cols3 = [], [], []
        for lbl, col, df in lat_data[1:]:
            trt_lat = float(df.loc["total", "median"]) if "total" in df.index else None
            if trt_lat:
                su = pt_lat / trt_lat
                trt_lbls.append(lbl); speedups.append(su); cols3.append(col)
        if speedups:
            bars = ax.bar(np.arange(len(speedups)), speedups, 0.45, color=cols3, zorder=3)
            ax.axhline(1.0, color="black", linestyle="--", lw=1, alpha=0.5)
            for bar, val in zip(bars, speedups):
                ax.text(bar.get_x()+bar.get_width()/2, val+0.05,
                        f"x{val:.1f}", ha="center", va="bottom",
                        fontsize=11, fontweight="bold")
            ax.set_xticks(np.arange(len(speedups))); ax.set_xticklabels(trt_lbls)
            ax.set_ylabel(f"Speedup over PT CPU ({pt_lat:.1f}ms)")
            ax.set_title("TensorRT Speedup over PyTorch CPU")
            ax.grid(axis="y", alpha=0.35)
            _save(fig, "07_trt_speedup_over_pt.png")

print(f"\n  All plots saved to: {_PLOT_DIR.relative_to(_ROOT)}")
