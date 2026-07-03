"""Generate publication-quality plots for the TRT deployment evaluation.

Reads evaluation_trt_fp16.xlsx and evaluation_trt_fp32.xlsx from
outputs/ablation_noseg/meta/deployment/tensorrt/ and writes plots to
outputs/ablation_noseg/meta/deployment/tensorrt/plots/comparison/

Usage:
    python3 scripts/no_seg/deployedTensorrt/plot_results.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

_ROOT     = Path(__file__).resolve().parents[3]
_TRT_DIR  = _ROOT / "outputs" / "ablation_noseg" / "meta" / "deployment" / "tensorrt"
_PLOT_DIR = _TRT_DIR / "plots" / "comparison"
_PLOT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.size":         11,
    "axes.grid":         True,
    "grid.alpha":        0.35,
})

_FP16   = "#4C72B0"
_FP32   = "#DD8452"
_R50    = "#4C72B0"
_MFN    = "#DD8452"
_META   = "#55A868"
_RED    = "#C44E52"

_DS_COL   = {"ham10000": "#4C72B0", "isic2019": "#DD8452", "isic2020": "#55A868"}
_DS_LABEL = {"ham10000": "HAM10000", "isic2019": "ISIC-2019", "isic2020": "ISIC-2020"}
_DS_ORDER = ["ham10000", "isic2019", "isic2020"]

W = 0.28   # bar width


def _save(fig: plt.Figure, name: str) -> None:
    path = _PLOT_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path.relative_to(_ROOT)}")


def _load(prec: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    xl   = pd.ExcelFile(_TRT_DIR / f"evaluation_trt_{prec}.xlsx")
    res  = xl.parse("Results")
    lat  = xl.parse("Latency").set_index("component")
    res["dataset_label"] = res["dataset"].map(_DS_LABEL)
    return res, lat


res16, lat16 = _load("fp16")
res32, lat32 = _load("fp32")

ds_labels = [_DS_LABEL[d] for d in _DS_ORDER]
xs = np.arange(len(_DS_ORDER))


def _res_col(df: pd.DataFrame, col: str) -> list[float]:
    return [float(df[df["dataset"] == d][col].iloc[0]) for d in _DS_ORDER]


fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, (res, prec, col) in zip(axes, [(res16, "FP16", _FP16), (res32, "FP32", _FP32)]):
    r50  = _res_col(res, "auc_resnet50")
    mfn  = _res_col(res, "auc_medfusionnet")
    meta = _res_col(res, "auc_meta")
    b1 = ax.bar(xs - W,  r50,  W, label="ResNet-50",    color=_R50,  zorder=3)
    b2 = ax.bar(xs,      mfn,  W, label="MedFusionNet", color=_MFN,  zorder=3)
    b3 = ax.bar(xs + W,  meta, W, label="Meta-Learner", color=_META, zorder=3)
    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(ds_labels)
    ax.set_ylim(0.65, 1.07)
    ax.set_ylabel("AUC-ROC")
    ax.set_title(f"TRT {prec}")
    ax.legend(fontsize=9)
axes[0].set_title("AUC-ROC: ResNet-50 vs MedFusionNet vs Meta-Learner\n(TRT FP16)", fontsize=11)
axes[1].set_title("AUC-ROC: ResNet-50 vs MedFusionNet vs Meta-Learner\n(TRT FP32)", fontsize=11)
_save(fig, "01_auc_comparison.png")


fig, axes = plt.subplots(1, 2, figsize=(13, 5))
w2 = 0.35
for ax, metric, ylabel in [
    (axes[0], "deployed_sensitivity", "Sensitivity"),
    (axes[1], "deployed_specificity", "Specificity"),
]:
    v16 = _res_col(res16, metric)
    v32 = _res_col(res32, metric)
    b16 = ax.bar(xs - w2/2, v16, w2, label="TRT FP16", color=_FP16, zorder=3)
    b32 = ax.bar(xs + w2/2, v32, w2, label="TRT FP32", color=_FP32, zorder=3)
    for bars in (b16, b32):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    if metric == "deployed_sensitivity":
        ax.axhline(0.85, color=_RED, linestyle="--", lw=1.2, label="Target ≥0.85")
    ax.set_xticks(xs)
    ax.set_xticklabels(ds_labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} @ deployed threshold")
    ax.legend(fontsize=9)
fig.suptitle("Meta-Learner Performance:FP16 vs FP32", fontsize=13, y=1.01)
_save(fig, "02_sensitivity_specificity.png")


fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
components = ["resnet50", "medfusionnet", "meta_learner"]
comp_labels = ["ResNet-50", "MedFusionNet", "Meta-Learner"]
comp_cols   = [_R50, _MFN, _META]
x2 = np.array([0, 1])
bottoms = np.zeros(2)
for comp, label, col in zip(components, comp_labels, comp_cols):
    vals = np.array([
        float(lat16.loc[comp, "median"]),
        float(lat32.loc[comp, "median"]),
    ])
    bars = ax.bar(x2, vals, 0.45, bottom=bottoms, label=label, color=col, zorder=3)
    for i, (bar, val) in enumerate(zip(bars, vals)):
        if val > 0.5:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bottoms[i] + val / 2,
                    f"{val:.1f}ms", ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")
    bottoms += vals

fps16 = float(lat16.loc["fps_total", "median"]) if "fps_total" in lat16.index else 0
fps32 = float(lat32.loc["fps_total", "median"]) if "fps_total" in lat32.index else 0

for xi, (tot, fps) in enumerate([(bottoms[0], fps16), (bottoms[1], fps32)]):
    ax.text(xi, tot + 0.3, f"{tot:.1f}ms\n({fps:.0f} FPS)",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xticks(x2)
ax.set_xticklabels(["TRT FP16", "TRT FP32"])
ax.set_ylabel("Latency (ms)")
ax.set_title("Pipeline Latency Breakdown")
ax.legend(loc="upper right", fontsize=9)

ax2 = axes[1]
w3  = 0.3
xi2 = np.arange(len(components))
for shift, lat, prec, col in [(-w3/2, lat16, "FP16", _FP16), (w3/2, lat32, "FP32", _FP32)]:
    meds = [float(lat.loc[c, "median"]) for c in components]
    p95s = [float(lat.loc[c, "p95"])    for c in components]
    spikes = [p - m for p, m in zip(p95s, meds)]
    bars = ax2.bar(xi2 + shift, meds,   w3, label=f"TRT {prec} median", color=col, zorder=3)
    ax2.bar(xi2 + shift, spikes, w3, bottom=meds, color=col, alpha=0.35, zorder=3)
    for i, (bar, p95) in enumerate(zip(bars, p95s)):
        ax2.text(bar.get_x() + bar.get_width() / 2, p95 + 0.2,
                 f"p95={p95:.1f}", ha="center", va="bottom", fontsize=7, color="gray")

ax2.set_xticks(xi2)
ax2.set_xticklabels(comp_labels)
ax2.set_ylabel("Latency (ms)")
ax2.set_title("Per-Component Median + P95 Tail")
ax2.legend(fontsize=9)
fig.suptitle("Deployment Pipeline Latency:TRT FP16 vs FP32", fontsize=13, y=1.01)
_save(fig, "03_latency_breakdown.png")


fig, axes = plt.subplots(1, 2, figsize=(11, 5))

fps_vals = [fps16, fps32]
lat_vals = [float(lat16.loc["total", "median"]), float(lat32.loc["total", "median"])]
labels_2 = ["TRT FP16", "TRT FP32"]
cols_2   = [_FP16, _FP32]

ax = axes[0]
bars = ax.bar([0, 1], fps_vals, 0.5, color=cols_2, zorder=3)
for bar, val in zip(bars, fps_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3,
            f"{val:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_xticks([0, 1])
ax.set_xticklabels(labels_2)
ax.set_ylabel("Frames per second")
ax.set_title("End-to-End FPS")

ax = axes[1]
bars = ax.bar([0, 1], lat_vals, 0.5, color=cols_2, zorder=3)
for bar, val in zip(bars, lat_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.2,
            f"{val:.1f} ms", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_xticks([0, 1])
ax.set_xticklabels(labels_2)
ax.set_ylabel("Total latency (ms)")
ax.set_title("End-to-End Total Latency")

fig.suptitle("Deployment Pipeline Throughput:TRT FP16 vs FP32", fontsize=13, y=1.01)
_save(fig, "04_fps_latency.png")


fig, ax = plt.subplots(figsize=(9, 5))
v16 = _res_col(res16, "deployed_f2")
v32 = _res_col(res32, "deployed_f2")
b16 = ax.bar(xs - w2/2, v16, w2, label="TRT FP16", color=_FP16, zorder=3)
b32 = ax.bar(xs + w2/2, v32, w2, label="TRT FP32", color=_FP32, zorder=3)
for bars in (b16, b32):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8.5)
ax.set_xticks(xs)
ax.set_xticklabels(ds_labels)
ax.set_ylim(0, 1.1)
ax.set_ylabel("F2 Score")
ax.set_title("Meta-Learner F2 Score @ Deployed Threshold\nTRT FP16 vs FP32")
ax.legend()
_save(fig, "05_f2_score.png")


metrics = [
    ("auc_meta",              "AUC-ROC"),
    ("deployed_sensitivity",  "Sensitivity"),
    ("deployed_specificity",  "Specificity"),
    ("deployed_f2",           "F2 Score"),
]
fig, axes = plt.subplots(1, len(metrics), figsize=(16, 5), sharey=False)
for ax, (col, title) in zip(axes, metrics):
    v16 = _res_col(res16, col)
    v32 = _res_col(res32, col)
    b16 = ax.bar(xs - w2/2, v16, w2, color=_FP16, zorder=3)
    b32 = ax.bar(xs + w2/2, v32, w2, color=_FP32, zorder=3)
    for bars in (b16, b32):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels(ds_labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_title(title)
fp16_p = mpatches.Patch(color=_FP16, label="TRT FP16")
fp32_p = mpatches.Patch(color=_FP32, label="TRT FP32")
fig.legend(handles=[fp16_p, fp32_p], loc="upper right", fontsize=10)
fig.suptitle("Meta-Learner:All Key Metrics: FP16 vs FP32", fontsize=13)
_save(fig, "06_all_metrics.png")


print(f"\n  All plots saved to: {_PLOT_DIR.relative_to(_ROOT)}")
