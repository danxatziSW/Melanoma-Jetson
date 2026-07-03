"""Regenerate the standalone full-pipeline stacked-latency figure from benchmark_results.xlsx.

Plots the four profiled stages (YOLO/ResNet-50/MedFusionNet/Meta-LR) and labels
the total as the sum of those stages, so the annotated total always matches
what the stacked bar actually shows.

Usage:
    python scripts/deployedTensorrt/plot_pipeline_latency_stacked.py [xlsx_path]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]

_DEFAULT_XLSX = (_ROOT / "outputs" / "ablation_noseg" / "metaJetson" / "deployment"
                  / "tensorrt" / "plots" / "fullPipeline" / "benchmark_results.xlsx")

_C_YOLO = "#E07B54"
_C_R50  = "#4C72B0"
_C_MFN  = "#C15CA0"
_C_META = "#55A868"


def main() -> None:
    xlsx_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_XLSX
    out_path  = xlsx_path.parent / "pipeline_latency_stacked.png"

    lat = pd.read_excel(xlsx_path, sheet_name="Latency").set_index("component")
    fps = pd.read_excel(xlsx_path, sheet_name="FPS").set_index("metric")["value"]

    yolo = float(lat.loc["YOLO FP16 (detect)",       "median_ms"])
    r50  = float(lat.loc["ResNet-50 FP16 (classify)", "median_ms"])
    mfn  = float(lat.loc["MedFusionNet FP16 (classify)", "median_ms"])
    meta = float(lat.loc["Meta-LR sklearn (classify)", "median_ms"])
    fps_total = float(fps.loc["FPS (full pipeline)"])

    components = ["YOLO\n(detect)", "ResNet-50\n(classify)", "MedFusion\n(classify)",
                  "Meta\n(sklearn)"]
    values = [yolo, r50, mfn, meta]
    colors = [_C_YOLO, _C_R50, _C_MFN, _C_META]
    total = sum(values)

    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
    })

    fig, ax = plt.subplots(figsize=(6, 6))
    bottom = 0.0
    for comp, val, col in zip(components, values, colors):
        ax.bar([0], [val], 0.45, bottom=bottom, color=col, label=comp, zorder=3)
        if val > 1.0:
            ax.text(0, bottom + val / 2, f"{val:.1f}ms",
                    ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        bottom += val

    ax.text(0, bottom + 1, f"{total:.1f}ms\n({fps_total:.0f} FPS)",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xlim(-0.5, 0.5); ax.set_xticks([])
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Full Pipeline\nLatency Breakdown")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, bottom * 1.3)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"stage sum (labeled total) = {total:.2f}ms")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
