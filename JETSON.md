# Jetson Deployment Guide

Reproducing the on-device inference results from the paper on an **NVIDIA Jetson Orin Nano**.

---

## Requirements

| | |
|---|---|
| Device | NVIDIA Jetson Orin Nano 8 GB |
| JetPack | 6 — L4T R36.4.7 (must match the version used to build the engines — see note below) |
| Python | 3.10+ |
| Node.js | 18+ (live dashboard only) |

> **TensorRT engine compatibility**: the pre-built `.engine` files in this repository were compiled on **JetPack 6 (L4T R36.4.7)**. They will work on any Jetson Orin Nano running the same release. If your device runs a different JetPack version, see [Rebuilding the engines](#rebuilding-the-engines) at the bottom of this page.

---

## Setup

```bash
git clone https://github.com/danxatziSW/Melanoma-Jetson
cd Melanoma-Jetson

pip install -r requirements.txt
pip install jetson-stats        # Jetson GPU / power monitor
pip install "fastapi[standard]" uvicorn psutil   # dashboard backend
```

The pre-built TensorRT FP16 engines and the meta-learner are already included in the repository under `outputs/`. No conversion step is needed.

---

## Run the Latency Benchmark

```bash
python3 scripts/deployedTensorrt/benchmark_pipeline.py
```

This runs the full pipeline (YOLO detection → ResNet-50 → MedFusionNet → meta-learner) on the 10 sample dermoscopy images in `scripts/test_images/` and reports per-stage latency statistics.

**Optional flags:**

| Flag | Default | Description |
|---|---|---|
| `--images <dir>` | `scripts/test_images` | Directory of `.jpg` / `.png` images |
| `--runs <n>` | `100` | Timed passes per image |
| `--warmup <n>` | `20` | Warmup passes before timing |
| `--conf <f>` | `0.35` | YOLO confidence threshold |

**Output** — console table + four plots saved to:
```
outputs/ablation_noseg/meta/deployment/tensorrt/plots/
```

---

## Live Dashboard

Streams inference from a camera with real-time per-stage latency and hardware stats.

```bash
# Terminal 1 — backend
cd dashboard/api
uvicorn main:app --host 0.0.0.0 --port 8000

# Terminal 2 — frontend
cd dashboard/frontend
npm install
npm run dev
```

Open `http://localhost:5173` on the Jetson, or replace `localhost` with the Jetson's IP from another device on the same network.

---

## Pipeline

```
Input frame
    │
    ▼
[Quality]     Rejects blurry frames up front (Laplacian variance < 80) — before spending
              any GPU time on detection or classification
    │
    ▼
[Detect]      YOLOv8n TRT FP16 — locates lesion ROI (falls back to centre crop)
    │
    ▼
[Crop]        Tight crop of the detected lesion region
    │
    ▼
[Classify]    ResNet-50 TRT FP16  ┐
              MedFusionNet TRT FP16 ┘ → sklearn meta-learner → malignancy probability
```

(`dashboard/api/main.py`'s `infer_frame` is the reference implementation of this order; the
latency benchmark script skips the quality gate since it isn't measuring rejection behavior.)

**Engine files used:**

| File | Purpose | Trained on |
|---|---|---|
| `outputs/detection/checkpoints/best_fp16.engine` | YOLOv8n lesion detector | HAM10000 Segmentation |
| `outputs/ablation_noseg/meta/deployment/tensorrt/resnet50_none_sens_fp16.engine` | ResNet-50 classifier | ISIC2019 |
| `outputs/ablation_noseg/meta/deployment/tensorrt/medfusionnet_none_sens_fp16.engine` | MedFusionNet classifier | ISIC2020 |
| `outputs/ablation_noseg/meta/deployment/meta_learner.pkl` | Sklearn meta-learner | 

---

## Troubleshooting

**`Cannot find libcudart.so.12`**
```bash
ls /usr/local/cuda/lib64/libcudart*    # verify CUDA is installed
dpkg -l | grep jetpack                 # verify JetPack version
```

**`tensorrt not found`**
```bash
python3 -c "import tensorrt; print(tensorrt.__version__)"
# If this fails:
sudo apt-get install python3-libnvinfer python3-libnvinfer-dev
```

**Engine fails to deserialize / version mismatch**
The pre-built engines require the same JetPack version used to compile them. See [Rebuilding the engines](#rebuilding-the-engines).
In case rebuilding is needed, models are available at https://ihuedu-my.sharepoint.com/:f:/g/personal/iee2021233_ihu_gr/IgDpItKuXnNyRJrXnqsvkjkJAeCMcCzO9QOeByRVVrQZWrU?e=yz4ly0. 


**Low FPS / thermal throttling**
```bash
sudo nvpmodel -m 0    # maximum power mode
sudo jetson_clocks    # lock CPU/GPU clocks
```

---

## Rebuilding the Engines

To keep the repo small, only the compiled `.engine` files (JetPack 6 / L4T R36.4.7) are
checked in — not the PyTorch checkpoints or ONNX exports they were built from. If your
JetPack version differs, you'll need to reproduce those first. This assumes the classifiers are
already fine-tuned (`*_none_sens.pt` checkpoints) — see the "Meta-learner" section of
[README.md](README.md) if not:

```bash
# 1. Export to ONNX and fit the meta-learner
python3 scripts/no_seg/sens/export_for_deployment.py

# 2. Convert the classifiers to TensorRT
python3 scripts/deployedTensorrt/convert.py --precision fp16

# 3. Export the YOLO detector (Ultralytics' built-in exporter)
yolo export model=outputs/detection/checkpoints/best.pt format=engine half=True imgsz=640
```
