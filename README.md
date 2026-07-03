# Melanoma Detection Pipeline

A dermoscopy image pipeline for melanoma (mel) vs. non-melanoma classification, built around
lesion detection (YOLOv8) feeding a small classifier ensemble, with an edge-deployment path
(TensorRT on Jetson) and a live camera dashboard.

Trained and evaluated on HAM10000, ISIC 2019, and ISIC 2020 for classification, plus ISIC 2018
(Task 1) for the lesion detector.

---

## Pipeline

```
image → [YOLOv8 detector] → lesion crop → [quality check] →
        [ResNet-50, MedFusionNet, ...] → sklearn meta-learner → benign / malignant
```

The pipeline classifies the YOLO crop directly with no intermediate segmentation step, which is
why the training code lives under the "no-seg" path (`scripts/no_seg/`).

## Repo layout

```
src/                   Model definitions, datasets, augmentation, training/eval utilities
configs/               YAML configs (base + per-model hyperparameters)
scripts/no_seg/        Training + ablation for the detect-then-classify pipeline
scripts/deployedTensorrt/  ONNX → TensorRT conversion, Jetson latency/accuracy evaluation
dashboard/             FastAPI backend + React frontend for the live camera demo
outputs/               Checkpoints, exported models, metrics (mostly gitignored, see below)
data_splits/           train/val/test CSVs (tracked in git, see Data)
```

## Setup

```bash
git clone <repository-url>
cd <cloned-directory>
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### GPU acceleration (CUDA)

`requirements.txt` pins `torch>=2.2.0` with no index URL, so plain `pip install -r requirements.txt`
resolves to the CPU-only PyPI wheel, even with an NVIDIA GPU present. Check what you got:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If `cuda.is_available()` is `False` and you have an NVIDIA GPU, reinstall torch from PyTorch's
CUDA index instead of PyPI, matching `<tag>` to your driver's `CUDA Version` from `nvidia-smi`
(any tag at or below that number works):

```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/<tag>
```

Optional extras depending on what you run:
```bash
pip install ultralytics                       # YOLOv8 detector/classifier
pip install "fastapi[standard]" uvicorn psutil  # dashboard backend
cd dashboard/frontend && npm install            # dashboard frontend
```

TensorRT itself isn't pip-installable generically; it ships with JetPack on Jetson devices.
See [JETSON.md](JETSON.md) for the on-device deployment path.

## Data

This repo doesn't include the datasets or a data-preparation script, so `data_splits/*.csv`
needs to be rebuilt by hand. Point `configs/base.yaml`'s `paths.melanoma_data` (or the
`MELANOMA_DATA_DIR` environment variable) at a directory containing HAM10000, ISIC 2019,
ISIC 2020, and ISIC 2018 Task 1 (for the detector). `image_path` in the split CSVs is relative
to `melanoma_data`; `src.utils.io.resolve_dataset_paths` joins it at load time, so the same CSVs
work unchanged on any machine once `melanoma_data` points at a local copy of the data.

Detection labels (`outputs/detection/train.txt`/`val.txt`) are consumed directly by Ultralytics
rather than our own loaders, so they need `scripts/prepare_detection_lists.py` run once per
machine to resolve them to absolute paths — this happens automatically as part of
`train_detection.py` (step 3 below).

If you ever regenerate splits with absolute paths baked in, run
`scripts/normalize_split_paths.py` once to convert them back to relative.

## Full pipeline, in order

Everything below assumes `data_splits/*.csv` already exist (see Data). Training scripts don't
take CLI flags for picking models/datasets beyond what's shown; where a script needs specific
checkpoints (a model pair, a triplet), that's a constant near the top of the file (`MODEL1`,
`DATASET1`, ...), not an argument — edit it before running.

### 1. Train classifiers (ablation)

```bash
python scripts/no_seg/run_ablation_no_segmentation.py --models resnet50 --datasets all
# or: --models all --datasets all
```

Six architectures available (`ALL_MODELS` in the script): `resnet50`, `efficientnet_b2`,
`mobilenetv3_large`, `convnext_tiny_se`, `medfusionnet`, `yolov8_cls`.

### 2. Fine-tune for sensitivity

```bash
python scripts/no_seg/sens/train_sensitivity_all.py
```

Produces the `*_none_sens.pt` checkpoints that the meta-learner and deployment steps expect.

### 3. Train the lesion detector

```bash
python scripts/no_seg/train_detection.py --epochs 40 --imgsz 640 --batch 16 --seed 0
```

Thin wrapper around `yolo detect train`. Runs `prepare_detection_lists.py` first, then copies
`checkpoints/yolov8n_lesion/weights/best.pt` up to the flat `checkpoints/best.pt` that
`convert_to_onnx.py`, the dashboard, and JETSON.md all expect. `yolov8n.pt` (override with
`--model`) downloads automatically on first run if missing.

### 4. Evaluate

```bash
python scripts/no_seg/evaluate_noseg_models.py                 # cross-dataset accuracy, single models
python scripts/no_seg/nonSens/evaluate_ensemble_3models.py      # ensembles of base checkpoints
python scripts/no_seg/sens/evaluate_3models_majority_sens.py    # ensembles of sensitivity checkpoints
```

`scripts/no_seg/nonSens/` and `scripts/no_seg/sens/` both follow majority-vote vs.
mean-probability, 2-model vs. 3-model — pick the file matching what you want to check.

### 5. Fit and pick the meta-learner

The deployed ensemble stacks two models' probabilities through a `StandardScaler` +
`LogisticRegression`, rather than a fixed vote/mean rule.

```bash
python scripts/no_seg/sens/evaluate_meta_2models.py     # rank all pairs by sensitivity/F2
python scripts/no_seg/sens/evaluate_meta_stacking.py    # same, for triplets (set FOCUS_TRIPLETS to narrow)

# edit MODEL1/DATASET1/MODEL2/DATASET2 in evaluate_deployment_pair.py AND export_for_deployment.py
# to match the winning pair, then:
python scripts/no_seg/sens/evaluate_deployment_pair.py  # validate that specific pair
```

These two scripts fit a meta-learner in memory just to score each candidate; nothing is saved to
disk here — that happens in step 6.

### 6. Convert to ONNX / TensorRT

```bash
# ONNX for the deployed pair + pickle the fitted meta-learner
python scripts/no_seg/sens/export_for_deployment.py

# ONNX for any other checkpoint, or the detector
python scripts/no_seg/convert_to_onnx.py --models resnet50 --datasets ham10000 --aug none_sens
python scripts/no_seg/convert_to_onnx.py --detection --skip-classifiers
```

Step 6 produces `outputs/ablation_noseg/meta/deployment/meta_learner.pkl` and the two
`*_none_sens.onnx` files that `scripts/deployedTensorrt/convert.py` turns into the TensorRT
engines the dashboard and JETSON.md both consume. See [JETSON.md](JETSON.md) for that
conversion, the latency benchmark, and running the live camera dashboard on a Jetson device.

## License

MIT. See [LICENSE](LICENSE) for the full text.
