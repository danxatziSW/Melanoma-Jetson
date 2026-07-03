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
resolves to the default PyPI wheel — which on Windows and most Linux distros is **CPU-only**, even
if you have an NVIDIA GPU. Check what you got:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If `cuda.is_available()` is `False` and you do have an NVIDIA GPU, reinstall torch from PyTorch's
CUDA index instead of PyPI:

```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/<tag>
```

Pick `<tag>` to match your GPU driver, not your GPU model — run `nvidia-smi` and read the `CUDA
Version` in the top-right of the header; that's the *maximum* CUDA version your driver supports.
Any index tag at or below that number will work (the CUDA runtime is backward compatible). Exact
tags available shift as PyTorch releases new versions, so check what actually exists before
picking one:

```bash
curl -s https://download.pytorch.org/whl/torch/ | grep -oE "torch-<version>\+cu[0-9]+-cp<major><minor>[^\"]*win_amd64\.whl" | sort -u
# e.g. torch-2.12.1 / cp312 -> cu126, cu130, cu132 were the choices at time of writing
```

(swap `win_amd64` for `linux_x86_64` on Linux). No NVIDIA GPU, or on a Mac / CPU-only box? Skip
this section — the default `pip install -r requirements.txt` is exactly what you want.

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
needs to be rebuilt by hand. There's no schema enforcement: the loaders just read whatever
columns they expect out of these CSVs, so this section documents what those columns are.

`image_path` is relative to `paths.melanoma_data`, not an absolute path. Every
loader calls `src.utils.io.resolve_dataset_paths` right after reading a split CSV, which joins
these relative paths onto whatever `melanoma_data` resolves to on the machine running the
script. That's what lets the same CSVs work unchanged on any computer: clone the repo, point
`melanoma_data` at your own copy of the raw data, and the splits just work.

### Raw dataset layout

Point `configs/base.yaml`'s `paths.melanoma_data` (or the `MELANOMA_DATA_DIR` environment
variable) at a directory containing the three datasets, laid out like:

```
<melanoma_data>/HAMM10000/merged/ISIC_0024343.jpg
<melanoma_data>/ISIC2019/ISIC_2019_Training_Input/ISIC_2019_Training_Input/ISIC_0067139.jpg
<melanoma_data>/ISIC2020/ISIC_2020_Training_JPEG/train/ISIC_6189375.jpg
<melanoma_data>/isic2018-challenge-task1-data-segmentation/versions/1/ISIC2018_Task1-2_Training_Input/ISIC_0000000.jpg
```

(`ham10000`'s folder is spelled `HAMM10000`, and the ISIC2019 image folder is nested twice.
Both are quirks of the original download layout, not typos here. The ISIC2018 path has a Kaggle
`versions/1` segment because that's how it was downloaded.)

### `cls_train.csv` / `cls_val.csv` / `cls_test.csv`

Consumed by `scripts/no_seg/*.py` (the first four columns).

| column | meaning | example |
|---|---|---|
| `image_path` | path to the image, relative to `melanoma_data` | `HAMM10000/merged/ISIC_0024343.jpg` |
| `label_str` | class name | one of `mel`, `nv`, `bcc`, `akiec`, `bkl`, `df`, `vasc` |
| `label_int` | integer-encoded `label_str` | `1` |
| `dataset_source` | which dataset the row came from | `ham10000`, `isic2019`, `isic2020` |
| `lesion_id` | source dataset's lesion identifier | `HAM_0000150` |
| `age_approx` | patient age (metadata input for MedFusionNet) | `50.0` |
| `sex` | patient sex (metadata input for MedFusionNet) | `male` / `female` |
| `anatom_site_general_challenge` | lesion body site (metadata input for MedFusionNet) | `back` |

### Detection: `outputs/detection/train.txt` / `val.txt`

The YOLOv8 lesion detector doesn't go through `src/utils/config.py` at all: Ultralytics reads
its own file list directly, so it needs different handling. `train.txt`/`val.txt` are plain
lists of image paths (one per line, relative to `melanoma_data`, same convention as the CSVs
above). Each image has a matching YOLO-format label file sitting right next to it on disk,
`ISIC_0024306.jpg` next to `ISIC_0024306.txt`, containing one line per box:

```
0 0.573333 0.500000 0.600000 1.000000
```

(`class x_center y_center width height`, all normalized 0-1; class `0` is the only class,
`lesion`. Ultralytics finds these automatically next to each image since the paths don't
contain an `images/` folder component for it to swap for `labels/`.)

Since Ultralytics reads `train.txt`/`val.txt` itself, our `resolve_dataset_paths` helper (which
only works on DataFrames) can't fix these paths at load time. Instead, run
`scripts/prepare_detection_lists.py` once per machine: it reads the tracked, relative
`train.txt`/`val.txt` and writes `train_resolved.txt`/`val_resolved.txt` with real absolute
paths for whatever `melanoma_data` resolves to locally. `outputs/detection/dataset.yaml` points
at the `_resolved` versions, and those are gitignored since they're machine-specific. Ultralytics
resolves `train:`/`val:` in the data yaml relative to the yaml file's own directory, so
`dataset.yaml` references them by filename only (`train_resolved.txt`, not
`outputs/detection/train_resolved.txt`) since it already lives in `outputs/detection/`.

`train.txt` is all HAM10000; `val.txt` is entirely from the ISIC 2018 Task 1 dataset, so the
detector is validated on data it never saw during training.

`paths.data_splits` and `paths.outputs` in the configs are resolved relative to the repo root,
so they work out of the box after cloning. Only the raw dataset location and the splits
themselves need providing.

## Reproducibility

- All training entry points call `src.utils.reproducibility.seed_everything(seed)` (default
  seed `42`, set in `configs/base.yaml`), which seeds Python/NumPy/PyTorch and forces
  deterministic cuDNN kernels.
- Hyperparameters live in `configs/base.yaml` + `configs/models/<model>.yaml`, merged by
  `src.utils.config.load_config`.
- `image_path` in `data_splits/*.csv` is relative to `paths.melanoma_data`, not a
  hardcoded absolute path. `src.utils.io.resolve_dataset_paths` joins it onto whatever
  `melanoma_data` resolves to on the machine that's running the script, which is what makes the
  same CSVs work unchanged on any computer. If you ever regenerate splits with absolute paths
  baked in, run `scripts/normalize_split_paths.py` once to convert them back to relative.
- Detection's `train.txt`/`val.txt` follow the same relative-path convention but are consumed
  directly by Ultralytics, not our own loaders, so they need `scripts/prepare_detection_lists.py`
  run once per machine instead (see Training the detector below).

## Training & evaluation

Everything under `scripts/no_seg/` assumes `data_splits/*.csv` already exist. None of these
scripts take CLI flags for picking models/datasets beyond what's shown below; where a script
needs to know which specific checkpoints to use (a model pair, a triplet), that's a constant
near the top of the file (`MODEL1`, `DATASET1`, `AUG_MODE`, ...), not an argument.

```bash
# Train one model across all datasets (6 models available, see ALL_MODELS in the script)
python scripts/no_seg/run_ablation_no_segmentation.py --models resnet50 --datasets all

# Cross-dataset evaluation of trained checkpoints
python scripts/no_seg/evaluate_noseg_models.py
```

### Ensembles

`scripts/no_seg/nonSens/` searches ensembles of the base ("none" aug) checkpoints;
`scripts/no_seg/sens/` does the same for sensitivity fine-tuned ("none_sens") checkpoints. Both
follow the same pattern: majority vote vs. mean-probability, 2-model vs. 3-model. Pick the file
that matches what you want, e.g.:

```bash
python scripts/no_seg/nonSens/evaluate_ensemble_3models.py       # majority vote, all triplets
python scripts/no_seg/nonSens/evaluate_ensemble_3models_mean.py  # mean probability, all triplets
python scripts/no_seg/sens/evaluate_3models_majority_sens.py     # same, on sens-tuned checkpoints
```

### Meta-learner (the part that gets deployed)

Rather than a fixed vote/mean rule, the deployed ensemble stacks two models' probabilities
through a small `StandardScaler` + `LogisticRegression`. Getting from "trained checkpoints" to
"deployable meta-learner" is a few steps:

```bash
# 1. Fine-tune for sensitivity (produces the *_none_sens.pt checkpoints)
python scripts/no_seg/sens/train_sensitivity_all.py

# 2. Search all model pairs / triplets, fit a meta-learner per combo, rank by avg F1
python scripts/no_seg/sens/evaluate_meta_2models.py     # pairs
python scripts/no_seg/sens/evaluate_meta_stacking.py    # triplets (set FOCUS_TRIPLETS to narrow)

# 3. Validate the specific pair you've decided to deploy (edit MODEL1/DATASET1/MODEL2/DATASET2
#    at the top of both this script and export_for_deployment.py to match)
python scripts/no_seg/sens/evaluate_deployment_pair.py

# 4. Export that pair to ONNX and pickle the fitted meta-learner
python scripts/no_seg/sens/export_for_deployment.py
```

Step 4 is what produces `outputs/ablation_noseg/meta/deployment/meta_learner.pkl` and the two
`*_none_sens.onnx` files that `scripts/deployedTensorrt/convert.py` turns into the TensorRT
engines the dashboard and JETSON.md both consume.

### Converting any checkpoint to ONNX

ONNX exports aren't committed to the repo (regenerate-able, and the ablation ones alone run to
several GB), so training or re-training a model leaves you with just a `.pt` checkpoint.
`export_for_deployment.py` above only exports the two models it's hardcoded to deploy;
`scripts/no_seg/convert_to_onnx.py` is the general version, for any checkpoint produced by
`run_ablation_no_segmentation.py` or `train_sensitivity_all.py`:

```bash
python scripts/no_seg/convert_to_onnx.py --models resnet50 --datasets ham10000 --aug none
python scripts/no_seg/convert_to_onnx.py --models all --datasets all --aug none light none_sens

# the YOLOv8 detector too (outputs/detection/checkpoints/best.pt -> best.onnx)
python scripts/no_seg/convert_to_onnx.py --detection --skip-classifiers
```

Classifier checkpoints write to `outputs/ablation_noseg/<dataset>/<model>_<aug>/onnx/<model>_<aug>.onnx`,
matching where the checkpoint they were built from lives. Handles the metadata input
(MedFusionNet) and the yolov8_cls ultralytics-vs-timm-fallback checkpoint shape automatically.

The detector goes through Ultralytics' own `.export()` instead of raw `torch.onnx.export`: YOLO
models need export-time graph surgery (folding the detection head) that plain tracing wouldn't
reproduce. It writes `best.onnx` next to `best.pt` in `outputs/detection/checkpoints/`. First run
auto-installs `onnxslim` if it's missing (an Ultralytics exporter dependency, listed but commented
out in `requirements.txt` alongside `ultralytics` itself).

### Training the detector

```bash
python scripts/no_seg/train_detection.py --epochs 40 --imgsz 640 --batch 16 --seed 0
```

Under the hood there's no Ultralytics Python training API in use here beyond what `yolo detect
train` already does: `train_detection.py` is a thin wrapper around that CLI command (using
`outputs/detection/dataset.yaml`; see the Data section above for the label format), kept as a
script so training the detector looks like every other training step in this repo instead of a
multi-line command you'd have to type and then fix up by hand. It also:

- Runs `scripts/prepare_detection_lists.py` first, so `train_resolved.txt`/`val_resolved.txt`
  are always current for whatever machine it's running on.
- Resolves the `yolo` executable next to the current Python interpreter, so it works whether or
  not the venv happens to be on `PATH`.
- Passes `exist_ok=True` and copies `checkpoints/yolov8n_lesion/weights/best.pt` up to the flat
  `checkpoints/best.pt` afterward, since that's where `convert_to_onnx.py`, the dashboard, and
  JETSON.md all expect to find it (Ultralytics always nests its own output one level down,
  regardless of `project=`).

`yolov8n.pt` (`--model` to override) is the stock pretrained COCO checkpoint Ultralytics
fine-tunes from; it downloads automatically the first time you run this if it's not already
sitting at the repo root.

## Edge deployment & live dashboard

See [JETSON.md](JETSON.md) for converting to TensorRT, running the latency benchmark, and
starting the live camera dashboard on a Jetson device. Pre-built FP16 TensorRT engines for the
detector + classifier ensemble are included under `outputs/` for JetPack 6 (L4T R36.4.7); other
setups need to rebuild them (instructions in JETSON.md).

## License

MIT. See [LICENSE](LICENSE) for the full text.
