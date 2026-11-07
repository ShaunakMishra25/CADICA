# CADICA Coronary Artery Disease Severity Classification

A reproducible research pipeline for classifying coronary artery stenosis severity from
coronary angiographic video frames in the CADICA dataset: a patient-level, three-class
benchmark built with PyTorch.

> **Status:** data preparation, sampling, patient-level cross-validation, dataset, and
> baseline model are implemented and tested. A three-class training/evaluation CLI is the
> next milestone (see [Roadmap](#roadmap)).

---

## Highlights

- **Patient-level evaluation.** Folds are stratified by severity and split by patient, so
  frames from one patient never appear in both training and validation.
- **Deterministic sampling.** Up to eight evenly spaced frames are sampled from every
  angiographic video, always including the first and last frame.
- **Portable manifests.** Sampling and split manifests store paths relative to the raw
  dataset root, so they work across machines.
- **Reproducible by default.** A fixed seed (`42`) and a documented pipeline make every
  split easy to regenerate.
- **Packaged and tested.** A `src/` layout, console entry points, and a pytest suite.

---

## Dataset

The pipeline expects the extracted CADICA distribution with this layout:

```text
CADICA/
├── metadata.xlsx
└── selectedVideos/
    ├── p1/
    │   ├── v1/
    │   │   └── input/
    │   │       ├── p1_v1_00001.png
    │   │       └── ...
    │   └── v2/
    └── p2/
```

The severity label comes from the metadata column
`Maximum degree of the coronary artery involvement` and is mapped to three classes:

| Severity | Class index | Patients | Sampled frames |
| -------- | ----------: | -------: | -------------: |
| `<20%`   |           0 |        6 |            384 |
| `20-50%` |           1 |        8 |            448 |
| `>70%`   |           2 |       28 |          2,210 |
| **Total**|             |   **42** |      **3,042** |

The cohort is strongly imbalanced (`>70%` accounts for roughly two thirds of patients),
which is why balanced accuracy, macro F1, and per-class recall should accompany plain
accuracy in any reported result.

The raw dataset is **not** stored in this repository. Point the pipeline at it with the
`CADICA_DATA_ROOT` environment variable.

---

## Repository layout

```text
.
├── data/
│   ├── folds/                    # patient_folds.csv + generated train/val splits
│   └── manifests/                # sampled_frames.csv
├── docs/                         # dataset and pipeline documentation
├── scripts/                      # exploratory and diagnostic scripts
├── src/cadica/
│   ├── cli/                      # build/sample/prepare/verify pipeline commands
│   ├── data/                     # loaders, sampling, PyTorch dataset
│   ├── models/                   # baseline CNN
│   └── config.py                 # paths, labels, experiment defaults
├── tests/                        # pytest suite
├── checkpoints/                  # model artifacts (ignored by Git)
├── results/                      # experiment outputs (ignored by Git)
├── Makefile
├── pyproject.toml
└── requirements.txt
```

---

## Getting started

### Requirements

- Python 3.10+
- PyTorch and torchvision
- The raw CADICA dataset

### Install

```bash
git clone <your-repository-url>
cd ClassificationImplementations

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
```

### Point to the dataset

```bash
export CADICA_DATA_ROOT="/absolute/path/to/CADICA/CADICA"
```

If `CADICA_DATA_ROOT` is unset, `src/cadica/config.py` falls back to the original author's
local path. Set the variable for any portable or shared setup.

---

## Reproduce the data pipeline

Run every stage in order:

```bash
make pipeline
```

Or run the stages individually:

| Command                  | Output                                   | Purpose                                  |
| ------------------------ | ---------------------------------------- | ---------------------------------------- |
| `cadica-build-folds`     | `data/folds/patient_folds.csv`           | Patient-level stratified 5-fold splits   |
| `cadica-sample-frames`   | `data/manifests/sampled_frames.csv`      | Up to 8 evenly spaced frames per video   |
| `cadica-prepare-folds`   | `data/folds/train_fold_*.csv`, `val_*`   | Leave-one-fold-out train/validation CSVs |
| `cadica-verify-folds`    | console report                           | Leakage and coverage checks              |
| `cadica-inspect-folds`   | console report                           | Class balance per fold                   |

The pipeline is idempotent: rerunning it with the same seed and dataset produces the same
splits.

---

## Python API

```python
from cadica.data import CADICADataset
from cadica.models import CADICACNN

dataset = CADICADataset()          # defaults to data/manifests/sampled_frames.csv
sample = dataset[0]

print(sample["image"].shape)       # torch.Size([3, 224, 224])
print(sample["label"])             # tensor(0..2)
print(sample["patient_id"])        # e.g. "p1"

model = CADICACNN(num_classes=3)
logits = model(sample["image"].unsqueeze(0))
```

To use a specific split:

```python
train_dataset = CADICADataset("data/folds/train_fold_1.csv")
val_dataset = CADICADataset("data/folds/val_fold_1.csv")
```

---

## Analysis scripts

```bash
python scripts/check_dataset.py      # raw dataset and metadata coverage
python scripts/inspect_videos.py     # videos per patient
python scripts/inspect_frames.py     # frames per video
python scripts/inspect_images.py     # image sizes and formats
python scripts/inspect_model.py      # architecture and parameter count
python scripts/view_sequence.py --patient p1 --video v1
```

`view_sequence.py` writes a frame-difference plot to `results/` and is headless-safe.

---

## Tests

```bash
pytest
```

Tests that require the raw dataset or generated manifests skip automatically when those
files are unavailable, so the suite also runs on a clean checkout.

---

## Reproducibility notes

- **Seed:** `42` for fold generation and future training runs.
- **Split unit:** patient. `cadica-verify-folds` asserts that train and validation patient
  sets are disjoint for every fold.
- **Sampling:** `round(i * (n - 1) / (k - 1))` for `i = 0..k-1`, where `k = min(8, n)`.
  The first and last frames are always included.
- **Manifest paths:** relative to `CADICA_DATA_ROOT/selectedVideos`, never absolute.

See [`docs/dataset.md`](docs/dataset.md) and [`docs/pipeline.md`](docs/pipeline.md) for
details.

---

## Roadmap

- [ ] Three-class training and evaluation CLI with per-class metrics, confusion matrix,
      and checkpoint selection.
- [ ] Optional augmentation and class-balanced sampling.
- [ ] Video-level models (temporal pooling or 3D CNN) instead of independent frames.
- [ ] Probability calibration and threshold analysis for clinical operating points.
- [ ] Experiment tracking and figure generation.

---

## Citation

If you use the CADICA dataset, cite the original dataset publication and follow its
license terms. This repository only provides the preprocessing and modelling code.

---

## License

No license has been added yet. Add a `LICENSE` file before distributing or publishing this
repository.
