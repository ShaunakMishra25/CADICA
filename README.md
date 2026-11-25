# CADICA Coronary Artery Disease Severity Classification

A reproducible research pipeline for classifying coronary artery stenosis severity from
coronary angiographic video frames in the CADICA dataset: a patient-level, three-class
benchmark built with PyTorch.

> **Status:** data preparation, sampling, patient-level cross-validation, dataset,
> baseline model, and a three-class training/evaluation CLI are implemented and tested.
> See [Roadmap](#roadmap) for planned extensions.

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

## Train a fold

```bash
cadica-train --fold 1 --epochs 10 --batch-size 16
```

The trainer:

- loads `data/folds/train_fold_<k>.csv` and `data/folds/val_fold_<k>.csv`;
- balances classes **and** patients with `WeightedRandomSampler`;
- augments only the training split (small rotation, translation, jitter, noise);
- averages frame probabilities per patient before scoring;
- schedules the learning rate with `ReduceLROnPlateau` on patient-level macro F1;
- supports early stopping with a minimum-epoch guard;
- reports patient-level accuracy, balanced accuracy, macro F1, a confusion matrix, and a
  per-class classification report, for both the last and best epoch;
- writes outputs to the ignored `results/` and `checkpoints/` directories.

`<tag>` is `--run-tag` (default `baseline`). Use a unique tag per condition and seed.

| Output | Description |
| --- | --- |
| `results/<tag>_training_log_fold_<k>.csv` | Per-epoch loss, LR, metrics, best flag |
| `results/<tag>_patient_predictions_fold_<k>.csv` | Last-epoch predictions (fixed-budget headline) |
| `results/<tag>_best_patient_predictions_fold_<k>.csv` | Best-epoch predictions (secondary) |
| `checkpoints/<tag>_fold_<k>_last.pt` | Final model and optimizer state |
| `checkpoints/<tag>_fold_<k>_best.pt` | Best-macro-F1 weights |
| `results/<tag>_<variant>_per_fold.csv` | Aggregated per-fold metrics |
| `results/<tag>_<variant>_summary.csv` | Mean ± std across folds |

Useful options: `--run-tag`, `--seed`, `--lr`, `--num-workers`, `--device`, `--augment`,
`--no-early-stopping`, `--es-patience`, `--es-min-epochs`, `--scheduler`.

For the fixed-budget comparison protocol, disable early stopping so the last epoch is not
selected on the validation fold:

```bash
cadica-train --fold 1 --epochs 60 --run-tag e2_aug_s42 --no-early-stopping
cadica-aggregate --run-tag e2_aug_s42 --variant last
```

Or run all five folds and aggregate with Make:

```bash
make train-all RUN_TAG=e2_aug_s42 EPOCHS=60 EXTRA=--no-early-stopping
make aggregate RUN_TAG=e2_aug_s42 VARIANT=last
```

You can also run the module directly:

```bash
python -m cadica.cli.train --fold 1 --epochs 10
```

Or through Make:

```bash
make train FOLD=1 EPOCHS=10
```

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

- **Seed:** `42` for fold generation and training runs.
- **Split unit:** patient. `cadica-verify-folds` asserts that train and validation patient
  sets are disjoint for every fold.
- **Sampling:** `round(i * (n - 1) / (k - 1))` for `i = 0..k-1`, where `k = min(8, n)`.
  The first and last frames are always included.
- **Manifest paths:** relative to `CADICA_DATA_ROOT/selectedVideos`, never absolute.

See [`docs/dataset.md`](docs/dataset.md) and [`docs/pipeline.md`](docs/pipeline.md) for
details. Design rationale, the change log, and the experiment registry live in
[`docs/experiments.md`](docs/experiments.md).

---

## Roadmap

- [x] Three-class training and evaluation CLI with per-class metrics, confusion matrix,
      and checkpointing.
- [ ] Optional augmentation and additional imbalance-handling strategies.
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
