# Data pipeline

The pipeline turns the raw CADICA distribution into leakage-free, frame-level manifests.
Every stage is deterministic and can be rerun safely.

## Stage 1 — Build patient-level folds

```bash
cadica-build-folds
```

Reads `metadata.xlsx` and the frame folders, builds a patient index, and assigns each
patient to one of five severity-stratified folds.

- **Input:** `CADICA_DATA_ROOT/metadata.xlsx`, `CADICA_DATA_ROOT/selectedVideos/`
- **Output:** `data/folds/patient_folds.csv`
- **Seed:** `42`
- **Splitter:** `sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True)`

## Stage 2 — Sample frames

```bash
cadica-sample-frames
```

Walks every patient and video folder and selects up to eight evenly spaced frames per
video.

- **Input:** `data/folds/patient_folds.csv`, raw frames
- **Output:** `data/manifests/sampled_frames.csv`
- **Columns:** `patient_id`, `video_id`, `image_path`, `severity`, `fold`
- **Path convention:** `image_path` is relative to `selectedVideos/`

## Stage 3 — Prepare fold splits

```bash
cadica-prepare-folds
```

For each fold `k`, the fold itself becomes validation and the remaining four folds become
training.

- **Input:** `data/manifests/sampled_frames.csv`
- **Outputs:** `data/folds/train_fold_1..5.csv`, `data/folds/val_fold_1..5.csv`
- **Invariant:** train and validation patient sets are disjoint.

## Stage 4 — Verify integrity

```bash
cadica-verify-folds
```

Checks that:

- every patient appears in exactly one fold;
- all five folds are present;
- train/validation patient sets are disjoint for every fold;
- each validation split contains only its own fold;
- train plus validation covers every patient.

The command raises `AssertionError` on failure, so it can gate CI or a training run.

## Stage 5 — Inspect class balance

```bash
cadica-inspect-folds
```

Prints the patient-level class distribution and a fold-by-severity crosstab.

## Output summary

| File                                   | Rows (approx.) | Committed? |
| -------------------------------------- | -------------: | ---------- |
| `data/folds/patient_folds.csv`         |             42 | yes        |
| `data/manifests/sampled_frames.csv`    |          3,042 | yes        |
| `data/folds/train_fold_*.csv`          |         ~2,400 | no         |
| `data/folds/val_fold_*.csv`            |           ~600 | no         |

Generated split files are ignored by Git because they are fully reproducible with
`make pipeline`. The small patient fold table and the relative-path sampling manifest are
tracked for convenience.

## Stage 6 — Train a fold

```bash
cadica-train --fold 1 --epochs 10 --batch-size 16
```

The trainer reads the fold manifests, balances classes and patients with a
`WeightedRandomSampler`, and evaluates at the patient level by averaging frame
probabilities per patient.

- **Inputs:** `data/folds/train_fold_<k>.csv`, `data/folds/val_fold_<k>.csv`
- **Outputs:**
  - `results/training_log_fold_<k>.csv`
  - `results/patient_predictions_fold_<k>.csv`
  - `checkpoints/baseline_fold_<k>.pt`

Reported metrics: patient-level accuracy, balanced accuracy, macro F1, confusion matrix,
and a per-class classification report.

Equivalent module invocation:

```bash
python -m cadica.cli.train --fold 1 --epochs 10
```

## Design notes

- **Patient-level evaluation.** Frame-level predictions are averaged within each patient
  before metrics are computed, which matches the clinical unit of analysis and avoids
  letting patients with many frames dominate.
- **Balanced sampling.** Each frame is weighted by the inverse of its patient's frame
  count and its class's patient count, so classes and patients contribute equally during
  training.
- **Reproducibility.** Training uses the global seed in `cadica.config`; outputs are
  written to Git-ignored `results/` and `checkpoints/` directories.

