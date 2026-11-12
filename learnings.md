# CADICA Severity Classification — Learnings and Roadmap

Last updated: 2026-11-12

This document records what has been built so far, what has been verified, what was learned,
and what should be improved next. It is both a project notebook and a practical TODO list.

---

## 1. Project goal

Build a reproducible, patient-level benchmark for classifying coronary artery stenosis
severity from CADICA angiographic video frames.

- **Input:** selected angiographic frames from the CADICA dataset
- **Output:** one of three severity classes per patient
- **Unit of analysis:** patient, not frame
- **Framework:** PyTorch + torchvision
- **Evaluation:** patient-level cross-validation metrics

The project deliberately avoids frame-level random splits because frames from the same
patient and video are highly correlated. A random frame split would leak patient identity
and inflate results.

---

## 2. Dataset facts

Verified against the current manifests and raw dataset:

| Item | Value |
|---|---:|
| Patients | 42 |
| Angiographic videos | 382 |
| Sampled frames | 3,042 |
| Cross-validation folds | 5 |
| Frames sampled per video | up to 8 |
| Image size | 512 × 512 |
| Image type | 8-bit grayscale PNG |
| Label source | `metadata.xlsx` |
| Target column | `Maximum degree of the coronary artery involvement` |

Class distribution:

| Severity | Class index | Patients | Frames |
|---|---:|---:|---:|
| `<20%` | 0 | 6 | 384 |
| `20-50%` | 1 | 8 | 448 |
| `>70%` | 2 | 28 | 2,210 |

Key observations:

- The cohort is strongly imbalanced: `>70%` accounts for 28/42 patients.
- The minority class has only 6 patients total, so a single patient is about 17% of that
  class.
- Each validation fold contains 8–9 patients and only 1–2 patients from each minority
  class. Single-fold metrics are therefore high variance.
- Metadata labels appear in slightly different forms, for example `> 70%`, `>70%`,
  `< 20%`, and `<20%`. The loader normalizes whitespace correctly.

---

## 3. What has been implemented

### 3.1 Configuration

File: `src/cadica/config.py`

- Central project paths and dataset paths
- `CADICA_DATA_ROOT` environment-variable override
- Label map and class names
- Experiment defaults: 8 frames per video, seed 42, 5 folds
- Helpers for train/validation fold manifest paths

### 3.2 Raw-data indexing

File: `src/cadica/data/loader.py`

- `load_metadata()` reads `metadata.xlsx` and converts integer patient IDs to folder IDs
  such as `p1`.
- `collect_image_paths()` finds frames under `selectedVideos/p*/v*/input/*.png`.
- `normalize_severity()` converts raw labels into `<20%`, `20-50%`, or `>70%`.
- `build_patient_index()` joins patient image counts with metadata severity labels.

### 3.3 Deterministic frame sampling

File: `src/cadica/data/sampling.py`

- `evenly_spaced_indices()` selects up to 8 frames per video.
- First and last frames are included when possible.
- Sampling is deterministic and reproducible.
- `build_sampled_frames()` produces the frame-level manifest with relative image paths.
- Manifest columns: `patient_id`, `video_id`, `image_path`, `severity`, `fold`.

### 3.4 PyTorch dataset

File: `src/cadica/data/dataset.py`

- `CADICADataset` reads a manifest CSV.
- Relative paths are resolved against `SELECTED_ROOT`, so manifests are portable.
- Images are converted to RGB and resized to 224 × 224.
- Returns a dictionary with `image`, `label`, and `patient_id`.
- Validates required manifest columns.
- Rejects unknown severity labels.

### 3.5 Fold construction and verification

Files:

- `src/cadica/cli/build_folds.py`
- `src/cadica/cli/sample_frames.py`
- `src/cadica/cli/prepare_folds.py`
- `src/cadica/cli/verify_folds.py`
- `src/cadica/cli/inspect_folds.py`

Implemented behavior:

- Patient-level stratified 5-fold split using severity labels.
- One fold becomes validation; the other four become training.
- Train/validation patient sets are checked to be disjoint.
- Fold coverage and exact fold membership are verified.
- Class balance per fold is printed for inspection.

Verified state:

- 42 unique patients assigned to exactly one fold.
- No patient appears in more than one fold.
- No patient appears in both training and validation for any fold.
- All five folds are present and cover every sampled patient.
- No duplicate image paths and no missing image files.

### 3.6 Baseline CNN

File: `src/cadica/models/cnn.py`

`CADICACNN` is a small VGG-style convolutional network:

- Four convolution blocks: 32 → 64 → 128 → 256 channels
- 3 × 3 convolutions, ReLU activations, 2 × 2 max pooling
- Global average pooling
- Single linear classification head
- Three output logits

This is intentionally a baseline, not a state-of-the-art architecture.

### 3.7 Training and evaluation

File: `src/cadica/cli/train.py`

Implemented:

- CLI arguments for fold, epochs, batch size, learning rate, workers, and device
- Reproducibility seeding
- Patient- and class-balanced `WeightedRandomSampler`
- Cross-entropy loss and Adam optimizer
- Per-epoch training loss and validation metrics
- Patient-level evaluation by averaging frame-level softmax probabilities per patient
- Accuracy, balanced accuracy, macro F1, confusion matrix, and per-class report
- Saved training log, patient predictions, and model checkpoint

Default training configuration:

| Setting | Value |
|---|---:|
| Epochs | 25 |
| Batch size | 16 |
| Learning rate | 1e-3 |
| Optimizer | Adam |
| Loss | Cross-entropy |
| Sampler | `WeightedRandomSampler` |
| Classes | 3 |

Outputs:

- `results/training_log_fold_<k>.csv`
- `results/patient_predictions_fold_<k>.csv`
- `checkpoints/baseline_fold_<k>.pt`

### 3.8 Tests and documentation

Implemented:

- Pytest suite covering config, loader, sampling, folds, dataset, dataloader, model, and
  training helpers
- README with installation and usage instructions
- `docs/dataset.md` and `docs/pipeline.md`
- `data/README.md` explaining tracked versus generated artifacts
- Diagnostic scripts in `scripts/`

The manifests themselves are clean:

- No null values
- No duplicate image paths
- No missing files
- Consistent labels and fold assignments

---

## 4. Key lessons learned

1. **Split by patient, never by frame.**
   Frames from the same patient and video are correlated. Patient-level folds are the
   minimum requirement for a meaningful result.

2. **Evaluate at the patient level.**
   Averaging frame probabilities per patient matches the clinical unit of analysis and
   prevents patients with many frames from dominating the metrics.

3. **Balance both classes and patients.**
   Weighting only by class would still let patients with many frames dominate. The
   sampler correctly gives each patient equal total weight within its class.

4. **Store paths relative to the dataset root.**
   Relative manifest paths make the manifests portable across machines and dataset
   locations. Absolute paths would make the repository machine-specific.

5. **Manifests are the contract.**
   The pipeline works because every stage reads and writes an explicit CSV with a known
   schema. This makes debugging, verification, and regeneration much easier.

6. **Do not trust default paths.**
   Hardcoded fallback paths are convenient during development but break on other
   machines. Environment variables and clear errors are safer.

7. **Always verify leakage explicitly.**
   The fold verifier is valuable because it turns an assumption into an executable check.
   The same idea should be extended to labels, image existence, and patient coverage.

8. **Single-fold results are noisy.**
   With only 8–9 validation patients and 1–2 minority-class patients per fold, report
   mean and standard deviation across all folds.

9. **Composite keys matter.**
   `video_id` such as `v1` is only unique within a patient. Any video-level count must
   use `(patient_id, video_id)`.

10. **Medical-image augmentation needs care.**
    Generic augmentations such as horizontal flips may not be clinically valid for
    coronary angiography. Prefer geometry-preserving or intensity-based augmentations
    unless a flip is explicitly justified.

---

## 5. Known issues to fix

### High priority

1. **Default dataset path is broken on this machine.**
   `config.py` falls back to
   `/run/media/shaunakmishra25/DATA/Coronary Research/...`, but the current dataset lives
   under `/run/media/shaunak/DATA/Coronary Research/...`.
   Without `CADICA_DATA_ROOT`, dataset loading and pipeline regeneration fail.

2. **Video counts in `verify_folds.py` are undercounted.**
   `groupby("fold")["video_id"].nunique()` treats `v1` from different patients as the
   same video. Count unique `(patient_id, video_id)` pairs instead.

3. **`evenly_spaced_indices(..., num_frames=0)` returns `[0]`.**
   The `count <= 1` branch should not apply when zero frames are requested. Return `[]`
   for `num_frames <= 0`.

4. **`build_sampled_frames()` can silently drop patients.**
   If a patient in the fold table has no video frames, that patient disappears from the
   manifest. Add a check that every fold-table patient is represented in the sampled
   manifest.

5. **`collect_image_paths()` has no empty-data guard.**
   An empty result leads to a confusing `KeyError` in `build_patient_index()`.
   Raise a clear error when no frames are found.

### Medium priority

6. **Dataset validation is lazy.**
   `CADICADataset` only detects unknown labels when the offending row is accessed.
   Validate all labels during `__init__` for fail-fast behavior.

7. **Image existence is not checked at dataset construction.**
   Missing files surface as PIL errors during training. Add an optional path check or a
   clearer error message.

8. **`PROJECT_ROOT` assumes an editable install.**
   A regular `pip install .` would place `data/`, `results/`, and `checkpoints/` inside
   `site-packages`. Add a `CADICA_OUTPUT_ROOT` environment variable or document that the
   project must be installed with `pip install -e`.

9. **No separate train/eval transform.**
   `CADICADataset` accepts one transform, and the CLI uses the same transform for both
   splits. This is fine for the baseline but blocks augmentation later.

10. **No data augmentation or normalization.**
    The current transform is only resize + tensor. That is acceptable for a from-scratch
    CNN, but pretrained models will require ImageNet normalization and a proper training
    transform.

### Low priority

11. **`default_transform()` type hint is too narrow.**
    Accept any callable transform, not only `transforms.Compose`.

12. **Metadata loading is not defensive.**
    Missing columns, blank patient IDs, and duplicate patient IDs should produce explicit
    errors instead of pandas exceptions or silent overwrites.

13. **The label set is ordinal but training treats it as nominal.**
    Cross-entropy is a reasonable baseline, but severity has a natural order. Consider
    ordinal-aware loss or ordinal metrics later.

14. **Test coverage has edge-case gaps.**
    Missing tests include: missing manifest columns, unknown severity, empty manifest,
    `num_frames=0`, absolute versus relative paths, patient coverage between fold table
    and manifest, and the pair-key video count.

---

## 6. Improvement roadmap

### Phase 1 — Fix correctness and reliability

- [ ] Resolve or remove the hardcoded dataset fallback path.
- [ ] Add a clear error when `CADICA_DATA_ROOT` is missing.
- [ ] Fix `num_frames=0` behavior.
- [ ] Fix per-fold video counting to use `(patient_id, video_id)`.
- [ ] Add patient coverage assertions in the sampling pipeline.
- [ ] Add empty-frame guards in `collect_image_paths()`.
- [ ] Validate all dataset labels during initialization.
- [ ] Add optional manifest path-existence validation.

### Phase 2 — Strengthen engineering quality

- [ ] Add the missing edge-case tests listed above.
- [ ] Add a manifest schema validator shared by training and evaluation.
- [ ] Add a `root` parameter to `CADICADataset` so it is not tied to global config.
- [ ] Separate training and evaluation transforms.
- [ ] Add `pin_memory` and configurable `persistent_workers` to the DataLoader.
- [ ] Save the best checkpoint, not only the final epoch.
- [ ] Add early stopping and a learning-rate scheduler.
- [ ] Add an aggregate cross-validation report: mean ± std for every metric.
- [ ] Make training fully deterministic on GPU, including cuDNN settings.
- [ ] Add mixed-precision training for GPU efficiency.
- [ ] Add a single command that runs all five folds and aggregates results.

### Phase 3 — Improve modeling

- [ ] Compare the baseline CNN against a pretrained ResNet/EfficientNet backbone.
- [ ] Add medically appropriate augmentation:
      - small rotations
      - translation/zoom
      - brightness and contrast variation
      - mild Gaussian noise
      - avoid unjustified horizontal flips
- [ ] Compare `WeightedRandomSampler` with class-weighted cross-entropy.
- [ ] Try focal loss for the severe imbalance.
- [ ] Add ordinal-aware modeling or metrics for severity.
- [ ] Add calibration analysis and threshold tuning.
- [ ] Add video-level aggregation and compare patient-level versus video-level metrics.
- [ ] Add Grad-CAM or similar explainability visualizations.
- [ ] Investigate whether vessel/lesion localization or segmentation features help.

### Phase 4 — Strengthen experimental rigor

- [ ] Report all five folds with mean ± std, not a single fold.
- [ ] Use repeated stratified cross-validation or nested CV for more stable estimates.
- [ ] Report per-class precision, recall, F1, and support for every fold.
- [ ] Report quadratic weighted kappa or ordinal MAE because severity is ordered.
- [ ] Add confidence intervals or bootstrap estimates for patient-level metrics.
- [ ] Keep a fixed test protocol when comparing models.
- [ ] Record hardware, library versions, seed, and commit hash with every experiment.
- [ ] Add a results table comparing all baselines on the same folds.

### Phase 5 — Productization and reproducibility

- [ ] Add a `make experiment` target that trains all folds and writes a summary CSV.
- [ ] Add a results aggregation script.
- [ ] Add a clean environment setup script or lockfile.
- [ ] Add CI that runs linting and the non-GPU test suite.
- [ ] Document the exact metadata label normalization rules.
- [ ] Add a dataset checksum or version marker.
- [ ] Add a model card and experiment log.

---

## 7. Current repository state

- Dataset pipeline: implemented and verified.
- Dataset implementation: logically sound; edge cases need hardening.
- Baseline CNN: implemented.
- Training CLI: implemented; currently has uncommitted changes.
- Tests: present, but the local virtual environment is currently unusable, so the suite
  could not be run during this review.
- Manifests: clean and consistent with the raw dataset.

Environment note:

```text
/run/media/shaunak/DATA/Coronary Research/.venv/bin/python
```

is not executable on this machine (reported as a `data` file). Recreate the environment
before relying on the existing install:

```bash
cd "/run/media/shaunak/DATA/Coronary Research/ClassificationImplementations"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Also set the dataset root explicitly:

```bash
export CADICA_DATA_ROOT="/run/media/shaunak/DATA/Coronary Research/CADICA a new dataset for coronary artery disease/CADICA/CADICA"
```

---

## 8. Recommended next action

Do Phase 1 first. The pipeline is already conceptually correct; the highest-value work is
making failures explicit instead of silent. After that, add the cross-validation
aggregation and a second model (pretrained backbone) so the baseline can be evaluated
against a stronger reference.
