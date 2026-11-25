# Experiments, decisions, and change log

This file records **why** an approach was chosen over its alternatives, **what** changed in
the code, and **what** was measured. It is the project's memory: `learnings.md` explains
how the system works, this file explains why it is built that way and what we tried.

## Rules

1. Record a decision **before** implementing it, with the alternatives that were rejected.
2. Change **one factor at a time**. If two things change, add two entries.
3. Freeze the evaluation protocol before comparing experiments. If the protocol changes,
   previously reported numbers are no longer comparable — say so explicitly.
4. Never augment or resample the validation split.
5. Report patient-level metrics as **mean ± std across folds**, and across seeds when
   augmentation is on.
6. Append corrections; do not silently rewrite history.

---

## 1. Current protocol

Status: **accepted** — protocol decided 2026-11-24.

| Item | Setting | Status |
|---|---|---|
| Split | Patient-level stratified 5-fold, seed 42 | frozen |
| Unit of analysis | Patient (mean of frame probabilities) | frozen |
| Input | 512×512 grayscale → RGB, resize 224×224 | frozen |
| Eval transform | resize + tensor, deterministic | frozen |
| Train transform | affine ±7°, translate 5%, scale 0.95–1.05, brightness/contrast 0.15, Gaussian noise std 0.02 p=0.5 | accepted |
| Horizontal flip | disabled (p=0.0) | provisional |
| Epochs | 60 maximum; `min_epochs=20` before early stopping may fire | accepted |
| Optimizer | Adam, lr 1e-3, batch 16 | accepted |
| Imbalance handling | `WeightedRandomSampler` (patient- and class-balanced) | frozen for baseline |
| LR schedule | `ReduceLROnPlateau`, `mode=max`, factor 0.3, patience 3, min_lr 1e-6 | accepted |
| Early stopping | patience 10, `min_delta=0.02`, `min_epochs=20`, enabled by default | accepted |
| Monitored metric | patient-level macro F1 | accepted |
| Headline metric | patient-level macro F1 and balanced accuracy, mean ± std | frozen |
| Reporting | **last epoch (fixed budget)** is the headline; **best epoch** is reported as secondary and restores best weights | accepted |

**Important:** the CLI enables early stopping by default, which makes the last epoch
selected on the validation fold. For the fixed-budget headline, run with
`--no-early-stopping`. Separately, `--early-stopping` runs produce the secondary
best-epoch table.

---

## 2. Decision log

### D-001 — Split and evaluate by patient, not by frame

**Status:** accepted
**Context:** Frames from one patient and video are highly correlated.
**Decision:** Patient-level stratified 5-fold CV; patient-level metrics.
**Alternatives rejected:** random frame split; video-level split.
**Why rejected:** frame splits leak patient identity and inflate results; video-level splits
still put the same patient on both sides.
**Consequences:** validation folds contain 8–9 patients and only 1–2 patients from each
minority class, so single-fold metrics are high variance.

### D-002 — On-the-fly augmentation instead of offline upsampling

**Status:** accepted
**Context:** The cohort is imbalanced (6 / 8 / 28 patients; 384 / 448 / 2210 frames).
**Decision:** Augment inside `CADICADataset.__getitem__` on the training split only.
**Alternatives rejected:** pre-generating augmented PNGs until each class reaches N;
hybrid offline + online.
**Why rejected:** offline copies add little new information, inflate disk use, risk
train/validation leakage through augmented duplicates, and go stale whenever the
augmentation policy changes. On-the-fly also gives a different variant each epoch.
**Consequences:** augmentation cannot change the class prior or create new patients. It
reduces overfitting; the sampler remains responsible for the prior. Convergence is slower,
so the epoch budget must be larger.

### D-003 — Horizontal flip disabled by default

**Status:** provisional
**Context:** Angiography encodes coronary laterality (LCA vs RCA).
**Decision:** Ship the flag, default `horizontal_flip_p=0.0`.
**Alternatives rejected:** flip with p=0.5 as a generic augmentation.
**Why rejected:** generic CNN practice does not automatically transfer to anatomy where
left/right is meaningful. The label may depend on it.
**Consequences:** enable only if an ablation improves mean macro F1 across folds and seeds.

### D-004 — Augment training only; keep the evaluation transform fixed

**Status:** accepted
**Context:** Any change at validation time invalidates comparisons.
**Decision:** `build_train_transform()` for train, `build_eval_transform()` everywhere else.
**Alternatives rejected:** augmenting validation as a form of TTA inline.
**Why rejected:** contaminates model selection; TTA belongs in a separate, explicit step.
**Consequences:** monitored validation metrics stay deterministic and comparable.

### D-005 — Keep augmentation mild

**Status:** accepted
**Context:** 42 patients, low data.
**Decision:** rotation ≤10°, translation ≤7%, zoom 0.95–1.05, jitter ≤0.2, noise std
≤0.03.
**Alternatives rejected:** large rotations (>15°), elastic warps, cutout.
**Why rejected:** they distort the vessel geometry the label depends on, and with this
sample size they cost more than they add.
**Consequences:** augmentation is a regularizer here, not a source of new information.

### D-006 — Monitor patient-level macro F1; never train loss

**Status:** accepted
**Context:** Early stopping and LR scheduling need one signal.
**Decision:** Monitor patient-level macro F1 (`mode="max"`).
**Alternatives rejected:** train loss; frame-level validation loss; accuracy.
**Why rejected:** train loss is not comparable across epochs under on-the-fly
augmentation; accuracy is dominated by `>70%` (28/42 patients); frame-level loss ignores
the patient-level evaluation unit.
**Consequences:** the signal is noisy. In fold 1, `20-50%` has a single patient, so one
misclassified patient moves macro F1 by roughly 0.33. Patience and `min_delta` must absorb
that.

### D-007 — Correct imbalance one mechanism at a time

**Status:** accepted
**Context:** Sampler, class-weighted loss, focal loss, and augmentation all push toward the
minority classes.
**Decision:** Keep `WeightedRandomSampler` as the baseline; add class-weighted or focal
loss as a separate experiment, not on top by default.
**Alternatives rejected:** sampler + class weights + augmentation stacked immediately.
**Why rejected:** over-correction can hurt `>70%` recall and is impossible to attribute.
**Consequences:** the ablation matrix in §5 tests each mechanism separately.

### D-008 — Fixed budget for the headline, early stopping as secondary

**Status:** accepted (2026-11-24)
**Context:** Early stopping selects the best epoch on the same validation fold later used
for reporting. Best-of-N selection favors the condition with more variance (augmentation
increases variance), which biases the comparison.
**Decision:** the headline comparison uses a fixed 60-epoch budget with early stopping
**off** and reports the last epoch. Early stopping and best-checkpoint are a second,
clearly labelled analysis; the best-epoch weights are restored before scoring and both
last and best results are written.
**Alternatives rejected:** headline numbers from the best early-stopped epoch.
**Why rejected:** optimistic selection bias, larger here because validation folds have
8–9 patients.
**Consequences:** two result tables. `--no-early-stopping` produces the fixed-budget table;
`--early-stopping` produces the secondary best-epoch table. Aggregation supports both via
`--variant last` and `--variant best`.

### D-009 — Report across folds and seeds

**Status:** accepted
**Context:** One validation fold has 1–2 minority patients; one run is not evidence.
**Decision:** Aggregate all 5 folds, and use 2–3 seeds per condition when augmentation is
on.
**Alternatives rejected:** single-fold development runs as final results.
**Why rejected:** the effect size of augmentation is likely smaller than fold-to-fold and
seed-to-seed noise.
**Consequences:** more compute; a fold-aggregation script is required, not optional.

### D-010 — Normalize intensity per image before training

**Status:** accepted (2026-11-25), to be validated in E7/E8
**Context:** §4.2c shows class-correlated global intensity and contrast: a brightness
threshold beats the model on `>70%` vs rest. Photometric augmentation destroys that
shortcut and the augmented run then fails to fit.
**Decision:** add `--intensity-norm {none,standardize}`. `standardize` applies a per-image
z-score after `ToTensor`, so every frame has zero mean and unit variance.
**Alternatives rejected:** dataset-level mean/std normalization (does not remove per-image
exposure differences); CLAHE or histogram equalization (better for vessel contrast, but
adds an OpenCV/scikit-image dependency and is a larger change).
**Why rejected:** they do not isolate the exposure confound as cleanly, or cost a new
dependency.
**Consequences:** input range changes, so Gaussian-noise strength must be re-tuned;
normalization may also remove clinically meaningful contrast information. This is a
diagnostic ablation, not automatically the final preprocessing.

### D-011 — Test loss-based imbalance handling without the sampler

**Status:** accepted (2026-11-25), to be tested in E9/E10
**Context:** §4.2e shows the balanced sampler over-predicts the rarest class at
convergence; D-007 forbids stacking mechanisms without attribution.
**Decision:** add `--loss {ce,weighted_ce,focal}` and `--sampler {balanced,none}`. Test
weighted CE and focal loss with the sampler **off** against the sampler-only baseline.
Class weights are derived from patient-level counts (6 / 8 / 28), normalized to mean 1.
**Alternatives rejected:** sampler plus class weights (double correction, unidentifiable);
prior correction at inference (complementary, but later).
**Why rejected:** they confound the comparison or require calibrated probabilities first.
**Consequences:** focal loss adds a `gamma` hyperparameter; results must be compared on
macro F1 and per-class recall, not accuracy.

### D-012 — Add ordinal metrics

**Status:** accepted (2026-11-25)
**Context:** severity is ordered, and §4.2b shows the errors are mostly between adjacent
classes. Macro F1 treats `<20%`→`20-50%` like `<20%`→`>70%`.
**Decision:** add quadratic weighted kappa (QWK) and ordinal MAE to fold aggregation.
**Alternatives rejected:** keep macro F1 only.
**Why rejected:** it hides whether a model is improving the clinically meaningful
boundaries.
**Consequences:** aggregation emits QWK and MAE alongside macro F1; macro F1 stays the
primary metric for comparability with E1/E2.

---

## 3. Change log

| Date | Change | Files | Reason | Verification |
|---|---|---|---|---|
| 2026-11-24 | On-the-fly augmentation pipeline with train/eval transforms; Gaussian-noise transform; CLI flags for rotation, translation, jitter, noise, flip | `src/cadica/data/transforms.py` (new), `src/cadica/data/dataset.py`, `src/cadica/data/__init__.py`, `src/cadica/cli/train.py`, `tests/test_transforms.py` (new) | D-002, D-004, D-005 | `pytest`: 44 passed after environment setup; transform tests cover shape, range, randomness, noise, and flip opt-in |
| 2026-11-24 | Added this decision and change log | `docs/experiments.md` (new) | Requested project memory | n/a |
| 2026-11-24 | Validation per-fold class support measured: every fold has all three classes; minority support is 1–2 patients | n/a (analysis) | Justify D-006 and D-009 | Recomputed from `data/folds/val_fold_*.csv` |
| 2026-11-24 | `ReduceLROnPlateau` on patient-level macro F1, early stopping with `min_epochs`, best-state tracking, dual last/best evaluation, per-run tags, seed and protocol provenance | `src/cadica/cli/train.py` | D-006, D-008, D-009 | `pytest`: 44 passed; 5-fold smoke run wrote last/best predictions and checkpoints |
| 2026-11-24 | Fold/seed aggregation CLI with mean ± std and per-class recall; `cadica-aggregate` entry point and Make targets | `src/cadica/cli/aggregate.py` (new), `pyproject.toml`, `Makefile`, `tests/test_aggregate.py` (new), `tests/test_train.py` | D-009 | `pytest`: 44 passed; `cadica-aggregate` produced per-fold and mean ± std tables for `--variant last` and `--variant best` |
| 2026-11-24 | Reproducible environment at `~/.venvs/cadica` (Python 3.12 via `uv`) because the old project `.venv` on NTFS is corrupted and the system Python 3.14 has no pip | outside the repo | Unblock running experiments | see §9 |
| 2026-11-24 | Dataset root fallback corrected to the path that exists on this machine (`/run/media/shaunak/...`); `CADICA_DATA_ROOT` still overrides | `src/cadica/config.py` | `learnings.md` known issue 1; otherwise training cannot find metadata or frames | path verified to contain `metadata.xlsx` and `selectedVideos/` |
| 2026-11-24 | Environment verified: CUDA available, full test suite green, 5-fold smoke run end-to-end, aggregation for both variants | outside the repo | Close environment setup | torch 2.15.0+cu132, torchvision 0.29.2+cu132, RTX 4050 detected; `pytest` 44 passed; ruff clean on `src` and `tests` |
| 2026-11-25 | Ran E1 (no augmentation) and E2 (augmentation): 5 folds, 60 epochs, seed 42, `num_workers=4`, early stopping off; aggregated `last` and `best` variants | outputs ignored under `results/` and `checkpoints/` | D-008, D-009 | 10 fold runs completed; 5/5 folds aggregated per condition; numbers in §4.1 |
| 2026-11-25 | Logged diagnosis of the low scores and added decisions D-010 (per-image intensity normalization), D-011 (loss-based imbalance handling), D-012 (ordinal metrics) | `docs/experiments.md` | Explain E1/E2 before changing code | n/a (analysis of existing runs) |

### Pending changes (not implemented yet)

| Planned change | Reason | Depends on |
|---|---|---|
| Class-weighted / focal loss ablation | D-007 | §5 |
| Pretrained backbone | largest expected gain | separate phase |

---

## 4. Experiment registry

Add one row per condition actually run. Never overwrite a row; add a new one.

| ID | Date | Condition | Folds | Seeds | Protocol | Aug | Other | Mean macro F1 | Mean bal. acc. | Notes |
|---|---|---|---:|---:|---|---|---|---|---|---|
| E0 | legacy | baseline CNN | 1 | 1 | 25 epochs, last | off | sampler | — | — | not comparable after protocol change |
| E1 | 2026-11-25 | no augmentation | 5 | 1 (s42) | 60 epochs, last | off | sampler | 0.411 ± 0.187 | 0.424 ± 0.236 | best-epoch variant 0.650 ± 0.241 |
| E2 | 2026-11-25 | augmentation | 5 | 1 (s42) | 60 epochs, last | on | sampler | 0.360 ± 0.066 | 0.456 ± 0.135 | best-epoch variant 0.538 ± 0.060 |
| E3 | | augmentation + ES | 5 | 2 | ES, best ckpt | on | sampler | | | secondary, selection-biased |
| E4 | | class-weighted loss | 5 | 2 | 60 epochs, last | on | no sampler | | | D-007 |
| E5 | | sampler + class weights | 5 | 2 | 60 epochs, last | on | both | | | over-correction check |
| E6 | | horizontal flip | 5 | 2 | 60 epochs, last | on | flip p=0.5 | | | only if D-003 revisited |

Results must be produced by the aggregation script so the numbers are reproducible.

### E1 vs E2 — seed 42 (2026-11-25)

Five folds, 60 epochs, `WeightedRandomSampler`, `--no-early-stopping`. Command:
`make train-all RUN_TAG=<tag> EPOCHS=60 EXTRA="--no-early-stopping [--no-augment]"`,
then `cadica-aggregate`.

**Headline — last epoch, fixed budget (unbiased):**

| Metric | E1 no aug | E2 aug | Difference |
|---|---:|---:|---:|
| Accuracy | 0.544 ± 0.133 | 0.525 ± 0.068 | −0.019 |
| Balanced accuracy | 0.424 ± 0.236 | 0.456 ± 0.135 | +0.032 |
| Macro F1 | 0.411 ± 0.187 | 0.360 ± 0.066 | −0.051 |

**Secondary — best epoch (selection-biased, do not headline):**

| Metric | E1 no aug | E2 aug |
|---|---:|---:|
| Accuracy | 0.758 ± 0.146 | 0.642 ± 0.091 |
| Balanced accuracy | 0.696 ± 0.223 | 0.644 ± 0.080 |
| Macro F1 | 0.650 ± 0.241 | 0.538 ± 0.060 |

**Per-class recall, last epoch:**

| Class | E1 no aug | E2 aug |
|---|---:|---:|
| `<20%` | 0.30 ± 0.45 | **0.00 ± 0.00** |
| `20-50%` | 0.30 ± 0.45 | 0.80 ± 0.45 |
| `>70%` | 0.67 ± 0.10 | 0.57 ± 0.12 |

Observations:

- **No evidence that augmentation helps.** The headline macro F1 is lower with
  augmentation, while balanced accuracy is slightly higher. With five folds and one seed,
  both differences are inside fold noise.
- **The differences are not trustworthy yet.** Fold-level macro F1 ranges 0.27–0.72 and
  per-class recall std reaches 0.45 because each fold has 1–2 minority patients. D-009
  asks for a second seed before drawing conclusions.
- **Augmentation did not rescue the rarest class.** `<20%` recall fell to 0 in every fold
  at the last epoch, despite balanced sampling.
- **The validation metric is noise-dominated.** Best epochs are scattered between 2 and 20,
  so 60 epochs trains well past the point where the fold metric looked good. A single
  patient flipping changes macro F1 by ~0.33 in folds with one minority patient.
- **Absolute performance is low.** Macro F1 ≈ 0.4 from scratch on 42 patients suggests
  model capacity/pretraining matters more than augmentation strength.

Next actions: run seed 137 for E1 and E2 before accepting or rejecting augmentation; then
compare against a class-weighted or focal loss, and a pretrained backbone.

### 4.2 Diagnosis — why E1/E2 scores are low (2026-11-25)

Evidence gathered from the completed runs, the manifests, and the raw frames. These are
competing explanations, not one proven cause.

**a. Frame labels are weak.** The metadata label is the patient's maximum stenosis across
all vessels, but the training unit is a frame from one video. A `>70%` patient also has
videos of less-diseased vessels, so many of their frames contain no evidence for the
label.

| Quantity | Value |
|---|---:|
| Frames | 3,042 |
| Videos | 382 |
| Patients | 42 |
| Videos per patient (median, range) | 8 (3–16) |
| Frames per patient (median, range) | 64 (17–128) |

Eight evenly spaced frames per video are near-duplicates, so the effective sample is closer
to 382 correlated images from 42 people than to 3,042 independent samples.

**b. The two minority classes are not separable from each other.** Pooled confusion over
all five folds (rows = actual, columns = predicted):

| | E1 no aug | E2 aug |
|---|---|---|
| `<20%` → correct | 2 / 6 | 0 / 6 |
| `20-50%` → correct | 2 / 8 | 6 / 8 |
| `>70%` → correct | 19 / 28 | 16 / 28 |
| `20-50%` predicted as `<20%` | 6 / 8 | 2 / 8 |
| `<20%` predicted as `20-50%` | 3 / 6 | 6 / 6 |

The model behaves as a "severe vs not" detector. The `<20%` vs `20-50%` boundary is where
single-frame visual estimation is also unreliable, and macro F1 penalises that adjacent
error as harshly as a far error.

**c. Global intensity is a shortcut, and augmentation removes it.** Patient-level mean
intensity of the sampled frames:

| Class | Mean intensity | Std |
|---|---:|---:|
| `>70%` | 124.8 | 12.7 |
| `<20%` | 138.5 | 3.8 |
| `20-50%` | 141.8 | 6.4 |

A single brightness threshold reaches **78.6%** on `>70%` vs rest, above the model's 54%
accuracy. Training behaviour differs sharply:

| | E1 no aug | E2 aug |
|---|---:|---:|
| Train loss, epoch 1 → 60 | 1.08 → **0.25** | 1.08 → **0.87** |
| Mean prediction confidence | 0.78 | 0.45–0.50 |

E1 fits the training data and leans on the exposure difference. E2 never fits
(0.87 ≈ barely below `ln 3 = 1.10`), becomes uncertain, and collapses toward `20-50%`.
So E1 vs E2 currently measures "the model uses an exposure shortcut and photometric
augmentation destroys it", not "augmentation helps or hurts".

**d. The epoch budget overshoots the useful range.** Best epochs were 2, 3, 4, 11, 12 (E1)
and 2, 2, 3, 10, 20 (E2). Sixty epochs is far past the point where the fold metric peaked.
E1 fold 1 reached macro F1 1.0 on 9 patients at epoch 11 and fell to 0.72 by epoch 60; that
one fold drives most of the best-epoch mean and its standard deviation.

**e. The balanced sampler over-corrects at convergence.** E1 predicted 12 `<20%` patients
when 6 exist and missed 9 of 28 `>70%` patients. Training makes each class equally likely,
so the implicit prior is 1/3–1/3–1/3 while validation is 6/8/28. This costs accuracy and
majority recall.

**f. The metric is noise-dominated at this sample size.** With 1–2 minority patients per
fold, one patient flipping changes a fold's macro F1 by ~0.11–0.33. A 0.05 difference
between conditions cannot be resolved with five folds and one seed.

Implications: remove the exposure shortcut before judging augmentation; stop relying on
sampling alone for imbalance; shorten the budget or rely on checkpoint selection; add
ordinal metrics; and expect a pretrained backbone to matter more than augmentation tuning.

---

## 5. Planned experiment matrix and run order

1. **Smoke test (fold 1, 1 seed).** Augmented, 60 epochs, ES on. Confirm the loss falls,
   LR reductions fire, ES does not stop before `min_epochs`, and the best checkpoint
   restores correctly.
2. **E1 vs E2, all folds, 2 seeds, fixed budget.** Accept fixed-budget last-epoch as the
   headline (D-008). This answers "does augmentation help?".
3. **E3 secondary.** Same runs with ES enabled, reported separately.
4. **Only after E1/E2:** one imbalance mechanism at a time (E4, E5), then flips (E6) only if
   a reason appears.
5. **Then** move to modeling changes (pretrained backbone), which are expected to matter
   more than any of the above.

Cost note: 5 folds × 2 conditions × 2 seeds × 60 epochs is 20 runs. Run the smoke test
before launching the full matrix.

---

## 6. Guardrails and known biases

- **Selection bias.** Best-epoch or early-stopped numbers are optimistic because the
  validation fold is used for selection. Disclose it; keep a fixed-budget table.
- **Tiny validation sets.** 8–9 patients, 1–2 per minority class. A single patient changes
  macro F1 by ~0.11–0.33 depending on the fold.
- **No held-out test set.** CV validation doubles as model selection. Nested CV or a
  repeated split would be more rigorous; note this as a limitation rather than pretending
  otherwise.
- **Augmentation variance.** Seeds matter more with augmentation on; do not compare
  single runs.
- **Worker count.** `num_workers` changes the RNG stream. Fix it across compared runs.
- **Metric drift.** If the eval transform, fold manifests, or label map changes, previous
  numbers are void. Record it in §3.

---

## 7. Open decisions

| # | Decision | Options | Recommended | Status |
|---|---|---|---|---|
| 1 | Reporting protocol | fixed budget / early stopping / both | both, fixed budget as headline | **resolved:** both, fixed budget headline (D-008) |
| 2 | Restore best weights before final evaluation | yes / no | yes, and report best and last epoch | **resolved:** yes, report both |
| 3 | Horizontal flip | off / p=0.5 | off until ablated | open |
| 4 | Seeds per condition | 1 / 2 / 3 | 2 first, 3 if differences are borderline | open |

---

## 8. Entry templates

New decision:

```text
### D-0NN — <title>
**Status:** proposed | accepted | rejected | superseded by D-0NN
**Context:** <what forced the decision>
**Decision:** <what we do>
**Alternatives rejected:** <what else we considered>
**Why rejected:** <the actual reason>
**Consequences:** <costs, limitations, follow-up work>
```

New change:

```text
| YYYY-MM-DD | <change> | <files> | <decision IDs> | <how it was verified> |
```

New experiment:

```text
| E<ID> | YYYY-MM-DD | <condition> | <folds> | <seeds> | <protocol> | on/off | <other> | x.xx ± x.xx | x.xx ± x.xx | <notes> |
```

---

## 9. Environment

The raw dataset lives on an NTFS mount. A virtual environment created there is unreliable
(the previous `.venv` ended up with a non-executable `bin/python`). The environment is
therefore kept on the btrfs home partition.

| Item | Value |
|---|---|
| Interpreter | CPython 3.12.16, installed by `uv` |
| Environment | `~/.venvs/cadica` |
| Why not system Python | system Python is 3.14 and has no `pip` |
| Why not the old `.venv` | NTFS mount corrupted `bin/python` |
| Install command | `uv pip install --python ~/.venvs/cadica/bin/python -e ".[dev]"` |
| Verified stack | torch 2.15.0+cu132, torchvision 0.29.2+cu132, CUDA available on RTX 4050 Laptop |
| Verification | `pytest`: 44 passed; `ruff check src tests`: clean; 5-fold smoke run completed |

Activate and point at the dataset:

```bash
source ~/.venvs/cadica/bin/activate
export CADICA_DATA_ROOT="/run/media/shaunak/DATA/Coronary Research/CADICA a new dataset for coronary artery disease/CADICA/CADICA"
```

Run the fixed-budget headline protocol (early stopping off):

```bash
cadica-train --fold 1 --epochs 60 --run-tag e2_aug_s42 --no-early-stopping
```

Run the secondary early-stopping protocol:

```bash
cadica-train --fold 1 --epochs 60 --run-tag e3_aug_es_s42
```

Aggregate a run tag after training all five folds:

```bash
cadica-aggregate --run-tag e2_aug_s42 --variant last
cadica-aggregate --run-tag e3_aug_es_s42 --variant best
```

Collect output provenance. Checkpoints store seed, augmentation settings, scheduler
settings, early-stopping settings, completed epochs, and the best epoch.
