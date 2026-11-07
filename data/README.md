# Derived data

This directory contains small, reproducible artifacts derived from the raw CADICA dataset.
The raw dataset itself is **not** stored here; see the repository README for setup.

## Layout

```text
data/
├── folds/
│   ├── patient_folds.csv       # tracked: patient -> fold assignment
│   ├── train_fold_1..5.csv     # generated, ignored
│   └── val_fold_1..5.csv       # generated, ignored
└── manifests/
    └── sampled_frames.csv      # tracked: relative paths + labels + fold
```

## Regenerate everything

```bash
make pipeline
```

## Why some files are ignored

`train_fold_*.csv` and `val_fold_*.csv` are deterministic expansions of
`sampled_frames.csv`. Ignoring them keeps the repository small while `make pipeline`
recreates them byte-for-byte.

`patient_folds.csv` and `sampled_frames.csv` are tracked because they document the exact
cohort and split used by experiments. Image paths in the manifest are relative to
`CADICA_DATA_ROOT/selectedVideos`, so they remain valid on another machine with the same
dataset.
