# Dataset notes

## Source

This project uses the **CADICA** coronary angiography dataset. The raw data is not
distributed with this repository. Download it from the original distribution channel and
extract it so that `metadata.xlsx` and `selectedVideos/` sit side by side.

```text
CADICA/
├── metadata.xlsx
└── selectedVideos/
    └── p<patient>/v<video>/input/<frame>.png
```

Set `CADICA_DATA_ROOT` to the `CADICA/CADICA` directory that contains both entries.

## Metadata

The workbook is read with `pandas.read_excel`, which requires `openpyxl`.

- **Patient identifier:** `Patient ID` (integer) is normalised to the folder format
  `p<id>`, for example `1` → `p1`.
- **Target column:** `Maximum degree of the coronary artery involvement`.
- **Supported labels:** `<20%`, `20-50%`, `>70%`. Whitespace is normalised before
  validation; any other value raises `ValueError`.

## Class definition

| Severity | Class index | Patients | Frames |
| -------- | ----------: | -------: | -----: |
| `<20%`   |           0 |        6 |    384 |
| `20-50%` |           1 |        8 |    448 |
| `>70%`   |           2 |       28 |  2,210 |

The dataset is imbalanced at the patient level. Report balanced accuracy, macro F1, and
per-class recall alongside overall accuracy.

## Frame sampling

Each angiographic video is represented by at most eight evenly spaced frames:

```text
k = min(8, n_frames)
indices[i] = round(i * (n_frames - 1) / (k - 1)),  i = 0..k-1
```

For a single-frame video the index is `0`. The first and last frames are always selected
when `k > 1`.

Sampling is deterministic, so the manifest can be regenerated exactly. The manifest stores
image paths **relative** to `selectedVideos/`; `CADICADataset` resolves them against
`CADICA_DATA_ROOT`.

## Cohort statistics

- 42 patients
- 382 angiographic videos
- 3,042 sampled frames
- 5 patient-level folds

Per-fold patient counts and class distributions are printed by `cadica-inspect-folds` and
checked by `cadica-verify-folds`.

## Known considerations

- **Patient leakage:** never split frames randomly. Always split by `patient_id`, which is
  what the fold builder does.
- **Class imbalance:** the severe class (`>70%`) dominates. Consider class-weighted loss,
  balanced sampling, or threshold tuning when training.
- **Video correlation:** frames from the same video are highly correlated. Frame-level
  metrics overstate independence; report patient- or video-level aggregates for clinical
  claims.
