"""Project-wide configuration, label definitions, and filesystem paths.

The raw CADICA dataset is not stored inside this repository. Point the
``CADICA_DATA_ROOT`` environment variable at the extracted dataset folder; the
default below keeps the original author's local path working out of the box.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
FOLDS_DIR = DATA_DIR / "folds"
MANIFESTS_DIR = DATA_DIR / "manifests"

PATIENT_FOLDS_PATH = FOLDS_DIR / "patient_folds.csv"
SAMPLED_FRAMES_PATH = MANIFESTS_DIR / "sampled_frames.csv"

RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

# ---------------------------------------------------------------------------
# Raw dataset
# ---------------------------------------------------------------------------
_DEFAULT_DATA_ROOT = Path(
    "/run/media/shaunakmishra25/DATA/Coronary Research/"
    "CADICA a new dataset for coronary artery disease/CADICA/CADICA"
)

DATA_ROOT = Path(os.environ.get("CADICA_DATA_ROOT", _DEFAULT_DATA_ROOT)).expanduser()
SELECTED_ROOT = DATA_ROOT / "selectedVideos"
METADATA_PATH = DATA_ROOT / "metadata.xlsx"

#: Column in ``metadata.xlsx`` that stores the stenosis severity label.
TARGET_COL = "Maximum degree of the coronary artery involvement"

# ---------------------------------------------------------------------------
# Task definition
# ---------------------------------------------------------------------------
#: Severity label -> integer class index used by the model.
LABEL_MAP = {"<20%": 0, "20-50%": 1, ">70%": 2}

#: Human-readable class names ordered by class index.
CLASS_NAMES = ["<20%", "20-50%", ">70%"]

NUM_CLASSES = len(CLASS_NAMES)

# ---------------------------------------------------------------------------
# Experiment defaults
# ---------------------------------------------------------------------------
#: Number of evenly spaced frames sampled from every angiographic video.
NUM_FRAMES = 8

#: Global random seed for folds and training.
SEED = 42

#: Number of patient-level cross-validation folds.
N_SPLITS = 5


def train_fold_path(fold: int) -> Path:
    """Return the training manifest path for ``fold``."""
    return FOLDS_DIR / f"train_fold_{fold}.csv"


def val_fold_path(fold: int) -> Path:
    """Return the validation manifest path for ``fold``."""
    return FOLDS_DIR / f"val_fold_{fold}.csv"
