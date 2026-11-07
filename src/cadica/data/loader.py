"""Loading and indexing utilities for the raw CADICA dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cadica.config import (
    METADATA_PATH,
    SELECTED_ROOT,
    TARGET_COL,
)


def load_metadata(metadata_path: Path | str = METADATA_PATH) -> pd.DataFrame:
    """Load the CADICA metadata workbook and normalise patient identifiers.

    The workbook uses integer patient IDs while the frame folders use ``p<id>``
    (for example ``p1``). The returned frame adds a ``patient_id`` column in the
    folder format.
    """
    metadata = pd.read_excel(metadata_path)
    metadata["patient_id"] = metadata["Patient ID"].apply(lambda x: f"p{int(x)}")
    return metadata


def collect_image_paths(
    selected_root: Path | str = SELECTED_ROOT,
) -> pd.DataFrame:
    """Collect every selected angiographic frame and its patient ID.

    Returns a frame with ``image_path`` and ``patient_id`` columns. Paths are
    absolute so they can be opened directly.
    """
    selected_root = Path(selected_root)
    image_paths = sorted(selected_root.glob("p*/v*/input/*.png"))

    return pd.DataFrame(
        {
            "image_path": [str(path) for path in image_paths],
            "patient_id": [path.parents[2].name for path in image_paths],
        }
    )


def normalize_severity(label: object) -> str:
    """Normalise a raw severity label to one of ``<20%``, ``20-50%``, ``>70%``.

    Raises
    ------
    ValueError
        If the label does not match a supported CADICA severity category.
    """
    cleaned = str(label).strip().replace(" ", "")

    if cleaned in {"<20%", "20-50%", ">70%"}:
        return cleaned

    raise ValueError(f"Unknown severity label: {label!r}")


def build_patient_index(
    metadata_path: Path | str = METADATA_PATH,
    selected_root: Path | str = SELECTED_ROOT,
) -> pd.DataFrame:
    """Build one row per patient with image counts and severity labels.

    Only patients that appear in both the metadata workbook and the frame
    folders are retained (inner join).
    """
    metadata = load_metadata(metadata_path)
    images = collect_image_paths(selected_root)

    image_counts = (
        images.groupby("patient_id").size().reset_index(name="image_count")
    )

    patients = image_counts.merge(
        metadata[["patient_id", TARGET_COL]],
        on="patient_id",
        how="inner",
    )

    patients["severity"] = patients[TARGET_COL].apply(normalize_severity)

    return patients[["patient_id", "image_count", "severity"]].reset_index(drop=True)
