"""Dataset item tests against real manifest frames."""

from __future__ import annotations

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("PIL")

from cadica.config import LABEL_MAP  # noqa: E402
from cadica.data import CADICADataset  # noqa: E402


def test_dataset_item_shape_and_label(
    manifest: pd.DataFrame, valid_manifest_indices: list[int]
) -> None:
    index = valid_manifest_indices[0]
    dataset = CADICADataset()

    sample = dataset[index]

    assert sample["image"].shape == (3, 224, 224)
    assert sample["image"].dtype == torch.float32
    assert sample["label"].dtype == torch.long
    assert int(sample["label"]) == LABEL_MAP[manifest.iloc[index]["severity"]]
    assert sample["patient_id"] == manifest.iloc[index]["patient_id"]


def test_dataset_length_matches_manifest(
    manifest: pd.DataFrame, valid_manifest_indices: list[int]
) -> None:
    dataset = CADICADataset()
    assert len(dataset) == len(manifest)


def test_manifest_paths_are_portable(manifest: pd.DataFrame) -> None:
    # Manifests should use relative paths so they work across machines.
    assert not any(
        str(path).startswith("/") for path in manifest["image_path"].head(50)
    )
