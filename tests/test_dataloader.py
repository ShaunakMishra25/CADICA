"""DataLoader collation tests."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from torch.utils.data import DataLoader, Subset  # noqa: E402

from cadica.data import CADICADataset  # noqa: E402


def test_dataloader_batch_shapes(valid_manifest_indices: list[int]) -> None:
    if len(valid_manifest_indices) < 2:
        pytest.skip("Need at least two readable frames")

    dataset = CADICADataset()
    subset = Subset(dataset, valid_manifest_indices[:4])
    loader = DataLoader(subset, batch_size=2, shuffle=False)

    batch = next(iter(loader))

    assert batch["image"].shape[0] == 2
    assert batch["image"].shape[1:] == (3, 224, 224)
    assert batch["label"].shape == (2,)
    assert batch["label"].dtype == torch.long
    assert len(batch["patient_id"]) == 2
