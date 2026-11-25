"""PyTorch dataset for sampled CADICA angiographic frames."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from cadica.config import LABEL_MAP, SAMPLED_FRAMES_PATH, SELECTED_ROOT
from cadica.data.transforms import build_eval_transform

_REQUIRED_COLUMNS = {"image_path", "severity", "patient_id"}


def default_transform() -> transforms.Compose:
    """Return the deterministic preprocessing used by the baseline CNN."""
    return build_eval_transform()


def resolve_image_path(image_path: str | Path) -> Path:
    """Resolve a manifest path against the dataset root when it is relative."""
    path = Path(image_path)
    return path if path.is_absolute() else SELECTED_ROOT / path


class CADICADataset(Dataset):
    """Frame-level dataset backed by a sampling manifest CSV.

    Each manifest row must contain ``image_path``, ``severity`` and
    ``patient_id``. Relative image paths are resolved against
    :data:`cadica.config.SELECTED_ROOT`, which keeps manifests portable between
    machines.

    Parameters
    ----------
    csv_path:
        Sampling manifest to load. Defaults to
        :data:`cadica.config.SAMPLED_FRAMES_PATH`.
    transform:
        Optional torchvision transform. Defaults to resize + tensor.
    """

    def __init__(
        self,
        csv_path: str | Path | None = None,
        transform: Callable[[Image.Image], object] | None = None,
    ) -> None:
        self.csv_path = Path(csv_path or SAMPLED_FRAMES_PATH)
        self.data = pd.read_csv(self.csv_path)

        missing = _REQUIRED_COLUMNS - set(self.data.columns)
        if missing:
            raise ValueError(
                f"Manifest {self.csv_path} is missing columns: {sorted(missing)}"
            )

        self.label_map = dict(LABEL_MAP)
        self.transform = transform or default_transform()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.data.iloc[index]

        path = resolve_image_path(row["image_path"])
        with Image.open(path) as image:
            image = image.convert("RGB")
        image = self.transform(image)

        severity = str(row["severity"])
        if severity not in self.label_map:
            raise ValueError(
                f"Unknown severity label {severity!r} at manifest index {index}"
            )

        return {
            "image": image,
            "label": torch.tensor(self.label_map[severity], dtype=torch.long),
            "patient_id": str(row["patient_id"]),
        }
