"""Data loading, sampling, and PyTorch datasets."""

from cadica.data.dataset import CADICADataset, default_transform, resolve_image_path
from cadica.data.loader import (
    build_patient_index,
    collect_image_paths,
    load_metadata,
    normalize_severity,
)
from cadica.data.sampling import build_sampled_frames, evenly_spaced_indices

__all__ = [
    "CADICADataset",
    "build_patient_index",
    "build_sampled_frames",
    "collect_image_paths",
    "default_transform",
    "evenly_spaced_indices",
    "load_metadata",
    "normalize_severity",
    "resolve_image_path",
]
