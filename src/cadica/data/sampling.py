"""Deterministic frame sampling for CADICA angiographic videos."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cadica.config import (
    NUM_FRAMES,
    PATIENT_FOLDS_PATH,
    SELECTED_ROOT,
)


def evenly_spaced_indices(n_frames: int, num_frames: int = NUM_FRAMES) -> list[int]:
    """Return up to ``num_frames`` evenly spaced frame indices.

    The first and last frames are always included when more than one frame is
    available. If ``n_frames`` is smaller than ``num_frames``, every frame is
    returned.

    Examples
    --------
    >>> evenly_spaced_indices(10, 3)
    [0, 4, 9]
    """
    if n_frames <= 0:
        return []

    count = min(num_frames, n_frames)
    if count <= 1:
        return [0]

    return [
        round(index * (n_frames - 1) / (count - 1)) for index in range(count)
    ]


def build_sampled_frames(
    folds_path: Path | str = PATIENT_FOLDS_PATH,
    selected_root: Path | str = SELECTED_ROOT,
    num_frames: int = NUM_FRAMES,
) -> pd.DataFrame:
    """Sample evenly spaced frames from every video for every patient.

    Image paths are stored relative to ``selected_root`` so the resulting
    manifest is portable across machines.

    Parameters
    ----------
    folds_path:
        Patient-level fold assignment produced by ``cadica-build-folds``.
    selected_root:
        Directory containing ``p*/v*/input/*.png`` frames.
    num_frames:
        Maximum number of frames to sample per video.

    Returns
    -------
    pandas.DataFrame
        Columns: ``patient_id``, ``video_id``, ``image_path``, ``severity``,
        ``fold``.
    """
    folds_path = Path(folds_path)
    selected_root = Path(selected_root)

    patients = pd.read_csv(folds_path)
    required = {"patient_id", "severity", "fold"}
    missing = required - set(patients.columns)
    if missing:
        raise ValueError(
            f"Fold file {folds_path} is missing columns: {sorted(missing)}"
        )

    fold_map = patients.set_index("patient_id")["fold"].to_dict()
    severity_map = patients.set_index("patient_id")["severity"].to_dict()

    records: list[dict[str, object]] = []

    for patient_dir in sorted(selected_root.glob("p*")):
        if not patient_dir.is_dir():
            continue

        patient_id = patient_dir.name
        if patient_id not in severity_map:
            raise ValueError(
                f"Patient folder {patient_id!r} has no metadata/fold assignment"
            )

        for video_dir in sorted(patient_dir.glob("v*")):
            input_dir = video_dir / "input"
            frames = sorted(input_dir.glob("*.png"))
            if not frames:
                continue

            for index in evenly_spaced_indices(len(frames), num_frames):
                frame_path = frames[index]
                records.append(
                    {
                        "patient_id": patient_id,
                        "video_id": video_dir.name,
                        "image_path": frame_path.relative_to(
                            selected_root
                        ).as_posix(),
                        "severity": severity_map[patient_id],
                        "fold": fold_map[patient_id],
                    }
                )

    return pd.DataFrame(
        records,
        columns=["patient_id", "video_id", "image_path", "severity", "fold"],
    )
