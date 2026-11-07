"""Frame-sampling tests."""

from __future__ import annotations

import pytest

from cadica.data.sampling import evenly_spaced_indices


def test_empty_video_returns_no_indices() -> None:
    assert evenly_spaced_indices(0) == []


def test_single_frame_returns_first_index() -> None:
    assert evenly_spaced_indices(1) == [0]


def test_fewer_frames_than_requested_returns_all() -> None:
    assert evenly_spaced_indices(3, num_frames=8) == [0, 1, 2]


def test_evenly_spaced_indices_include_endpoints() -> None:
    indices = evenly_spaced_indices(10, num_frames=8)

    assert len(indices) == 8
    assert indices[0] == 0
    assert indices[-1] == 9
    assert indices == sorted(set(indices))


def test_single_requested_frame_returns_first_index() -> None:
    assert evenly_spaced_indices(100, num_frames=1) == [0]


@pytest.mark.parametrize("n_frames", [2, 5, 17, 250])
def test_indices_stay_in_bounds(n_frames: int) -> None:
    indices = evenly_spaced_indices(n_frames, num_frames=8)

    assert all(0 <= index < n_frames for index in indices)
