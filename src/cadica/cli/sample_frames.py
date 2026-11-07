"""Sample evenly spaced frames from every CADICA video."""

from __future__ import annotations

from cadica.config import SAMPLED_FRAMES_PATH
from cadica.data.sampling import build_sampled_frames


def main() -> None:
    """Build and save the frame-level sampling manifest."""
    sampled = build_sampled_frames()

    SAMPLED_FRAMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(SAMPLED_FRAMES_PATH, index=False)

    videos = sampled[["patient_id", "video_id"]].drop_duplicates()

    print(f"Total sampled frames: {len(sampled)}")
    print(f"Unique patients: {sampled['patient_id'].nunique()}")
    print(f"Unique videos: {len(videos)}")

    print("\nFrames per fold:")
    print(sampled.groupby("fold").size().to_string())

    print("\nFrames per severity:")
    print(sampled.groupby("severity").size().to_string())

    print(f"\nSaved {SAMPLED_FRAMES_PATH}")


if __name__ == "__main__":
    main()
