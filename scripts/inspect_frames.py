"""Summarise frame counts per angiographic video."""

from __future__ import annotations

from cadica.config import SELECTED_ROOT


def main() -> None:
    frame_counts: dict[str, int] = {}

    for patient_dir in SELECTED_ROOT.glob("p*"):
        if not patient_dir.is_dir():
            continue

        for video_dir in patient_dir.glob("v*"):
            input_dir = video_dir / "input"
            if input_dir.exists():
                frames = list(input_dir.glob("*.png"))
                frame_counts[f"{patient_dir.name}/{video_dir.name}"] = len(frames)

    counts = list(frame_counts.values())
    if not counts:
        raise SystemExit(f"No frames found under {SELECTED_ROOT}")

    print("Total videos:", len(counts))
    print("Total frames:", sum(counts))
    print("Minimum frames per video:", min(counts))
    print("Maximum frames per video:", max(counts))
    print("Average frames per video:", round(sum(counts) / len(counts), 2))

    print("\nFrame count distribution:")
    for low, high in [(0, 25), (26, 50), (51, 100), (101, 200), (201, 500), (501, 10000)]:
        n = sum(low <= count <= high for count in counts)
        print(f"{low}-{high}: {n} videos")


if __name__ == "__main__":
    main()
