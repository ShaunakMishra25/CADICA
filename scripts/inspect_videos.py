"""Count angiographic videos per patient."""

from __future__ import annotations

from cadica.config import SELECTED_ROOT


def main() -> None:
    patient_video_counts: dict[str, int] = {}

    for patient_dir in SELECTED_ROOT.glob("p*"):
        if not patient_dir.is_dir():
            continue

        videos = [
            folder
            for folder in patient_dir.iterdir()
            if folder.is_dir() and folder.name.startswith("v")
        ]
        patient_video_counts[patient_dir.name] = len(videos)

    print("Total patients:", len(patient_video_counts))
    print("Total videos:", sum(patient_video_counts.values()))

    print("\nVideos per patient:")
    for patient, count in sorted(
        patient_video_counts.items(),
        key=lambda item: int(item[0][1:]),
    ):
        print(f"{patient}: {count}")


if __name__ == "__main__":
    main()
