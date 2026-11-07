"""Plot consecutive-frame differences for one angiographic video."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from cadica.config import RESULTS_DIR, SELECTED_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient", default="p1", help="Patient folder, e.g. p1")
    parser.add_argument("--video", default="v1", help="Video folder, e.g. v1")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: results/frame_differences_<patient>_<video>.png)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    video_dir = SELECTED_ROOT / args.patient / args.video / "input"
    frames = sorted(video_dir.glob("*.png"))
    if len(frames) < 2:
        raise SystemExit(f"Need at least two frames in {video_dir}")

    differences = []
    for first, second in zip(frames, frames[1:]):
        img1 = np.array(Image.open(first).convert("L"), dtype=np.float32)
        img2 = np.array(Image.open(second).convert("L"), dtype=np.float32)
        differences.append(float(np.mean(np.abs(img1 - img2))))

    output = args.output or (
        RESULTS_DIR
        / f"frame_differences_{args.patient}_{args.video}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.plot(range(1, len(frames)), differences)
    plt.xlabel("Frame number")
    plt.ylabel("Mean absolute pixel difference")
    plt.title(f"Consecutive frame differences — {args.patient}/{args.video}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()

    print("Average difference:", float(np.mean(differences)))
    print("Minimum difference:", float(np.min(differences)))
    print("Maximum difference:", float(np.max(differences)))
    print("Saved plot:", output)


if __name__ == "__main__":
    main()
