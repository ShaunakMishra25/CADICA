"""Inspect image dimensions and formats across the dataset."""

from __future__ import annotations

from collections import Counter

from PIL import Image

from cadica.data.loader import collect_image_paths


def main() -> None:
    images = collect_image_paths()

    dimensions: Counter[tuple[int, int]] = Counter()
    formats: Counter[str] = Counter()

    for path in images["image_path"]:
        with Image.open(path) as img:
            dimensions[img.size] += 1
            formats[img.format] += 1

    print("Total images:", len(images))

    print("\nImage dimensions:")
    for size, count in dimensions.most_common():
        print(size, ":", count)

    print("\nImage formats:")
    print(formats)


if __name__ == "__main__":
    main()
