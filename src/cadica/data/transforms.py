"""Training and evaluation transforms for CADICA angiographic frames.

The evaluation transform is deterministic and must be used for validation, test,
and any offline metric. The training transform adds mild, clinically plausible
variation: small rotations and translations, slight zoom, brightness/contrast
jitter, and Gaussian noise.

Horizontal flips are available but **off by default**. Angiography encodes
anatomy with a handedness (left versus right coronary system), so a flip is an
assumption that should be justified by an ablation rather than applied silently.
"""

from __future__ import annotations

import torch
from torchvision import transforms

#: Default input resolution used by the baseline CNN.
IMAGE_SIZE = 224


class AddGaussianNoise:
    """Add zero-mean Gaussian noise to a tensor image.

    torchvision has no built-in noise transform. This runs after ``ToTensor``,
    so ``std`` is expressed in normalised image-intensity units where the valid
    range is ``[0, 1]``.

    Parameters
    ----------
    std:
        Standard deviation of the noise. ``0.02`` is a mild perturbation.
    p:
        Probability of applying noise to a given image.
    """

    def __init__(self, std: float = 0.02, p: float = 0.5) -> None:
        if std < 0:
            raise ValueError(f"std must be non-negative, got {std}")
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.std = float(std)
        self.p = float(p)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.std == 0 or self.p == 0:
            return tensor
        if self.p < 1.0 and torch.rand(1).item() >= self.p:
            return tensor
        noise = torch.randn_like(tensor) * self.std
        return (tensor + noise).clamp_(0.0, 1.0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(std={self.std}, p={self.p})"


def build_train_transform(
    image_size: int = IMAGE_SIZE,
    rotation_degrees: float = 7.0,
    translate: float = 0.05,
    scale: tuple[float, float] = (0.95, 1.05),
    brightness: float = 0.15,
    contrast: float = 0.15,
    gaussian_std: float = 0.02,
    gaussian_p: float = 0.5,
    horizontal_flip_p: float = 0.0,
) -> transforms.Compose:
    """Build the augmentation pipeline used for training frames.

    Parameters
    ----------
    image_size:
        Output resolution (square).
    rotation_degrees:
        Maximum absolute rotation in degrees. Keep this small (5-10) so the
        vessel geometry stays plausible.
    translate:
        Maximum translation as a fraction of image size, e.g. ``0.05`` is 5%.
    scale:
        Zoom range. Values below 1 crop in slightly, above 1 pad out.
    brightness, contrast:
        ``ColorJitter`` magnitudes in ``[0, 1]``.
    gaussian_std, gaussian_p:
        Noise strength and application probability.
    horizontal_flip_p:
        Probability of a horizontal flip. Defaults to ``0.0`` (disabled) because
        flips change coronary laterality. Enable only after an ablation.
    """
    ops: list[object] = []

    if horizontal_flip_p > 0:
        ops.append(transforms.RandomHorizontalFlip(p=horizontal_flip_p))

    ops.extend(
        [
            # Small rotation + translation + zoom in one geometric pass. A black
            # fill matches the dark background of the angiographic frames.
            transforms.RandomAffine(
                degrees=rotation_degrees,
                translate=(translate, translate),
                scale=scale,
                fill=0,
            ),
            transforms.ColorJitter(brightness=brightness, contrast=contrast),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            AddGaussianNoise(std=gaussian_std, p=gaussian_p),
        ]
    )

    return transforms.Compose(ops)


def build_eval_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Build the deterministic transform used for validation and testing."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
