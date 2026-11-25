"""Tests for the training and evaluation transform pipelines."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")
PIL = pytest.importorskip("PIL")

from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402

from cadica.data.transforms import (  # noqa: E402
    AddGaussianNoise,
    PerImageStandardize,
    build_eval_transform,
    build_train_transform,
)


def make_image(size: int = 512) -> Image.Image:
    return Image.new("RGB", (size, size), color=(40, 40, 40))


def make_gradient(size: int = 64) -> Image.Image:
    image = Image.new("L", (size, size))
    image.putdata([(x * 255) // (size - 1) for x in range(size)] * size)
    return image.convert("RGB")


def test_eval_transform_is_deterministic_and_sized() -> None:
    transform = build_eval_transform()

    first = transform(make_image())
    second = transform(make_image())

    assert first.shape == (3, 224, 224)
    assert first.dtype == torch.float32
    assert torch.equal(first, second)


def test_train_transform_preserves_shape_and_range() -> None:
    transform = build_train_transform()

    image = transform(make_image())

    assert image.shape == (3, 224, 224)
    assert image.dtype == torch.float32
    assert float(image.min()) >= 0.0
    assert float(image.max()) <= 1.0


def test_train_transform_varies_between_calls() -> None:
    # Augmentation is random, so repeated calls on one image should differ.
    transform = build_train_transform()

    first = transform(make_image())
    second = transform(make_image())

    assert not torch.equal(first, second)


def test_gaussian_noise_changes_tensor_within_range() -> None:
    tensor = torch.full((3, 8, 8), 0.5)
    noisy = AddGaussianNoise(std=0.1, p=1.0)(tensor)

    assert noisy.shape == tensor.shape
    assert not torch.equal(noisy, tensor)
    assert float(noisy.min()) >= 0.0
    assert float(noisy.max()) <= 1.0


def test_gaussian_noise_p_zero_is_identity() -> None:
    tensor = torch.rand(3, 8, 8)
    assert torch.equal(AddGaussianNoise(std=0.1, p=0.0)(tensor), tensor)


def test_gaussian_noise_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        AddGaussianNoise(std=-0.1)
    with pytest.raises(ValueError):
        AddGaussianNoise(p=1.5)


def test_horizontal_flip_is_opt_in() -> None:
    # Default pipeline has no flip; the composed transform should still build.
    assert isinstance(build_train_transform(), transforms.Compose)
    assert isinstance(build_train_transform(horizontal_flip_p=0.5), transforms.Compose)


def test_per_image_standardize_removes_mean_and_scale() -> None:
    tensor = torch.rand(3, 16, 16) * 100 + 50
    out = PerImageStandardize()(tensor)

    assert torch.allclose(out.mean(dim=(1, 2)), torch.zeros(3), atol=1e-5)
    assert torch.allclose(out.std(dim=(1, 2), unbiased=False), torch.ones(3), atol=1e-5)


def test_intensity_norm_applies_to_eval_transform() -> None:
    transform = build_eval_transform(intensity_norm="standardize")
    out = transform(make_gradient())

    assert out.shape == (3, 224, 224)
    assert float(out.mean()) == pytest.approx(0.0, abs=1e-5)
    assert float(out.std(unbiased=False)) == pytest.approx(1.0, abs=1e-4)


def test_intensity_norm_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError):
        build_eval_transform(intensity_norm="clahe")
    with pytest.raises(ValueError):
        build_train_transform(intensity_norm="clahe")
