"""Model shape and capacity tests."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cadica.config import NUM_CLASSES  # noqa: E402
from cadica.models import CADICACNN  # noqa: E402


def test_forward_output_shape() -> None:
    model = CADICACNN().eval()
    inputs = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        logits = model(inputs)

    assert logits.shape == (2, NUM_CLASSES)


def test_adaptive_pooling_accepts_other_sizes() -> None:
    model = CADICACNN().eval()
    inputs = torch.randn(1, 3, 96, 96)

    with torch.no_grad():
        logits = model(inputs)

    assert logits.shape == (1, NUM_CLASSES)


def test_model_has_trainable_parameters() -> None:
    model = CADICACNN()
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) > 0
