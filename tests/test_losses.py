"""Tests for the focal loss."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch.nn.functional as F  # noqa: E402

from cadica.models import FocalLoss  # noqa: E402


def test_gamma_zero_matches_cross_entropy() -> None:
    logits = torch.randn(8, 3)
    target = torch.tensor([0, 1, 2, 1, 2, 0, 1, 2])

    focal = FocalLoss(gamma=0.0)(logits, target)
    cross_entropy = F.cross_entropy(logits, target)

    assert float(focal) == pytest.approx(float(cross_entropy), rel=1e-5)


def test_focal_downweights_easy_examples() -> None:
    # One confident correct prediction plus one hard one.
    logits = torch.tensor([[5.0, -5.0, -5.0], [0.1, 0.0, -0.1]])
    target = torch.tensor([0, 1])

    plain = FocalLoss(gamma=0.0)(logits, target)
    focused = FocalLoss(gamma=2.0)(logits, target)

    assert float(focused) < float(plain)


def test_class_weights_are_applied() -> None:
    logits = torch.zeros(2, 3)
    target = torch.tensor([0, 2])
    weight = torch.tensor([2.0, 1.0, 0.5])

    weighted = FocalLoss(gamma=0.0, weight=weight)(logits, target)
    unweighted = FocalLoss(gamma=0.0)(logits, target)

    # The rare class (index 0) has the largest weight, so the mean rises.
    assert float(weighted) != pytest.approx(float(unweighted))


def test_rejects_negative_gamma() -> None:
    with pytest.raises(ValueError):
        FocalLoss(gamma=-1.0)
