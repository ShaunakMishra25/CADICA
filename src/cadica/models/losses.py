"""Loss functions for imbalanced severity classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss, ``-(alpha_t) * (1 - p_t) ** gamma * log(p_t)``.

    Down-weights easy, well-classified examples so training focuses on hard
    cases. Useful when a minority class is both rare and difficult.

    Parameters
    ----------
    gamma:
        Focusing parameter. ``0`` reduces to (weighted) cross-entropy; ``2`` is
        the standard value from the original paper.
    weight:
        Optional per-class weights, applied as ``alpha_t`` for the target class.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")

        self.gamma = float(gamma)
        self.register_buffer(
            "weight", None if weight is None else weight.detach().float().clone()
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(logits, dim=1)

        log_pt = log_prob.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()

        loss = -((1.0 - pt).clamp_min(0.0) ** self.gamma) * log_pt

        if self.weight is not None:
            loss = loss * self.weight.gather(0, target)

        return loss.mean()

    def __repr__(self) -> str:
        weighted = self.weight is not None
        return f"{self.__class__.__name__}(gamma={self.gamma}, weighted={weighted})"
