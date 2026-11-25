"""Model definitions."""

from cadica.models.cnn import CADICACNN
from cadica.models.losses import FocalLoss

__all__ = ["CADICACNN", "FocalLoss"]
