"""
NeRF MLP Package

A PyTorch implementation of Neural Radiance Fields (NeRF) using MLPs.
"""

from .model import NeRFMLP
from .renderer import NeRFRenderer
from .data import NeRFDataset, auto_tune_batch_size

__version__ = "1.0.0"
__all__ = ["NeRFMLP", "NeRFRenderer", "NeRFDataset", "auto_tune_batch_size"] 