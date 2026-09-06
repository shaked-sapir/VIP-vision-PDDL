"""Seed every RNG the symbolic ROSAME arms draw from."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: Optional[int]) -> None:
    """Seed Python's, NumPy's and torch's generators; ``None`` leaves them untouched."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
