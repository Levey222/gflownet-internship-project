"""Process-level random seeding for project experiments."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_process(seed: int) -> None:
    """Seed model initialisation and process-level random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
