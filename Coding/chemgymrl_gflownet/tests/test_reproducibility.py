import random

import numpy as np
import torch

from chemgymrl_gflownet import seed_process


def test_seed_process_repeats_python_numpy_and_torch_draws():
    seed_process(17)
    first = (random.random(), np.random.random(), torch.rand(3))

    seed_process(17)
    second = (random.random(), np.random.random(), torch.rand(3))

    assert first[0] == second[0]
    assert first[1] == second[1]
    torch.testing.assert_close(first[2], second[2])
