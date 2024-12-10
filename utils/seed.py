import random

import numpy as np
import torch


def seed_everything(seed: int = 42):
    """Seed every random number generator to ensure reproducability.

    Args:
        seed (int, optional): Seed number to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
