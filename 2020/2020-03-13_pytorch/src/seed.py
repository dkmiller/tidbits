'''
Global logic for ensuring consistency across randomized numbers.
'''

import numpy as np
import random
import torch


_GLOBAL_SEED = 0


def set_seed():
    '''
    Use this instead of framework-specific "set seed" methods.
    This enforces the use of a single, global, seed.
    '''
    _set_seed(_GLOBAL_SEED)


def _set_seed(seed):
    _set_numpy_seed(seed)
    _set_pytorch_seed(seed)
    _set_random_seed(seed)


def _set_numpy_seed(seed):
    np.random.seed(seed)


def _set_pytorch_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _set_random_seed(seed):
    random.seed(seed)
