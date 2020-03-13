import torch


def set_manual_seed(seed=0):
    '''
    Use this instead of framework-specific "set seed" methods.
    This enforces the use of a single, global, seed.
    '''
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
