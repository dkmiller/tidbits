import torch
import src.seed


def setup_module():
    src.seed.set_manual_seed()


def test_foo():
    pass
