import numpy as np
import pytest
import torch
import src.seed


def setup_module():
    src.seed.set_seed()


@pytest.mark.parametrize('arg', [
    [],
    [1],
    [1, -1.123123],
    [[1, 2.3], [3.2, -1.33]],
    [float('inf')],
    [float('nan')],
    [[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]
])
def test_can_create_tensor_from(arg):
    t = torch.Tensor(arg)
    assert type(t) == torch.Tensor


@pytest.mark.parametrize('arg', [
    [[1], [2, 3]],
    ['a'],
    [[1, 2.3], [None, 2.1]],
    [[[1, 2], [3, 4], [4, 5]], [6, 7]]
])
def test_cannot_create_tensor_from(arg):
    with pytest.raises(Exception):
        torch.Tensor(arg)


@pytest.mark.parametrize('arr,args', [
    ([1, 2, 3, 4], [2, 2]),
    ([1, 2, 3, 4, 5, 6], [-1, 2])
])
def test_view_works(arr, args):
    t = torch.Tensor(arr)
    t.view(*args)


@pytest.mark.parametrize('arr,args', [
    ([1, 2, 3, 4, 5], [2, 3]),
    ([1, 2, 3, 4], [-1, 3])
])
def test_view_fails(arr, args):
    with pytest.raises(Exception):
        t = torch.Tensor(arr)
        t.view(*args)
