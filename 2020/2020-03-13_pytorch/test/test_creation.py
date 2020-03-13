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
    [float('nan')]
])
def test_can_create_tensor_from(arg):
    t = torch.Tensor(arg)
    assert type(t) == torch.Tensor


@pytest.mark.parametrize('arg', [
    [[1], [2, 3]],
    ['a'],
    [[1, 2.3], [None, 2.1]]
])
def test_cannot_create_tensor_from(arg):
    with pytest.raises(Exception):
        torch.Tensor(arg)
