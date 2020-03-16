import pytest
import torch
from torch.autograd import Variable
import src.seed


def setup_module():
    src.seed.set_seed()


@pytest.mark.parametrize('x,f,expected', [
    ([1], lambda x: 5 * (x + 1) ** 2, [20]),
    ([1, 2], lambda x: torch.dot(torch.Tensor([3, 4]), x), [3, 4]),
    ([1, 2], lambda x: torch.dot(torch.Tensor([3, 4]), x) ** 2, [66, 88])
])
def test_gradient_is_expected(x, f, expected):
    x = Variable(torch.Tensor(x), requires_grad=True)
    y = f(x)
    y.backward()
    assert expected == x.grad.tolist()
