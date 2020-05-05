import pytest
import torch
from torch import det, dot, exp, matrix_power, Tensor
from torch.autograd import Variable
import src.seed
from src.func import compose, jacobian


def setup_module():
    src.seed.set_seed()


@pytest.mark.parametrize('x,f,expected', [
    ([1], lambda x: 5 * (x + 1) ** 2, [20]),
    ([1, 2], lambda x: dot(Tensor([3, 4]), x), [3, 4]),
    ([1, 2], lambda x: dot(Tensor([3, 4]), x) ** 2, [66, 88]),
    # PyTorch has some rounding errors when computing the gradient
    # of the determinant function.
    ([[2, 3], [5, 7]], det, [[6.999999523162842, -4.999999523162842], [-3, 2]]),
    # Ditto about rounding errors.
    ([[2, 3], [5, 7]], lambda x: det(matrix_power(x, 2)), [
     [-14.000015258789062, 10.000007629394531], [5.999988555908203, -3.9999847412109375]]),
    ([[1, 2], [3, 4]], compose(exp, det), [[0.5413411855697632, - \
                                            0.4060058891773224], [-0.2706705927848816, 0.1353352963924408]])
])
def test_gradient_is_expected(x, f, expected):
    x = Variable(Tensor(x), requires_grad=True)
    y = f(x)
    y.backward()
    assert expected == x.grad.tolist()


@pytest.mark.parametrize('x,f,expected', [
    ([1], lambda x: 5 * (x + 1) ** 2, [20])
])
def test_jacobian_is_expected(x, f, expected):
    x = torch.Tensor(x)
    j = jacobian(f, x).tolist()
    assert j == expected
