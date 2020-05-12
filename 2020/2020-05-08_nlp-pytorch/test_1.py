import numpy as np
import pytest
from scipy.stats import kstest
import solution_1
import torch


def elements(t: torch.Tensor) -> list:
    '''
    Return a one-dimensional tensor containing all elements from the input.
    '''
    # https://stackoverflow.com/a/13792206
    n_elements = np.prod(t.shape)
    return list(t.view(n_elements))


@pytest.mark.parametrize('dim1,dim2', [
    (1, 1),
    (2, 3),
    (10, 7)
])
def test_create_and_add_to_2d_tensor(dim1: int, dim2: int) -> None:
    tensor = solution_1.create_and_add_to_2d_tensor(dim1, dim2)
    assert type(tensor) is torch.Tensor
    assert tensor.shape == torch.Size([1, dim1, dim2])


@pytest.mark.parametrize('n1,n2', [
    (1, 1),
    (2, 3),
    (10, 7)
])
def test_remove_dimension_at(n1: int, n2: int):
    tensor1 = solution_1.create_and_add_to_2d_tensor(n1, n2)
    tensor2 = solution_1.remove_dimension_at(0, tensor1)
    shape1 = list(tensor1.shape)
    # https://stackoverflow.com/a/627453
    del shape1[0]
    assert list(tensor2.shape) == shape1


@pytest.mark.parametrize('shape,lower,upper', [
    ([5, 3], 3, 7),
    ([2, 3, 4, 5], -1.2, 8.3)
])
def test_random_tensor(shape: list, lower: float, upper: float):
    t = solution_1.random_tensor(shape, lower, upper)
    assert list(t.shape) == shape
    es = elements(t)
    # https://stackoverflow.com/a/26539891
    assert all([lower <= e < upper for e in es])


@pytest.mark.parametrize('shape,μ,σ,pvalue', [
    ([5, 3], 0, 1, 0.05),
])
def test_normal_tensor(shape: list, μ: float, σ: float, pvalue: float):
    t = solution_1.normal_tensor(shape, μ, σ)
    assert list(t.shape) == shape

    es = elements(t)
    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.kstest.html
    statistic = kstest(es, 'norm', args=(μ, σ))
    assert statistic.pvalue >= pvalue


@pytest.mark.parametrize('tensor,expected', [
    ([[0, 1], [2, 0]], [[0, 1], [1, 0]])
])
def test_support(tensor, expected):
    support = solution_1.support(torch.Tensor(tensor))
    assert support.numpy().tolist() == expected
