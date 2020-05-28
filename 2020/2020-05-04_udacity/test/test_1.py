import pytest
from src import solutions_1
import torch
from typeguard import typechecked


@pytest.mark.parametrize('feature_shape,bias_shape', [
    ((1, 5), (1, 1))
])
@typechecked
def test_calculate_output(feature_shape: (int, int), bias_shape: (int, int)) -> None:
    features = torch.randn(feature_shape)
    weights = torch.randn_like(features)
    bias = torch.randn(bias_shape)

    result = solutions_1.calculate_output(features, weights, bias)

    assert result.shape == (1, 1)


@pytest.mark.parametrize('feature_shape,bias_shape', [
    ((1, 5), (1, 1))
])
def test_calculate_output_using_matrix_multiplication(feature_shape: (int, int), bias_shape: (int, int)) -> None:
    features = torch.randn(feature_shape)
    weights = torch.randn_like(features)
    bias = torch.randn(bias_shape)

    result = solutions_1.calculate_output_using_matrix_multiplication(features, weights, bias)

    assert result.shape == (1, 1)


@pytest.mark.parametrize('feature_shape,n_hidden,n_output', [
    ((1, 3), 2, 1)
])
def test_calculate_multilayer_output(feature_shape, n_hidden, n_output):
    features = torch.randn(feature_shape)
    n_input = features.shape[1]
    w1 = torch.randn(n_input, n_hidden)
    w2 = torch.randn(n_hidden, n_output)
    b1 = torch.randn((1, n_hidden))
    b2 = torch.randn((1, n_output))

    result = solutions_1.calculate_multilayer_output(features, w1, w2, b1, b2)

    assert result.shape == (1, 1)
