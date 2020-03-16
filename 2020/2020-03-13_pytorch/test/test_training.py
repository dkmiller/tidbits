import pytest
import torch
from torch.autograd import Variable
import src.seed
from src.func import compose
from src.models.linear_model import LinearModel, linear, mock_data


def setup_module():
    src.seed.set_seed()


@pytest.mark.parametrize('weights,bias,N,σ', [
    ([1, 2, 3], -5, 100, 1)
])
def test_can_create_linear_data(weights, bias, N, σ):
    (xs, ys) = mock_data(torch.Tensor(weights), bias, N, σ)
    assert N == len(xs)
    assert N == len(ys)
    assert all(type(x) == torch.Tensor for x in xs)
    assert all(type(y) == torch.Tensor for y in ys)


@pytest.mark.parametrize('weights,bias,N,σ,ε,lr,epochs', [
    ([1], 1, 1000, 0.1, 1, .01, 100),
    ([1, 2, 3], -1.5, 1000, 0, .01, .01, 10000)
])
def test_training_converges_to_linear_model(weights, bias, N, σ, ε, lr, epochs):
    weights = torch.Tensor(weights)
    (xs, ys) = mock_data(weights, bias, N, σ)

    model = LinearModel(len(weights))
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for _ in range(epochs):
        inputs = Variable(xs)
        labels = Variable(ys)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    x = torch.ones(len(weights))
    y = linear(weights, x, bias).item()
    ŷ = model(x).item()

    assert ŷ == pytest.approx(y, ε)
