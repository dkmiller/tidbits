import torch
from torch.distributions.normal import Normal


def linear(weights, x, bias):
    '''
    Compute `weights • x + bias`.
    '''
    return torch.dot(weights, x) + bias


def mock_data(weights, bias, N, σ) -> (torch.Tensor, torch.Tensor):
    '''
    Return mock linear data (N samples) of the form
    `weights • [0, 1]^dim + bias + Normal(0, σ)`.
    '''
    size = len(weights)
    noise = Normal(torch.zeros(1), torch.Tensor([float(σ)]))
    xs = []
    ys = []
    for _ in range(N):
        x = torch.rand(size)
        ε = noise.sample()
        y = linear(weights, x, bias) + ε
        xs.append(x)
        ys.append(y)
    xs = torch.stack(xs)
    ys = torch.cat(ys).view(-1, 1)
    return (xs, ys)


class LinearModel(torch.nn.Module):
    '''
    Neural network transforming a specified number of inputs into a single
    number via a linear transformation.
    '''

    def __init__(self, input_size):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(input_size, 1, True)

    def forward(self, x):
        return self.linear(x)
