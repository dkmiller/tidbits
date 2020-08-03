import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from typeguard import typechecked

class MultilayerPerceptron(nn.Module):
    @typechecked
    def __init__(self, dimensions: List[int], activation = F.relu):
        super(MultilayerPerceptron, self).__init__()
        self.activation = activation

        # https://discuss.pytorch.org/t/when-should-i-use-nn-modulelist-and-when-should-i-use-nn-sequential/5463
        self.linears = nn.ModuleList()

        # https://stackoverflow.com/a/21303286
        for first, second in zip(dimensions, dimensions[1:]):
            self.linears.append(nn.Linear(first, second))

    @typechecked
    def forward(self, x_in: torch.Tensor, apply_softmax: bool = False) -> torch.Tensor:
        result = x_in
        for linear in self.linears[:-1]:
            result = self.activation(linear(result))

        # No final activation function.
        result = self.linears[-1](result)

        if apply_softmax:
            result = F.softmax(result, dim=1)
        
        return result
