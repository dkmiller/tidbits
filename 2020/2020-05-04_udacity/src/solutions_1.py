import torch
from typeguard import typechecked

@typechecked
def activation(x: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-x))


@typechecked
def calculate_output(features: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    '''
    Calculate the output of the network with input features features, weights
    weights, and bias bias. Similar to Numpy, PyTorch has a torch.sum()
    function, as well as a .sum() method on tensors, for taking sums. Use the
    function activation defined above as the activation function.
    '''
    return activation((features * weights).sum() + bias)


@typechecked
def calculate_output_using_matrix_multiplication(features: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    '''
    Calculate the output of our little network using matrix multiplication.
    '''
    # Reverse a tuple.
    # https://stackoverflow.com/a/10201993
    return activation(torch.mm(features, weights.view(features.shape[::-1])) + bias)


@typechecked
def calculate_multilayer_output(features, w1, w2, b1, b2) -> torch.Tensor:
    '''
    Calculate the output for this multi-layer network using the weights W1 & W2,
    and the biases, B1 & B2.
    '''
    h1 = activation(torch.mm(features, w1) + b1)
    return activation(torch.mm(h1, w2) + b2)
