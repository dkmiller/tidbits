import torch


def create_and_add_to_2d_tensor(n1: int, n2: int) -> torch.Tensor:
    '''
    1. Create a 2D tensor and then add a dimension of size 1 inserted at dimension 0.
    '''
    tensor = torch.ones(n1, n2)

    # https://mlpipes.com/adding-a-dimension-to-a-tensor-in-pytorch/
    return tensor.unsqueeze(0)


def remove_dimension_at(dim: int, tensor: torch.Tensor) -> torch.Tensor:
    '''
    2. Remove the extra dimension you just added to the previous tensor.
    '''
    shape = list(tensor.shape)
    del shape[dim]
    return tensor.view(shape)


def random_tensor(shape: list, lower: float, upper: float) -> torch.Tensor:
    '''
    3. Create a random tensor of shape 5x3 in the interval [3, 7).
    '''
    # https://pythonexamples.org/pytorch-create-tensor-with-random-values-and-specific-shape/
    assert lower < upper
    t = torch.rand(shape)
    one = torch.ones(shape)
    return (upper - lower) * t + lower * one


def normal_tensor(shape: list, μ: float, σ: float) -> torch.Tensor:
    '''
    4. Create a tensor with values from a normal distribution (mean=0, std=1).
    '''
    assert σ > 0
    dist = torch.distributions.Normal(μ, σ)
    return dist.sample(shape)


def support(t: torch.Tensor) -> torch.Tensor:
    '''
    5. Retrieve the indexes of all the nonzero elements in the tensor
    torch.Tensor([1, 1, 1, 0, 1]).
    '''
    return t.nonzero()
