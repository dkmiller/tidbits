'''
Functional programming 
'''

import torch


def identity(x):
    '''
    Identity function, defined here because the Python team has explicitly
    rejected the proposal to include one in the standard library:
    https://bugs.python.org/issue1673203 .
    '''
    return x


def compose(*args):
    '''
    Compose functions f1, f2, f3, ... as x -> ...f3(f2(f1(x))). Defined here
    because the Python team has explicitly rejected the proposal to include one
    in the standard library: https://bugs.python.org/issue1506122 .
    '''

    def result(x):
        for f in reversed(args):
            x = f(x)
        return x
    return result


def jacobian(f, x):
    '''
    Return the Jacobian of `f`, evaluated at `x`. Included here because
    PyTorch doesn't include this as a native operation:
    https://github.com/pytorch/pytorch/issues/8304 .
    '''

    # Follows https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa .
    x = x.squeeze()
    n = x.size()[0]
    x = x.repeat(n, 1)
    x.requires_grad_(True)
    f(x).backward(torch.eye(n))
    return x.grad.data
