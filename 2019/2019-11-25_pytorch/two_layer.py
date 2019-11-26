'''
Follows: https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
'''

import torch
import torch.nn as nn

def loss_function():
    '''
    Mean-squared-error loss function.
    '''
    return nn.MSELoss(reduction='sum')

def model(input_dimension, hidden_dimension, output_dimension):
    '''
    Return a sequence of modules (layers): Linear -> ReLu -> Linear.
    '''
    return torch.nn.Sequential(
        nn.Linear(input_dimension, hidden_dimension),
        nn.ReLU(),
        nn.Linear(hidden_dimension, output_dimension)
    )

def train(model, x, y, loss_function, learning_rate=1e-4, num_iterations=500):
    for t in range(0, num_iterations):
        y_pred = model(x)
        loss = loss_function(y_pred, y)

        # TODO: replace this with a real progress function.
        if t % 100 == 0:
            print(t, loss.item())

        # TODO: why are these methods being called?
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad


if __name__ == '__main__':
    batch_size = 64
    input_dimension = 1000
    hidden_dimension = 100
    output_dimension = 10

    # Initialize objects.
    lf = loss_function()
    m = model(input_dimension, hidden_dimension, output_dimension)
    x = torch.randn(batch_size, input_dimension)
    y = torch.randn(batch_size, output_dimension)

    train(m, x, y, lf)
