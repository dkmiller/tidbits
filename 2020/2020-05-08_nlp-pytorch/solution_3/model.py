import torch
import torch.nn as nn
from torchvision.transforms import Compose
from typeguard import typechecked


class ReviewClassifier(nn.Module):
    '''
    Simple perceptron.
    '''

    @typechecked
    def __init__(self, num_features: int, activation=torch.sigmoid):
        super(ReviewClassifier, self).__init__()

        self.activation = activation        
        # This magically allows optimizers to work:
        # https://discuss.pytorch.org/t/error-optimizer-got-an-empty-parameter-list/1501
        self.linear = nn.Linear(in_features=num_features, out_features=1)

    def forward(self, x_in, apply_activation: bool=False):
        ŷ = self.linear(x_in).squeeze()

        # print(ŷ)

        if apply_activation:
            ŷ = self.activation(ŷ)

        # print(ŷ)

        return ŷ
