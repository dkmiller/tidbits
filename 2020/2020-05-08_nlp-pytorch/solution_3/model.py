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
        
        # This magically allows optimizers to work:
        # https://discuss.pytorch.org/t/error-optimizer-got-an-empty-parameter-list/1501
        self.linear = nn.Linear(in_features=num_features, out_features=1)

        self.forward = Compose([
            self.linear,
            lambda x: x.squeeze(),
            activation
        ])
