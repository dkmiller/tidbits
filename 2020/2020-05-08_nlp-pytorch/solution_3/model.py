import common
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose


class ReviewClassifier(nn.Module):
    '''
    Simple perceptron.
    '''

    def __init__(self, num_features: int, activation=F.sigmoid):
        super(ReviewClassifier, self).__init__()

        self.forward = Compose([
            nn.Linear(in_features=num_features, out_features=1),
            lambda x: x.squeeze(),
            activation
        ])
