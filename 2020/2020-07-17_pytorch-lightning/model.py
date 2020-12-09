import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torchvision import datasets, transforms
from torchvision.transforms import Compose
import pytorch_lightning as pl


class Mnist(pl.LightningModule):
    def __init__(self,
            batch_size: int,
            kernel_size: int,
            lr: float,
            momentum: float,
            weight_decay: float):
        '''
        TODO: pass in these Hyperparameters.
        '''
        super(Mnist, self).__init__()

        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.conv1 = nn.Conv2d(1, 10, kernel_size)
        self.conv2 = nn.Conv2d(10, 20, kernel_size)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        return Compose([
            self.conv1,
            lambda t: F.max_pool2d(t, 2),
            F.relu,
            self.conv2,
            self.conv2_drop,
            lambda t: F.max_pool2d(t, 2),
            lambda t: t.view(-1, 320),
            self.fc1,
            F.relu,
            lambda t: F.dropout(t, training=self.training),
            self.fc2,
            lambda t: F.log_softmax(t, dim=1)
        ])(x)

    def prepare_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train = datasets.MNIST('data', download=True, train=True, transform=transform)
        self.mnist_train = train

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.mnist_train,
            batch_size=self.batch_size
        )

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.parameters(),
            self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )

    def cross_entropy_loss(self, ŷ, y):
        return F.nll_loss(ŷ, y)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        ŷ = self.forward(x)

        loss = self.cross_entropy_loss(ŷ, y)
        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}
