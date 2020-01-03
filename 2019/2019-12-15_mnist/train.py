'''
Train a model, producing an .ONNX file with the inference logic.
'''

import argparse
import logging
import os
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split
from torchvision import transforms

class Mnist(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        path = os.path.join(root_dir, csv_file)
        self.df = pd.read_csv(path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample = {
            'image': torch.Tensor(row.drop('class')),
            'class': torch.Tensor(row['class'])
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout2d(p=0.25)
        self.dropout2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(in_features=9216, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        '''
        Follow https://stackoverflow.com/a/10366417 to be as functional as
        possible.
        '''
        # Sadly, reduce doesn't accept named arguments.
        return reduce(lambda t, f: f(t),
            [
                self.conv1,
                F.relu, 
                self.conv2,
                lambda t: F.max_pool2d(t, 2),
                self.dropout1,
                lambda t: torch.flatten(t, 1),
                self.fc1,
                F.relu,
                self.dropout2,
                self.fc2,
                lambda t: F.log_softmax(t, dim=1)
            ],
            x)

def load(root_dir, csv_file):
    '''
    Follow https://stackoverflow.com/a/51768651 to create train / test data.
    '''
    return Mnist(csv_file, root_dir)
    # path = os.path.join(root_dir, csv_file)
    # df = pd.read_csv(path)
    # # https://stackoverflow.com/a/51859020
    # features = torch.Tensor(df.drop('class', axis=1).values)
    # labels = torch.Tensor((df[['class']].values))

    # print('baz!')
    # td = TensorDataset(features, labels)
    # print('baz2')
    # result = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.1307, ), (0.3081, ))]).__call__(df.drop('class', axis=1).values)
    # print('baz3')
    # return result

def train(model, optimizer, epoch, interval, loader):
    model.train()
    for batch, (data, target) in enumerate(loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch % interval == 0:
            print(f'{epoch} [{batch * len(data)}/{len(loader.dataset)}]\tloss: {loss.item()}')

def main(args):
    logging.basicConfig(level=logging.DEBUG)
    logging.info('boo!')
    logging.info(args)
    torch.manual_seed(args.seed)
    model = Net()
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # mnist = load(args.directory, args.csv)
    # print(mnist)
    test_loader = load(args.directory, args.csv)

    logging.info('pre-epoch')
    for epoch in range(1, args.epochs + 1):
        logging.info(f'train {epoch}')
        train(model, optimizer, epoch, args.log_interval, test_loader)
        scheduler.step()


if __name__ == '__main__':
    # https://docs.python.org/3/library/argparse.html
    parser = argparse.ArgumentParser(
        description='Train and serialize an image-classification model using PyTorch.')

    parser.add_argument('--csv', default='mnist.csv', help='File to load.')
    parser.add_argument('--directory', dest='directory', default=os.getcwd(), help='Default directory.')
    parser.add_argument('--epochs', type=int, default=14, help='Number of epochs.')
    parser.add_argument('--gamma', type=float, default=0.7, help='Learning step rate.')
    parser.add_argument('--log-interval', type=int, default=10, help='How many batches to log.')
    parser.add_argument('--lr', type=float, default=1, help='Learning rate.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')

    args = parser.parse_args()

    main(args)
