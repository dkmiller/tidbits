import argparse
import horovod.torch as hvd
import torch
from azureml.core.compute import ComputeTarget
from azureml.core.run import Run
from azureml.core.workspace import Workspace
from torchvision import datasets, transforms


def main(args, run):
    if args.cuda:
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)
    kwargs = {}
    train_dataset = datasets.MNIST('data-%d' % hvd.rank(), train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                        help='use fp16 compression during allreduce')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    run = Run.get_context()
    hvd.init()
    torch.manual_seed(args.seed)

    main(args, run)
