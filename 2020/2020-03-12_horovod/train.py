import argparse
import horovod.torch as hvd
import model
import torch
import torch.nn.functional as F
import torch.optim as optim
from azureml.core.compute import ComputeTarget
from azureml.core.run import Run
from azureml.core.workspace import Workspace
from torchvision import datasets, transforms


def main(args, run):
    if args.cuda:
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)
    kwargs = {}

    ts = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        'data-%d' % hvd.rank(), train=True, download=True, transform=ts)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)

    test_dataset = datasets.MNIST(
        'data-%d' % hvd.rank(), train=False, transform=ts)
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.test_batch_size, sampler=test_sampler, **kwargs)

    m = model.Net()

    if args.cuda:
        m.cuda()

    hvd.broadcast_parameters(m.state_dict(), root_rank=0)
    optimizer = optim.SGD(m.parameters(), lr=args.lr * hvd.size(), momentum=args.momentum)
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
    optimizer = hvd.DistributedOptimizer(optimizer,
                                     named_parameters=m.named_parameters(),
                                     compression=compression)

    for epoch in range(1, args.epochs + 1):
        train(epoch, m, train_sampler, train_loader, optimizer)
        test(m, test_loader, args, test_sampler)


def test(model, test_loader, args, test_sampler):
    model.eval()
    test_loss = 0.
    test_accuracy = 0.
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        test_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum()

    test_loss /= len(test_sampler)
    test_accuracy /= len(test_sampler)

    test_loss = metric_average(test_loss, 'avg_loss')
    test_accuracy = metric_average(test_accuracy, 'avg_accuracy')

    if hvd.rank() == 0:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
            test_loss, 100. * test_accuracy))


def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


def train(epoch, model, train_sampler, train_loader, optimizer):
    model.train()
    train_sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_sampler),
                100. * batch_idx / len(train_loader), loss.item()))
            run.log('loss', loss.item())



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
