import argparse
import pytorch_lightning as pl
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


from ml import AutoEncoder


def main(args):
    pl.seed_everything(args.seed)

    model = AutoEncoder(
        image_size=args.image_size,
        color_range=args.color_range,
        hyperparam_1=args.hyperparam_1,
        learning_rate=args.learning_rate
    )

    dataset = MNIST(
        transform=ToTensor(),
        download=args.download,
        root=args.root
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int)

    model = parser.add_argument_group("model hyperparameters")
    model.add_argument("--image_size", default=28, type=int)
    model.add_argument("--color_range", default=64, type=int)
    model.add_argument("--hyperparam_1", default=3, type=int)
    model.add_argument("--learning_rate", default=0.001, type=float)

    dataset = parser.add_argument_group("dataset configuration")
    dataset.add_argument("--download", default=True, type=bool)
    dataset.add_argument("--root", default=".", type=str)

    args = parser.parse_args()
    main(args)
