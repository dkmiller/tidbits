import argparse
import json
import pytorch_lightning as pl
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


from ml import AutoEncoder


def main(config):
    pl.seed_everything(config["seed"])

    model = AutoEncoder(**config["model"])

    dataset = MNIST(
        transform=ToTensor(),
        **config["dataset"]
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="config.json", type=str)
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    main(config)
