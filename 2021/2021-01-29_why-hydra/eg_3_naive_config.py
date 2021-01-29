import argparse
import json
import logging
import pytorch_lightning as pl
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


from ml import AutoEncoder


def main(config):
    logging.basicConfig(level=args.log_level)
    logging.info(f"Configuration: {config}")

    pl.seed_everything(config["seed"])

    model = AutoEncoder(**config["model"])

    dataset = MNIST(
        transform=ToTensor(),
        **config["dataset"]
    )

    train_loader = DataLoader(dataset, **config["dataloader"])

    trainer = pl.Trainer(**config["trainer"])

    trainer.fit(model, train_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="eg_3_config.json", type=str)
    parser.add_argument("--log_level", default="INFO", type=str)
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    main(config)
