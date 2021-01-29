import hydra
import logging
import pytorch_lightning as pl
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


from ml import AutoEncoder


@hydra.main(config_name="eg_4_config")
def main(config):
    logging.info(f"Configuration: {config}")

    pl.seed_everything(config.seed)

    model = AutoEncoder(**config.model)

    dataset = MNIST(transform=ToTensor(), **config.dataset)

    train_loader = DataLoader(dataset, **config.dataloader)

    trainer = pl.Trainer(**config.trainer)

    trainer.fit(model, train_loader)


if __name__ == "__main__":
    main()
