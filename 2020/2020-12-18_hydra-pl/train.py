import hydra
import logging
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


from ml import AutoEncoder


@hydra.main(config_name="config")
def main(config):
    log = logging.getLogger(__name__)
    log.info(f"Configuration: {config}")

    model = AutoEncoder(**config.model)

    dataset = MNIST(transform=ToTensor(), **config.data)
    train_loader = DataLoader(dataset)

    trainer = pl.Trainer(**config.trainer)
    trainer.fit(model, train_loader)


if __name__ == "__main__":
    main()
