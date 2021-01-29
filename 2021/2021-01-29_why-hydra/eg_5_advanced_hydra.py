from dataclasses import asdict, dataclass
import hydra
import logging
import pytorch_lightning as pl
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


from ml import AutoEncoder


log = logging.getLogger(__name__)


@dataclass
class DataloaderConfig:
  batch_size: int


@dataclass
class DatasetConfig:
  download: bool
  root: str


@dataclass
class ModelConfig:
    image_size: int
    color_range: int
    hyperparam_1: int
    learning_rate: float


@dataclass
class TrainerConfig:
    gpus: int
    log_every_n_steps: int
    max_epochs: int
    num_nodes: int


@dataclass
class Configuration:
  dataloader: DataloaderConfig
  dataset: DatasetConfig
  model: ModelConfig
  seed: int
  trainer: TrainerConfig


@hydra.main(config_name="config")
def main(config: Configuration):
    log.info(f"Configuration: {config}")

    pl.seed_everything(config.seed)

    model = AutoEncoder(**asdict(config.model))

    dataset = MNIST(transform=ToTensor(), **asdict(config.dataset))

    train_loader = DataLoader(dataset, **asdict(config.dataloader))

    trainer = pl.Trainer(**asdict(config.trainer))

    trainer.fit(model, train_loader)


if __name__ == "__main__":
    main()
