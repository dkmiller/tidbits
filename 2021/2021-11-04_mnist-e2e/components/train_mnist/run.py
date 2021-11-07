# python .\run.py --train-image C:\tmp\train-image\ --train-label C:\tmp\train-label\ --test-image C:\tmp\test-image\ --test-label C:\tmp\test-label\

from argparse_dataclass import dataclass
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger

from data import Mnist
from ml import MnistEncoder


@dataclass
class Args:
    train_image: str
    train_label: str
    test_image: str
    test_label: str


def main(args: Args):
    train_image = Path(args.train_image)
    train_label = Path(args.train_label)
    test_image = Path(args.test_image)
    test_label = Path(args.test_label)
    data = Mnist(32, 0.9, train_image, train_label, test_image, test_label)
    model = MnistEncoder(28, 64, 3)

    # https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.loggers.mlflow.html
    mlflow_logger = MLFlowLogger()
    trainer = pl.Trainer(max_epochs=1, logger=mlflow_logger)

    trainer.fit(model, train_dataloader=data)

    # TODO: validation against test data?


if __name__ == "__main__":
    args = Args.parse_args()
    main(args)
