from typing import Union
from azureml.core import Run
from azureml.core.run import _SubmittedRun
import hydra
import logging
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger, LightningLoggerBase
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


from ml import AutoEncoder


def mlflow_logger() -> Union[bool, LightningLoggerBase]:
    """
    Loosely imitate:
    https://github.com/Azure/azureml-examples/blob/main/tutorials/using-pytorch-lightning/3.log-with-mlflow.ipynb
    """
    run = Run.get_context()
    if isinstance(run, _SubmittedRun):
        tracking_uri = run.experiment.workspace.get_mlflow_tracking_uri()
        exp_name = run.id
        rv = MLFlowLogger(exp_name, tracking_uri)
    else:
        rv = True

    return rv


@hydra.main(config_name="config")
def main(config):
    log = logging.getLogger(__name__)
    log.info(f"Configuration: {config}")

    pl.seed_everything(config.seed)

    model = AutoEncoder(**config.model)

    dataset = MNIST(transform=ToTensor(), **config.data)
    train_loader = DataLoader(dataset)

    logger = mlflow_logger()

    trainer = pl.Trainer(logger, **config.trainer)
    trainer.fit(model, train_loader)


if __name__ == "__main__":
    main()
