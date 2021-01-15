from typing import Union
from azureml.core import Run
from azureml.core.run import _SubmittedRun
import hydra
import logging
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger, LightningLoggerBase
import sys
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


from ml import AutoEncoder


log = logging.getLogger(__name__)


def create_logger() -> Union[bool, LightningLoggerBase]:
    """
    Loosely imitate:
    https://github.com/Azure/azureml-examples/blob/main/tutorials/using-pytorch-lightning/3.log-with-mlflow.ipynb
    """
    run = Run.get_context()
    if isinstance(run, _SubmittedRun):
        experiment = run.experiment
        tracking_uri = experiment.workspace.get_mlflow_tracking_uri()
        exp_name = run.experiment.name
        log.info(
            f"Using MLFlow logger with tracking URI {tracking_uri} and experiment name {exp_name}"
        )
        rv = MLFlowLogger(exp_name, tracking_uri)
        rv._run_id = run.id
    else:
        log.warning("Unable to get AML run context! Logging locally.")
        rv = True

    return rv


def create_trainer(config) -> pl.Trainer:
    logger = create_logger()
    trainer = pl.Trainer(logger, **config.trainer)
    return trainer


def set_environment_variables_for_nccl_backend(config) -> None:
    """
    Follow:
    https://github.com/Azure/azureml-examples/blob/main/tutorials/using-pytorch-lightning/src/azureml_env_adapter.py
    """

    os.environ["MASTER_ADDR"] = os.environ["AZ_BATCHAI_MPI_MASTER_NODE"]
    os.environ["MASTER_PORT"] = config.master_port

    # Node rank is the word rank from MPI run.
    os.environ["NODE_RANK"] = os.environ["AZ_BATCHAI_MPI_MASTER_NODE"]


@hydra.main(config_name="config")
def main(config):
    log.info(f"Arguments: {sys.argv}")
    log.info(f"Configuration: {config}")

    set_environment_variables_for_nccl_backend(config)

    pl.seed_everything(config.seed)

    model = AutoEncoder(**config.model)

    dataset = MNIST(transform=ToTensor(), **config.data)
    train_loader = DataLoader(dataset, **config.loader)

    trainer = create_trainer(config)
    trainer.fit(model, train_loader)


if __name__ == "__main__":
    main()
