from typing import Union
from azureml.core import Run
from azureml.core.run import _SubmittedRun
import hydra
import logging
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger, LightningLoggerBase
import sys
import torch
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


def set_environment_variable(variable: str, new_value: str) -> None:
    old_value = os.environ.get(variable)
    log.info(f"Setting {variable}: {old_value} -> {new_value}")
    os.environ[variable] = new_value


def copy_environment_variable(target: str, source: str) -> None:
    set_environment_variable(target, os.environ[source])


def set_environment_variables_for_nccl_backend(config) -> None:
    """
    Follow:
    ~~https://azure.github.io/azureml-web/docs/cheatsheet/distributed-training#pytorch-lightning-ddp-accelerator-per-node-launch
    https://github.com/Azure/azureml-examples/blob/main/tutorials/using-pytorch-lightning/src/azureml_env_adapter.py
    """

    copy_environment_variable("MASTER_ADDR", "AZ_BATCHAI_MPI_MASTER_NODE")
    set_environment_variable("MASTER_PORT", "6105")

    # Node rank is the word rank from MPI run.
    copy_environment_variable("NODE_RANK", "OMPI_COMM_WORLD_RANK")


    # single_node = config.trainer.num_nodes == 1

    # if not single_node:
    #     log.info(f"Running in multiple nodes")
    #     set_environment_variable(
    #         "MASTER_ADDR", os.environ["AZ_BATCH_MASTER_NODE"].split(":")[0]
    #     )

    #     copy_environment_variable("NODE_RANK", "OMPI_COMM_WORLD_RANK")
    #     copy_environment_variable("LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK")
    #     copy_environment_variable("WORLD_SIZE", "OMPI_COMM_WORLD_SIZE")

    # set_environment_variable("NCCL_SOCKET_IFNAME", "^docker0,lo")


@hydra.main(config_name="config")
def main(config):
    log.info(f"Arguments: {sys.argv}")
    log.info(f"Configuration: {config}")
    log.info(f"GPUs: {torch.cuda.device_count()}")
    log.info(f"Environment: {os.environ}")

    set_environment_variables_for_nccl_backend(config)

    pl.seed_everything(config.seed)

    model = AutoEncoder(**config.model)

    dataset = MNIST(transform=ToTensor(), **config.data)
    train_loader = DataLoader(dataset)

    trainer = create_trainer(config)
    trainer.fit(model, train_loader)


if __name__ == "__main__":
    main()
