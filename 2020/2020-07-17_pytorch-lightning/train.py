from azureml.core.run import Run
import model
from pyconfigurableml.entry import run
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger


# TODO: long-term, get this from pytorch-lightning.
from _azureml import AzureMlLogger


def main(config, log):
    loggers = [
        AzureMlLogger(),
        TensorBoardLogger('lightning_logs')
    ]

    trainer = pl.Trainer(logger=loggers, **config.trainer)
    net = model.Mnist(**config.model)
    trainer.fit(net)


run(main, __file__, __name__)
