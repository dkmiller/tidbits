from argparse import Namespace
from typing import Optional, Dict, Union, Any

try:
    from azureml.core import Run as AzureMlRun
    from azureml.core.run import _OfflineRun as AzureMlOfflineRun
except ImportError:  # pragma: no-cover
    AzureMlOfflineRun = None
    AzureMlRun = None
    _AZURE_ML_AVAILABLE = False
else:
    _AZURE_ML_AVAILABLE = True


import torch
from torch import is_tensor

from pytorch_lightning import _logger as log
from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities import rank_zero_only


class AzureMlLogger(LightningLoggerBase):
    r"""
    TODO: docs.
    """
    def __init__(self, run: Optional[AzureMlRun] = None):
        
        if not _AZURE_ML_AVAILABLE:
            raise ImportError('You want to use `azureml-defaults` logger which is not installed yet,'
                              ' install it with `pip install azureml-defaults`.')
        super().__init__()

        if run is None:
            self._experiment = AzureMlRun.get_context(allow_offline=True)
        else:
            self._experiment = run

    @property
    @rank_zero_experiment
    def experiment(self) -> AzureMlRun:
        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = self._convert_params(params)
        params = self._flatten_dict(params)
        for name, value in params.items():
            self.experiment.tag(name, value)

    @rank_zero_only
    def log_metrics(
            self,
            metrics: Dict[str, Union[torch.Tensor, float]],
            step: Optional[int] = None
    ) -> None:
        assert rank_zero_only.rank == 0, 'experiment tried to log from global_rank != 0'

        # Azure ML expects metrics to be a dictionary of detached tensors on CPU
        for key, val in metrics.items():
            if is_tensor(val):
                metrics[key] = val.cpu().detach()
            self.experiment.log(key, val)

    @rank_zero_only
    def finalize(self, status: str) -> None:
        """
        https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.run.run?view=azure-ml-py#flush-timeout-seconds-300-
        """
        self.experiment.flush()

    @property
    def save_dir(self) -> Optional[str]:
        pass

    @property
    def name(self) -> str:
        """
        https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.run.run?view=azure-ml-py#id
        """
        return self.experiment.id

    @property
    def version(self) -> str:
        r"""
        Return

        https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.run.run?view=azure-ml-py#number
        """
        if isinstance(self.experiment, AzureMlOfflineRun):
            return '0'
        else:
            return str(self.experiment.number)
