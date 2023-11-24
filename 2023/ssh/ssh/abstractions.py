from abc import ABC, abstractmethod
from subprocess import Popen


class SshClient(ABC):
    """
    TODO: find a way to replace the "gross" `Popen` abstraction with `asyncio` coroutines:
    https://superfastpython.com/asyncio-cancel-task/ .

    Sadly, `multiprocess` won't work because there's no way to cancel `Process` objects.
    """

    @abstractmethod
    def exec(self, *args) -> Popen:
        pass

    @abstractmethod
    def forward(self, local_port: int, remote_port: int) -> Popen:
        pass
