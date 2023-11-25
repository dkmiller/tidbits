from abc import ABC, abstractmethod


class SshClient(ABC):
    """
    TODO: find a way to replace the "gross" `Popen` abstraction with `asyncio` coroutines:
    https://superfastpython.com/asyncio-cancel-task/ .

    Sadly, `multiprocess` won't work because there's no way to cancel `Process` objects.
    """

    @abstractmethod
    async def exec(self, *args) -> str:
        pass

    @abstractmethod
    async def forward(self, local_port: int, remote_port: int) -> None:
        pass
