from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from ssh.models import SshHost


# https://stackoverflow.com/a/44800925/
@dataclass
class SshClient(ABC):
    """
    Interface for implementation-agnostic SSH client, requiring two pieces of information:
    - identity
    - host configuration

    and exposing two key pieces of functionality:

    - execute command and return standard output
    - forward a specified remote to local port

    Sadly, `multiprocess` won't work because there's no way to cancel `Process` objects.
    """

    identity: Path
    host: SshHost

    @abstractmethod
    async def exec(self, *args) -> str:
        pass

    @abstractmethod
    async def forward(self, local_port: int, remote_port: int) -> None:
        pass
