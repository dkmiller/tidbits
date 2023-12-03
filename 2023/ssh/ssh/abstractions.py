"""
Implementation-agnostic interfaces.
"""
from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from typing_extensions import Self

from ssh.models import SshHost


@dataclass(frozen=True)
class Result:
    """
    Result of running a process, including standard output, error, and exit code.
    """

    stderr: str
    stdout: str
    status: int


# https://stackoverflow.com/a/44800925/
class SshClient(ABC):
    """
    Interface for implementation-agnostic SSH client, requiring two pieces of information:
    - identity
    - host configuration

    and exposing key implementation-agnostic functionality.

    The interface is synchronous, because the only case truly requiring parallelism (port forward
    an HTTP server running remotely) is better implemented using a remote `screen` session. In
    addition, native Python implementations can't use `multiprocess` because there is no way to
    cancel `Process` objects.
    """

    @classmethod
    @abstractmethod
    def construct(cls, identity: Path, host: SshHost) -> Self:
        """
        Construct an instance of the implementation from the path to a private key file along with
        host configuration.
        """

    @abstractmethod
    def exec(self, *args: str) -> Result:
        """
        Execute the provided command line arguments remotely, block while it runs, and return the
        resulting standard output, error, and code.
        """

    # Type hints and context managers:
    # - https://adamj.eu/tech/2021/07/04/python-type-hints-how-to-type-a-context-manager/
    # - https://stackoverflow.com/q/49733699/
    @abstractmethod
    @contextmanager
    def forward(self, local_port: int, remote_port: int) -> Generator[None, None, None]:
        """
        Context manager for an SSH tunnel connecting the specified remote port to the local port.
        The tunnel will be destroyed on exiting the context manager.
        """
