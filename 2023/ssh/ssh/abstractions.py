from abc import ABC, abstractmethod
from subprocess import Popen


class SshClient(ABC):
    @abstractmethod
    def exec(self, *args) -> Popen:
        pass

    @abstractmethod
    def forward(self, local_port: int, remote_port: int) -> Popen:
        pass
