from abc import ABC, abstractmethod


class SshClient(ABC):
    @abstractmethod
    def exec(self, args: list[str]):
        pass

    def tunnel(self, local_port: int, remote_port: int):
        pass


class SshServer(ABC):
    pass
