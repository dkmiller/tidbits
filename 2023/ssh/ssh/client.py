import logging
import shlex
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from subprocess import PIPE, Popen

from fabric import Config, Connection
from fabric.runners import Result as FabricResult
from paramiko.config import SSHConfig
from typing_extensions import Self

from ssh.abstractions import Result, SshClient
from ssh.config import host_config
from ssh.models import SshHost
from ssh.process import kill, wait

log = logging.getLogger(__name__)


@dataclass
class ClientBase:
    identity: Path
    host: SshHost

    @classmethod
    def construct(cls, identity: Path, host: SshHost) -> Self:
        return cls(identity, host)


class SshCliWrapper(ClientBase, SshClient):
    def prefix(self):
        """
        Shared prefix of SSH command: `ssh -i * -p * -o * ...`.
        """
        return [
            "ssh",
            "-i",
            str(self.identity.absolute()),
            "-p",
            str(self.host.port),
            "-o",
            # https://stackoverflow.com/a/61946687/
            "StrictHostKeyChecking=accept-new",
        ]

    def target(self):
        """
        Target of SSH commands and port forwarding, e.g. `dan@localhost`.
        """
        return f"{self.host.user}@{self.host.host}"

    def popen(self, args: tuple[str, ...]) -> Popen:
        # https://stackoverflow.com/a/31867499/
        rv = Popen(args, stderr=PIPE, stdout=PIPE)
        # Make logged commands copy/pasteable.
        copyable_args = " ".join(map(shlex.quote, args))
        log.info("Spawned process %s: %s", rv.pid, copyable_args)
        return rv

    def exec(self, *args: str) -> Result:
        args = (*self.prefix(), self.target(), *args)
        process = self.popen(args)
        return wait(process)

    @contextmanager
    def forward(self, local_port: int, remote_port: int):
        log.info("Forwarding remote port %s --> local port %s", remote_port, local_port)
        args = (
            *self.prefix(),
            # Why not -f? That runs the port forwarding in a process we can't control from
            # Python, which makes it difficult to cancel.
            "-N",
            "-L",
            # https://phoenixnap.com/kb/ssh-port-forwarding
            f"{local_port}:{self.host.host}:{remote_port}",
            self.target(),
        )
        process = self.popen(args)
        # TODO: this should not be necessary.
        time.sleep(0.2)

        try:
            yield
        finally:
            kill(process)


class FabricClient(ClientBase, SshClient):
    """
    SSH client implementation using the
    [Paramiko Fabric](https://docs.fabfile.org/en/latest/api/connection.html) library.
    """

    @cached_property
    def connection(self):
        text = host_config(self.host, self.identity)
        ssh_config = SSHConfig.from_text(text)
        config = Config(ssh_config=ssh_config)
        # This implicitly accepts new SSH hosts: https://github.com/fabric/fabric/issues/2071
        return Connection(
            self.host.host,
            user=self.host.user,
            port=self.host.port,
            config=config,
            forward_agent=True,
        )

    def exec(self, *args):
        """
        Naively convert the provided args into a shell command, invoke it remotely, and convert
        the resulting Fabric `Result` into the standard dataclass.

        Warning: this does not handle command escaping.
        """
        command = " ".join(args)
        res: FabricResult = self.connection.run(command, hide=True)
        return Result(res.stderr, res.stdout, res.return_code)

    @contextmanager
    def forward(self, local_port: int, remote_port: int):
        with self.connection.forward_local(local_port, remote_port=remote_port):
            yield
