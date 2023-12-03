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
from ssh.models import SshHost
from ssh.process import kill, wait

log = logging.getLogger(__name__)


@dataclass
class SshCliWrapper(SshClient):
    identity: Path
    host: SshHost

    @classmethod
    def construct(cls, identity: Path, host: SshHost) -> Self:
        return cls(identity, host)

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
            # "-fN",
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


@dataclass
class FabricClient(SshClient):
    """
    https://docs.fabfile.org/en/latest/api/connection.html
    TODO: handle

    ```
    tests/test_client_server_functionality.py .Warning: Permanently added '[localhost]:2222' (ED25519) to the list of known hosts.
    ```

    possibly via https://github.com/fabric/fabric/issues/2071
    """

    identity: Path
    host: SshHost

    @classmethod
    def construct(cls, identity: Path, host: SshHost) -> Self:
        return cls(identity, host)

    @cached_property
    def connection(self):
        # https://phoenixnap.com/kb/ssh-config
        ssh_conf = SSHConfig.from_text(
            f"""
        Host localhost
            HostName localhost
            User {self.host.user}
            IdentityFile {str(self.identity)}
            StrictHostKeyChecking accept-new
        """
        )
        config = Config(ssh_config=ssh_conf)
        # https://github.com/fabric/fabric/issues/2071
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

    # - https://adamj.eu/tech/2021/07/04/python-type-hints-how-to-type-a-context-manager/
    # - https://stackoverflow.com/q/49733699/
    @contextmanager
    def forward(self, local_port: int, remote_port: int):
        with self.connection.forward_local(local_port, remote_port=remote_port):
            yield
