import logging
import shlex
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from fabric import Config, Connection
from fabric.runners import Result
from paramiko.config import SSHConfig

from ssh.abstractions import SshClient
from ssh.models import SshHost
from ssh.process import spawn, spawn_and_wait

log = logging.getLogger(__name__)


@dataclass
class SshCliWrapper(SshClient):
    identity: Path
    host: SshHost

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

    async def exec(self, *args: str) -> str:
        """
        Remotely invoke a possibly long-running command and wait until it finishes, returning the
        standard output if it succeeds, throwing an exception if not.
        """
        args = (*self.prefix(), self.target(), *args)
        result = await spawn_and_wait(args)
        assert result.status == 0, result
        return result.stdout

    async def forward(self, local_port: int, remote_port: int):
        """
        Spawn a process that forwards the specified remote port to the specified local port.
        """
        log.info("Forwarding remote port %s --> local port %s", remote_port, local_port)
        args = (
            *self.prefix(),
            "-fN",
            "-L",
            # https://phoenixnap.com/kb/ssh-port-forwarding
            f"{local_port}:{self.host.host}:{remote_port}",
            self.target(),
        )

        return await spawn(args)


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

    async def exec(self, *args) -> str:
        # TODO: map(shlex.quote, args) ?
        command = " ".join(args)
        result: Result = self.connection.run(command, hide=True)
        assert result.ok, result
        return result.stdout

    async def forward(self, local_port: int, remote_port: int) -> None:
        raise NotImplementedError()
        # with self.connection.forward_local(local_port, remote_port=remote_port):
        #     import time
        #     # TODO: better logic for dropping the connection.
        #     time.sleep(4)
