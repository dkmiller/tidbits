import logging
from dataclasses import dataclass
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
    identity: Path
    host: SshHost

    async def exec(self, *args) -> str:
        # https://phoenixnap.com/kb/ssh-config
        ssh_conf = SSHConfig.from_text(
            f"""
        Host localhost
            HostName localhost
            User {self.host.user}
            IdentityFile {str(self.identity)}
        """
        )
        config = Config(ssh_config=ssh_conf)
        import shlex

        # https://github.com/fabric/fabric/issues/2071
        command = " ".join(map(shlex.quote, args))
        conn = Connection(
            self.host.host, user=self.host.user, port=self.host.port, config=config
        )
        result: Result = conn.run(command, hide=True)
        assert result.ok, result
        return result.stdout

    async def forward(self, local_port: int, remote_port: int) -> None:
        raise NotImplementedError()
