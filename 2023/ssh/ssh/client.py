import logging
import shlex
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from subprocess import PIPE, Popen

from ssh.models import SshHost
from ssh.process import kill

log = logging.getLogger(__name__)


@dataclass
class SshCliWrapper:
    identity: Path
    host: SshHost
    processes: list[Popen] = field(default_factory=list)

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

    def exec(self, *command):
        """
        Remotely invoke a possibly long-running command and return a reference to the process.
        """
        args = [*self.prefix(), self.target(), *command]
        return self._popen(args)

    def _popen(self, args: list[str]) -> Popen:
        # https://stackoverflow.com/a/31867499/
        rv = Popen(args, stderr=PIPE, stdout=PIPE)
        self.processes.append(rv)
        # Make logged commands copy/pasteable.
        copyable_args = " ".join(map(shlex.quote, args))
        log.info("Spawned process %s: %s", rv.pid, copyable_args)
        return rv

    def forward(self, local_port: int, remote_port: int) -> Popen:
        """
        Spawn a process that forwards the specified remote port to the specified local port.
        """
        log.info("Forwarding remote port %s --> local port %s", remote_port, local_port)
        args = [
            *self.prefix(),
            "-fN",
            "-L",
            # https://phoenixnap.com/kb/ssh-port-forwarding
            f"{local_port}:{self.host.host}:{remote_port}",
            self.target(),
        ]
        return self._popen(args)


@contextmanager
def ssh_cli_wrapper(identity: Path, host: SshHost):
    ssh_cli = SshCliWrapper(identity, host)
    try:
        yield ssh_cli
    finally:
        for process in ssh_cli.processes:
            kill(process)
