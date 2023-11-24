import logging
from dataclasses import dataclass
from pathlib import Path
from subprocess import PIPE, Popen

from ssh.models import SshHost

log = logging.getLogger(__name__)


@dataclass
class SshCliWrapper:
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

    def exec(self, *command):
        """
        Remotely invoke a possibly long-running command and return a reference to the process.
        """
        args = [*self.prefix(), self.target(), *command]
        return self._popen(args)

    def _popen(self, args: list[str]) -> Popen:
        log.info("Popen %s", args)
        # https://stackoverflow.com/a/31867499/
        return Popen(args, stderr=PIPE, stdout=PIPE)

    def forward(self, local_port: int, remote_port: int) -> Popen:
        """
        Spawn a process that forwards the specified remote port to the specified local port.
        """
        args = [
            *self.prefix(),
            "-fN",
            "-L",
            # https://phoenixnap.com/kb/ssh-port-forwarding
            f"{local_port}:{self.host.host}:{remote_port}",
            self.target(),
        ]
        return self._popen(args)
