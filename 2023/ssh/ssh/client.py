import logging
from dataclasses import dataclass
from pathlib import Path
from subprocess import Popen, check_output

log = logging.getLogger(__name__)


from ssh.models import SshHost


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
            "StrictHostKeyChecking=accept-new",
        ]

    def target(self):
        """
        Target of SSH commands and port forwarding, e.g. `dan@localhost`.
        """
        return f"{self.host.user}@{self.host.host}"

    def _parameters(self):
        return [
            "-o",
            # https://stackoverflow.com/a/61946687/
            # TODO: clean up host keys. This is needed for port forwarding.
            "StrictHostKeyChecking=accept-new",
            "-i",
            str(self.identity.absolute()),
            "-p",
            str(self.host.port),
        ]

    def _hostname(self):
        return f"{self.host.user}@{self.host.host}"

    def exec(self, *command) -> bytes:
        args = [
            "ssh",
            *self._parameters(),
            self._hostname(),
            *command,
        ]
        log.warning("exec %s", args)
        return check_output(args)

    def exec_background(self, *command) -> Popen:
        # TODO: dedup with self.exec
        args = [
            "ssh",
            *self._parameters(),
            self._hostname(),
            *command,
        ]
        log.warning("exec_background %s", args)
        return Popen(args)

    def popen(self, args: list[str]) -> Popen:
        log.info("Popen %s", args)
        return Popen(args)

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
        return self.popen(args)
