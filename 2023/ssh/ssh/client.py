from dataclasses import dataclass
from pathlib import Path
from subprocess import check_output

from ssh.models import SshHost


@dataclass
class SshCliWrapper:
    identity: Path
    host: SshHost

    def exec(self, *command) -> bytes:
        args = [
            "ssh",
            "-o",
            # https://stackoverflow.com/a/61946687/
            # TODO: relace this with accept-new and/or handle host key checking...
            "StrictHostKeyChecking=no",
            "-i",
            str(self.identity.absolute()),
            "-p",
            str(self.host.port),
            f"{self.host.user}@{self.host.host}",
            *command,
        ]
        return check_output(args)
