from dataclasses import dataclass
from pathlib import Path
from subprocess import check_output, Popen

from ssh.models import SshHost


@dataclass
class SshCliWrapper:
    identity: Path
    host: SshHost

    def _parameters(self):
        return [
            "-o",
            # https://stackoverflow.com/a/61946687/
            # TODO: relace this with accept-new and/or handle host key checking...
            "StrictHostKeyChecking=no",
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
        return check_output(args)

    def exec_background(self, *command) -> Popen:
        # TODO: dedup with self.exec
        args = [
            "ssh",
            *self._parameters(),
            self._hostname(),
            *command,
        ]
        return Popen(args)

    def forward(self, local_port: int, destination_port: int) -> Popen:
        args = [
            "ssh",
            *self._parameters(),
            "-fN",
            "-L",
            f"{local_port}:localhost:{destination_port}",
            self._hostname(),
        ]
        return Popen(args)
        #         ssh -o ConnectTimeout=3 -fN -L $LOCALHOST $TARGET &
        # return check_output(args)

        # ssh_target = f"{self.user}@{self.host}"
        # localhost = f"{port}:localhost:{port}"
        # url = f"http://localhost:{port}/?token={token}"
        # ssh_tunnel = common.ONE_BRAIN_CLI_ROOT / "one_brain/scripts/ssh_tunnel.sh"
        # command = f"/bin/bash {ssh_tunnel} {url} {localhost} {ssh_target} {port}"


# ssh -L local_port:destination_server_ip:remote_port ssh_server_hostname

# https://phoenixnap.com/kb/ssh-port-forwarding
