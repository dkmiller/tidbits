import logging
import shlex
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from subprocess import PIPE, Popen

from ssh.abstractions import SshClient
from ssh.models import SshHost
from ssh.process import kill

log = logging.getLogger(__name__)


@dataclass
class SshCliWrapper(SshClient):
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


# from fabric import Config, Connection
# from paramiko.config import SSHConfig

# import sys

# import signal


# from multiprocessing import Process, Queue


# # TODO: is this really necessary? https://stackoverflow.com/q/67023124/
# def __need_global(host: str, user: str, port: int, config: Config, command: str):
#     conn = Connection(host, user=user, port=port, config=config)
#     return conn.run(command, hide=True)


# @dataclass
# class ParamikoClient(SshClient):
#     identity: Path
#     host: SshHost
#     processes: list[Process] = field(default_factory=list)

#     def exec(self, *args):
#         # https://phoenixnap.com/kb/ssh-config
#         ssh_conf = SSHConfig.from_text(
#             f"""
#         Host localhost
#             HostName localhost
#             User {self.host.user}
#             IdentityFile {str(self.identity)}
#         """
#         )
#         config = Config(ssh_config=ssh_conf)
#         command = " ".join(map(shlex.quote, args))
#         # conn = Connection(
#         #     self.host.host, user=self.host.user, port=self.host.port, config=config
#         # )

#         rv = Process(target=__need_global, args=(self.host.host, self.host.user, self.host.port, config, command))
#         self.processes.append(rv)
#         rv.start()
#         return rv


#         # # https://github.com/fabric/fabric/issues/2071
#         # result = conn.run(" ".join(map(shlex.quote, args)), hide=True)

#         # return Process()

#         # assert result.exited == 0
#         # assert result.stdout.strip().lower() == "linux"


#     def forward(self, local_port: int, remote_port: int) -> Popen:
#         raise NotImplementedError()


# @contextmanager
# def ssh_paramiko_wrapper(identity: Path, host: SshHost):
#     client = ParamikoClient(identity, host)
#     try:
#         yield client
#     finally:
#         for process in client.processes:
#             kill(process)


# def test_ssh_with_dockerized_server():
#     ssh_host = SshHost("localhost", 2222, "dan")
#     private, public = private_public_key_pair()
#     container = run_dockerized_server(ssh_host, public.read_text())
#     time.sleep(3)

#     # https://phoenixnap.com/kb/ssh-config
#     ssh_conf = SSHConfig.from_text(
#         f"""
#     Host localhost
#         HostName localhost
#         User {ssh_host.user}
#         IdentityFile {str(private)}
#     """
#     )

#     config = Config(ssh_config=ssh_conf)
#     conn = Connection(
#         ssh_host.host, user=ssh_host.user, port=ssh_host.port, config=config
#     )

#     # https://github.com/fabric/fabric/issues/2071
#     result = conn.run("uname -s", hide=True)

#     assert result.exited == 0
#     assert result.stdout.strip().lower() == "linux"

#     result = conn.run("whoami", hide=True)

#     assert result.exited == 0
#     assert result.stdout.strip().lower() == ssh_host.user

#     container.stop()
