import logging
import os
import signal
import subprocess
from dataclasses import dataclass
from socket import socket
from typing import Optional

import pytest

from ssh.process import run_pipe

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Ports:
    local: int
    remote: int


def process_holding(port: int) -> Optional[int]:
    # Get only the port from lsof: https://stackoverflow.com/a/62453482/
    result = run_pipe(["lsof", "-t", "-i", f":{port}"])
    if result.returncode == 0:
        return int(result.stdout.decode().strip())
    return None


def ensure_free(port: int) -> None:
    """
    Ensure no processes are holding the specified port by killing (not just terminating) any that
    are.

    https://stackoverflow.com/a/17858114/
    """
    pid = process_holding(port)
    if pid:
        # TODO: this is too aggressive with Docker.
        log.info("Found process %s on port %s, killing it", pid, port)
        os.kill(pid, signal.SIGKILL)
    else:
        log.info("Port %s already free", port)


def free_ports() -> Ports:
    # https://www.scivision.dev/get-available-port-for-shell-script/
    with socket() as local_socket:
        local_socket.bind(("", 0))
        local_port = int(local_socket.getsockname()[1])

        with socket() as remote_socket:
            remote_socket.bind(("", 0))
            remote_port = int(remote_socket.getsockname()[1])

    return Ports(local_port, remote_port)


def assert_free(port: int):
    with pytest.raises(subprocess.CalledProcessError):
        process_holding(port)
