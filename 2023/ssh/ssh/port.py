import subprocess
from dataclasses import dataclass
from socket import socket

import pytest


@dataclass(frozen=True)
class Ports:
    local: int
    remote: int


def process_holding(port: int):
    # Get only the port from lsof: https://stackoverflow.com/a/62453482/
    return subprocess.check_output(["lsof", "-t", "-i", f":{port}"])


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
