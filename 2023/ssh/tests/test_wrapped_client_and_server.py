import logging
import time
from dataclasses import dataclass
from subprocess import Popen
from uuid import uuid4

import pytest
import requests

from ssh import (
    KnownHostsClient,
    SshCliWrapper,
    SshHost,
    dockerized_server_safe,
    run_dockerized_server,
    ssh_cli_wrapper,
)
from ssh.netcat import NetcatClient

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProcessInfo:
    stderr: str
    stdout: str
    status: int


def get_info(process: Popen) -> ProcessInfo:
    # https://stackoverflow.com/a/38956698/
    (output, error) = process.communicate()

    # This makes the wait possible
    status = process.wait()
    return ProcessInfo(error.decode(), output.decode(), status)


def test_client_can_call_whoami_in_server(key_pair):
    host = SshHost("localhost", 2222, "dan")
    ssh_client = SshCliWrapper(key_pair["private"], host)
    server = run_dockerized_server(host, key_pair["public"].read_text())
    time.sleep(3)

    try:
        whoami = ssh_client.exec("whoami")
        assert whoami.decode().strip() == host.user
    finally:
        server.stop()


def test_client_can_touch_file_in_server(key_pair):
    host = SshHost("localhost", 2222, "dan")
    ssh_client = SshCliWrapper(key_pair["private"], host)
    server = run_dockerized_server(host, key_pair["public"].read_text())
    time.sleep(3)

    file_name = str(uuid4())

    try:
        ssh_client.exec("touch", file_name)
        ls = ssh_client.exec("ls", file_name)
        assert ls.decode().strip() == file_name
    finally:
        server.stop()


@pytest.mark.parametrize("executable", ["ls", "nc"])
def test_client_can_which_executable_in_server(key_pair, executable):
    host = SshHost("localhost", 2222, "dan")
    ssh_client = SshCliWrapper(key_pair["private"], host)
    server = run_dockerized_server(host, key_pair["public"].read_text())
    time.sleep(2)

    try:
        which_executable = ssh_client.exec("which", executable)
        assert which_executable.decode().strip().split("/")[-1] == executable
    finally:
        server.stop()


@pytest.mark.timeout(15)
def test_client_can_forward_port_from_server(key_pair, ports, user):
    netcat = NetcatClient()
    host = SshHost("localhost", 2222, user)
    ssh_cli = SshCliWrapper(key_pair.private, host)
    KnownHostsClient().reset(host)

    response_body = f"\nHi from SSH server: {uuid4()}\n"
    netcat_command = netcat.ssh_exec(response_body, ports.remote)

    with dockerized_server_safe(host, key_pair.public, [ports.remote]):
        # TODO: find polling mechanism.
        time.sleep(3)

        with ssh_cli_wrapper(key_pair.private, host) as ssh_cli:
            ssh_cli.forward(ports.local, ports.remote)
            netcat_proc = ssh_cli.exec(*netcat_command)

            # TODO: find polling mechanism for finding out if netcat is ready.
            time.sleep(1)

            response = requests.get(f"http://{host.host}:{ports.local}/{uuid4()}")
            request = response.request
            response.raise_for_status()

            # TODO: figure out how to make sure newlines match.
            assert response.text.strip() == response_body.strip()

            netcat_info = get_info(netcat_proc)
            log.info("netcat info: %s", netcat_info)
            assert all(
                f"{name}: {value}" in netcat_info.stdout
                # Obtain raw request: https://stackoverflow.com/a/60058128/
                for name, value in request.headers.items()
            )

            assert f"{request.method} {request.path_url}" in netcat_info.stdout
