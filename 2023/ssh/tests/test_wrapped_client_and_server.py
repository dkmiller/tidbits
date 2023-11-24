import time
from uuid import uuid4

import pytest
import requests

from ssh import dockerized_server_safe, ssh_cli_wrapper, wait
from ssh.netcat import NetcatClient

TIMEOUT = pytest.mark.timeout(10)


def test_client_can_call_whoami_in_server(key_pair, ports, host):
    with dockerized_server_safe(host, key_pair.public, [ports.remote]):
        with ssh_cli_wrapper(key_pair.private, host) as ssh_cli:
            whoami = wait(ssh_cli.exec("whoami"))
            assert whoami.stdout.strip() == host.user


@TIMEOUT
def test_client_can_touch_file_in_server(key_pair, ports, host):
    with dockerized_server_safe(host, key_pair.public, [ports.remote]):
        with ssh_cli_wrapper(key_pair.private, host) as ssh_cli:
            file_name = str(uuid4())
            wait(ssh_cli.exec("touch", file_name))
            ls = wait(ssh_cli.exec("ls", file_name))
            assert ls.stdout.strip() == file_name


@pytest.mark.parametrize("executable", ["bash", "ls", "nc", "which"])
@TIMEOUT
def test_client_can_run_which_in_server(key_pair, executable, ports, host):
    with dockerized_server_safe(host, key_pair.public, [ports.remote]):
        with ssh_cli_wrapper(key_pair.private, host) as ssh_cli:
            which = wait(ssh_cli.exec("which", executable))
            assert which.stdout.strip().split("/")[-1] == executable


@TIMEOUT
def test_client_can_forward_port_from_server(key_pair, ports, host):
    netcat = NetcatClient()

    response_body = f"Hi from SSH server: {uuid4()}"
    netcat_command = netcat.ssh_exec(response_body, ports.remote)

    with dockerized_server_safe(host, key_pair.public, [ports.remote]):
        with ssh_cli_wrapper(key_pair.private, host) as ssh_cli:
            ssh_cli.forward(ports.local, ports.remote)
            netcat_proc = ssh_cli.exec(*netcat_command)

            # TODO: find polling mechanism for finding out if netcat is ready.
            time.sleep(1)

            response = requests.get(f"http://{host.host}:{ports.local}/{uuid4()}")
            request = response.request
            response.raise_for_status()

            # TODO: figure out how to make sure newlines match.
            assert response.text == response_body

            netcat_info = wait(netcat_proc)
            assert all(
                f"{name}: {value}" in netcat_info.stdout
                # Obtain raw request: https://stackoverflow.com/a/60058128/
                for name, value in request.headers.items()
            )

            assert f"{request.method} {request.path_url}" in netcat_info.stdout
