import asyncio
import logging
from functools import wraps
from uuid import uuid4

import pytest
import requests

from ssh import NetcatClient, dockerized_server_safe, ssh_cli_wrapper

log = logging.getLogger(__name__)


def standard(func):
    """
    Standard set of decorators for SSH client/server functionality testing. Injects all client/
    server implementations along with a standard timeout.
    """

    # Cartesian product via parametrize: https://stackoverflow.com/q/22171681/
    @pytest.mark.parametrize("client", [ssh_cli_wrapper])
    @pytest.mark.parametrize("server", [dockerized_server_safe])
    @pytest.mark.timeout(10)
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


@standard
@pytest.mark.asyncio
async def test_client_can_call_whoami_in_server(client, server, key_pair, ports, host):
    with server(host, key_pair.public, [ports.remote]):
        with client(key_pair.private, host) as ssh:
            whoami = await ssh.exec("whoami")
            assert whoami.strip() == host.user


@standard
@pytest.mark.asyncio
async def test_client_can_touch_file_in_server(client, server, key_pair, ports, host):
    with server(host, key_pair.public, [ports.remote]):
        with client(key_pair.private, host) as ssh:
            file_name = str(uuid4())
            await ssh.exec("touch", file_name)
            ls = await ssh.exec("ls", file_name)
            assert ls.strip() == file_name


@standard
@pytest.mark.asyncio
async def test_client_can_run_uname_in_server(client, server, key_pair, ports, host):
    with server(host, key_pair.public, [ports.remote]):
        with client(key_pair.private, host) as ssh:
            uname = await ssh.exec("uname", "-a")
            prefix = uname.split()[0]
            assert prefix in ["Darwin", "Linux"]


@pytest.mark.parametrize("executable", ["bash", "ls", "nc", "which"])
@standard
@pytest.mark.asyncio
async def test_client_can_run_which_in_server(
    client, server, key_pair, executable, ports, host
):
    with server(host, key_pair.public, [ports.remote]):
        with client(key_pair.private, host) as ssh:
            which = await ssh.exec("which", executable)
            assert which.strip().split("/")[-1] == executable


async def requests_get_with_retry(url: str):
    while True:
        try:
            return requests.get(url, timeout=1)
        except Exception as e:
            log.warning("Failure: %s", e)
            await asyncio.sleep(1)


@standard
@pytest.mark.asyncio
async def test_client_can_forward_port_from_server(
    client, server, key_pair, ports, host
):
    netcat = NetcatClient()

    response_body = f"Hi from SSH server: {uuid4()}"
    netcat_command = netcat.ssh_exec(response_body, ports.remote)

    with server(host, key_pair.public, [ports.remote]):
        with client(key_pair.private, host) as ssh:
            forward_coro = ssh.forward(ports.local, ports.remote)
            netcat_coro = ssh.exec(*netcat_command)
            request_coro = requests_get_with_retry(
                f"http://{host.host}:{ports.local}/{uuid4()}"
            )

            netcat_stdout, response, _ = await asyncio.gather(
                netcat_coro, request_coro, forward_coro, return_exceptions=True
            )
            response.raise_for_status()
            assert response.text == response_body
            # Obtain raw request: https://stackoverflow.com/a/60058128/
            request = response.request

            assert all(
                f"{name}: {value}" in netcat_stdout
                for name, value in request.headers.items()
            )

            assert f"{request.method} {request.path_url}" in netcat_stdout
