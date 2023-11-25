import time
from functools import wraps
from uuid import uuid4
import logging

import pytest
import requests

from ssh import NetcatClient, dockerized_server_safe, ssh_cli_wrapper, wait


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

import asyncio

@standard
@pytest.mark.asyncio
async def test_client_can_forward_port_from_server(client, server, key_pair, ports, host):
    netcat = NetcatClient()

    response_body = f"Hi from SSH server: {uuid4()}"
    netcat_command = netcat.ssh_exec(response_body, ports.remote)

    with server(host, key_pair.public, [ports.remote]):
        with client(key_pair.private, host) as ssh:
            coro = ssh.forward(ports.local, ports.remote)

            # loop = asyncio.get_event_loop()
            # https://medium.com/@burak.sezer/running-a-background-worker-in-python-with-asyncio-75231a1a9c45
            # task = loop.run_coroutine_threadsafe(ssh.forward(ports.local, ports.remote))
            # await asyncio.sleep(0)

            # ssh.forward(ports.local, ports.remote)
            netcat_proc = ssh.exec(*netcat_command)


            async def call_remote():

                response = None

                while not response:
                    try:
                        response = requests.get(f"http://{host.host}:{ports.local}/{uuid4()}", timeout=1)
                        response.raise_for_status()
                        request = response.request
                    except Exception as e:
                        log.warning("Failure: %s", e)
                        await asyncio.sleep(1)

                # coro.cancel()
                # netcat_proc.cancel()
                return response

            x = await asyncio.gather(coro, netcat_proc, call_remote(), return_exceptions=True)
            raise Exception(x)
            response = x[-1]

            # TODO: figure out how to make sure newlines match.
            assert response.text == response_body

            # netcat_info = wait(netcat_proc)
            # assert all(
            #     f"{name}: {value}" in netcat_info.stdout
            #     # Obtain raw request: https://stackoverflow.com/a/60058128/
            #     for name, value in request.headers.items()
            # )

            # assert f"{request.method} {request.path_url}" in netcat_info.stdout
