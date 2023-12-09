from socket import socket
from subprocess import PIPE, run
from uuid import uuid4

import paramiko
from pytest import fixture

from ssh import (
    FabricClient,
    OpensshDockerWrapper,
    ParamikoServer,
    SshCliWrapper,
    SshHost,
)
from ssh.abstractions import SshClient, SshServer
from ssh.port import free_ports
from ssh.process import run_pipe
from ssh.rsa import private_public_key_pair

# TODO: there should be separate, module-wide fixtures private_key and public_key, the latter
# depending on the former.

# private key
#   --> private bytes
#       --> (+path) private key path
#   --> public bytes
#       --> (+path) public key path


@fixture
def key_pair():
    pair = private_public_key_pair()

    # https://docs.pytest.org/en/7.1.x/how-to/fixtures.html#yield-fixtures-recommended
    yield pair

    pair.private.unlink()
    pair.public.unlink()


@fixture
def ports():
    return free_ports()


@fixture
def user():
    return str(uuid4())[:8]


@fixture
def host(user):
    # TODO: figure out a way to use different ports when the SSH server is not OpenSSH Docker based.
    return SshHost("localhost", 2222, user)


@fixture
def port():
    """
    TODO
    """
    # https://www.scivision.dev/get-available-port-for-shell-script/
    with socket() as s:
        s.bind(("", 0))
        yield s.getsockname()[1]


@fixture(scope="session", autouse=True)
def build_docker_image():
    # Ensure image is built before tests start so that timeouts are meaningful:
    # https://stackoverflow.com/a/40155582/
    OpensshDockerWrapper(None, None, []).build()


@fixture(scope="session", autouse=True)
def kill_all_screens():
    """
    TODO: move this into `screen.py`?
    """
    # https://unix.stackexchange.com/a/94528
    args = ["killall", "SCREEN"]
    run_pipe(args)
    yield
    run_pipe(args)


# Replace the nasty decorator:
# - https://docs.pytest.org/en/7.1.x/how-to/fixtures.html#fixture-parametrize
# - https://docs.pytest.org/en/stable/example/parametrize.html#indirect-parametrization
# - https://stackoverflow.com/a/62685274/
# - https://stackoverflow.com/q/63158867/


@fixture(params=[ParamikoServer, OpensshDockerWrapper])
def server(request, key_pair, host, ports):
    server_type: type[SshServer] = request.param
    server = server_type.construct(host, key_pair.public.read_text(), [ports.remote])

    with server.serve():
        yield


@fixture(params=[FabricClient, SshCliWrapper])
def client(request, key_pair, host, server) -> SshClient:
    """
    Iterate through all known SSH client implementations, ensuring there is an active server
    running as well.
    """
    client_type: type[SshClient] = request.param
    return client_type.construct(key_pair.private, host)
