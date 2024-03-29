from socket import socket
from uuid import uuid4

import paramiko
from pytest import fixture

from ssh import FabricClient, OpensshDockerWrapper, SshCliWrapper, SshHost
from ssh.abstractions import SshClient, SshServer
from ssh.port import free_ports
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
    return SshHost("localhost", 2222, user)


@fixture
def ssh(key_pair):
    # https://gist.github.com/cschwede/3e2c025408ab4af531651098331cce45
    host_key = paramiko.RSAKey(filename=key_pair["private"])  # key_pair["private"])
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", 0))

    sock.listen(100)
    client, addr = sock.accept()

    t = paramiko.Transport(client)
    t.set_gss_host(socket.getfqdn(""))
    t.load_server_moduli()
    t.add_server_key(host_key)
    server = Server()
    t.start_server(server=server)
    # Wait 30 seconds for a command
    server.event.wait(1000)

    # https://stackoverflow.com/a/1365284/
    yield {"port": sock.getsockname()[1], **key_pair}

    t.close()


@fixture
def port():
    """
    TODO
    """
    # https://www.scivision.dev/get-available-port-for-shell-script/
    with socket() as s:
        s.bind(("", 0))
        yield s.getsockname()[1]


# Replace the nasty decorator:
# - https://docs.pytest.org/en/7.1.x/how-to/fixtures.html#fixture-parametrize
# - https://docs.pytest.org/en/stable/example/parametrize.html#indirect-parametrization


@fixture(params=[OpensshDockerWrapper])
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
