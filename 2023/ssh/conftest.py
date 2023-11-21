import socket

import paramiko
from pytest import fixture

from ssh import rsa
from ssh.paramiko import Server
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
    private_key_file, public_key_file = private_public_key_pair()

    # https://docs.pytest.org/en/7.1.x/how-to/fixtures.html#yield-fixtures-recommended
    yield {
        "public": public_key_file.absolute(),
        "private": private_key_file.absolute(),
    }
    private_key_file.unlink()
    public_key_file.unlink()


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
