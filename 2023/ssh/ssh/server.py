import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
import socket

import docker
from paramiko import AUTH_FAILED, AUTH_SUCCESSFUL, OPEN_SUCCEEDED, ServerInterface, RSAKey, Transport

from ssh.models import SshHost

log = logging.getLogger(__name__)


def run_dockerized_server(host_config: SshHost, public_key: str):
    assert (
        host_config.port == 2222
    ), "https://github.com/linuxserver/docker-openssh-server/issues/30"
    client = docker.from_env()
    container = client.containers.run(
        "linuxserver/openssh-server:version-9.3_p2-r0",
        environment={
            "PUID": 1000,
            "PGID": 1000,
            "TZ": "Etc/UTC",
            "PUBLIC_KEY": public_key,
            "USER_NAME": host_config.user,
            # https://github.com/linuxserver/docker-openssh-server/issues/30#issuecomment-1525103465
            "LISTEN_PORT": host_config.port,
            "LOG_STDOUT": True,
        },
        ports={host_config.port: host_config.port},
        hostname=host_config.host,
        detach=True,
    )

    return container


# https://stackoverflow.com/q/68768419/
# https://docs.paramiko.org/en/3.3/api/server.html
# https://github.com/paramiko/paramiko/blob/main/demos/demo_server.py
# TODO: imitate https://github.com/kryptographik/ShuSSH/blob/master/shusshd.py ?
@dataclass
class Server(ServerInterface):
    user: str
    event: threading.Event = field(default_factory=threading.Event)

    def check_channel_request(self, kind, channelID):
        return OPEN_SUCCEEDED

    def get_allowed_auths(self, username):
        return "publickey"

    def check_auth_publickey(self, username, key):
        log.info("Check auth for %s", username)
        if username == self.user:
            return AUTH_SUCCESSFUL
        # TODO: actually check a key!
        return AUTH_FAILED

    def check_port_forward_request(self, address, port):
        raise NotImplementedError()

    def cancel_port_forward_request(self, address, port):
        raise NotImplementedError()

    def check_channel_pty_request(
        self, channel, term, width, height, pixelwidth, pixelheight, modes
    ):
        return True

    def check_channel_shell_request(self, channel):
        return True

    def check_channel_exec_request(self, channel, command):
        print(f"Command: {command}")
        # raise Exception(f"Command: {command}")
        log.info("Running `%s`", command)
        self.event.set()
        return True

    def get_banner(self):
        return (f"Welcome, {self.user}!\n\r", "EN")


def run_server(user: str, port: int, private_key: Path):
    host_key = RSAKey(filename=str(private_key.absolute()))
    ctx = Server(user=user)

    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    sock.listen(100)
    log.info("Listening for connection")
    while True:
        client, addr = sock.accept()
        log.info("Listening for SSH connections")
        server = Transport(client)
        server.add_server_key(host_key)
        server.start_server(server=ctx)
        channel = server.accept(30)
        if channel is None:
            log.info("No auth request was made")
            exit(1)
        channel.send("[+]*****************  Welcome ***************** \n\r")
        channel.event.wait(5)
