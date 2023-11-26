import logging
import socket
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import docker
from docker.models.containers import Container
from paramiko import (
    AUTH_FAILED,
    AUTH_SUCCESSFUL,
    OPEN_SUCCEEDED,
    RSAKey,
    ServerInterface,
    Transport,
)

from ssh.known_hosts import KnownHostsClient
from ssh.models import SshHost

log = logging.getLogger(__name__)


SLEEP_SECONDS = 0.1


def run_dockerized_server(
    host_config: SshHost, public_key: str, ports: list[int] = []
) -> Container:
    assert (
        host_config.port == 2222
    ), "https://github.com/linuxserver/docker-openssh-server/issues/30"
    client = docker.from_env()
    client.images.build(path=str(Path(__file__).parent.parent))
    ports_dict = {host_config.port: host_config.port}
    for port in ports:
        ports_dict[port] = port
    log.info("Ports: %s", ports_dict)
    container = None
    # TODO: cleanup only containers listening to the same port.
    while client.containers.list():
        for other in client.containers.list():
            log.warning("Stopping %s", other.name)
            other.stop(timeout=1)
    while container is None:
        try:
            container = client.containers.run(
                "ssh",
                environment={
                    "PUBLIC_KEY": public_key,
                    "USER_NAME": host_config.user,
                    # https://github.com/linuxserver/docker-openssh-server/issues/30#issuecomment-1525103465
                    "LISTEN_PORT": host_config.port,
                },
                ports=ports_dict,
                hostname=host_config.host,
                detach=True,
            )
        except Exception as e:
            log.warning("Failure starting container: %s", e)

    log.info("Spawned container %s", container.name)

    container: Container

    while b"Server listening on" not in container.logs():
        log.warning(
            "Container %s not ready yet: %s", container.name, container.logs().decode()
        )
        time.sleep(SLEEP_SECONDS)
        container.reload()
    log.info("Container %s is ready!", container.name)

    return container  # type: ignore


@contextmanager
def dockerized_server_safe(
    host_config: SshHost, public_key: Union[Path, str], ports: list[int]
):
    known_hosts = KnownHostsClient()
    known_hosts.reset(host_config)

    if isinstance(public_key, Path):
        public_key = public_key.read_text()
    container = run_dockerized_server(host_config, public_key, ports)

    try:
        yield container
    finally:
        log.info("Stopping container %s", container.name)
        container.stop(timeout=1)
        while container.status == "running":
            log.warning("Container %s still running", container.name)
            container.reload()
            time.sleep(SLEEP_SECONDS)
        known_hosts.reset(host_config)


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
        self.event.set()
        return True

    def check_channel_exec_request(self, channel, command):
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
