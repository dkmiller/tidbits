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

from ssh.abstractions import SshServer
from ssh.known_hosts import KnownHostsClient
from ssh.models import SshHost

log = logging.getLogger(__name__)


SLEEP_SECONDS = 0.1


@dataclass
class ServerBase:
    host: SshHost
    public_key: str
    ports_to_forward: list[int]

    client: docker.DockerClient = field(default_factory=docker.from_env)
    known_hosts: KnownHostsClient = field(default_factory=KnownHostsClient)

    @classmethod
    def construct(cls, host: SshHost, public_key: str, ports_to_forward: list[int]):
        return cls(host, public_key, ports_to_forward)


class OpensshDockerWrapper(ServerBase, SshServer):
    def __post_init__(self):
        assert (
            self.host.port == 2222
        ), "https://github.com/linuxserver/docker-openssh-server/issues/30"

    def build(self):
        """
        (Blocking) Ensure the necessary Docker image is built.

        Warning: this may take 20-30 seconds.
        """
        dockerfile = str(Path(__file__).parent.parent)
        log.info("Building Docker image from %s", dockerfile)
        self.client.images.build(path=dockerfile)

    def start(self) -> Container:
        ports_dict = {self.host.port: self.host.port}
        for port in self.ports_to_forward:
            ports_dict[port] = port
        log.info("Ports: %s", ports_dict)
        container: Union[None, Container] = None
        while container is None:
            try:
                container = self.client.containers.run(
                    "ssh",
                    environment={
                        "PUBLIC_KEY": self.public_key,
                        "USER_NAME": self.host.user,
                        # https://github.com/linuxserver/docker-openssh-server/issues/30#issuecomment-1525103465
                        "LISTEN_PORT": self.host.port,
                    },
                    ports=ports_dict,
                    hostname=self.host.host,
                    detach=True,
                )  # type: ignore
            except Exception as e:
                log.warning("Failure starting container: %s", e)
        log.info("Spawned container %s", container.name)
        return container

    def wait(self, container: Container):
        prefix_length = 0
        while b"Server listening on" not in container.logs():
            logs = container.logs().decode()
            log.warning(
                "Container %s not ready yet: %s",
                container.name,
                logs[prefix_length:],
            )
            time.sleep(SLEEP_SECONDS)
            prefix_length = len(logs)
            container.reload()
        log.info("Container %s is ready!", container.name)

    @contextmanager
    def serve(self):
        self.build()

        self.cleanup_containers()
        self.known_hosts.reset(self.host)

        container = self.start()
        self.wait(container)

        try:
            yield
        finally:
            self.known_hosts.reset(self.host)
            self.cleanup_containers()

    def cleanup_containers(self):
        """
        Clean up all conflicting Docker images. Can be invoked before or after startup.
        """
        # TODO: cleanup only containers listening to the same port.
        while self.client.containers.list():
            for other in self.client.containers.list():
                log.warning("Stopping %s", other.name)
                other.stop(timeout=1)


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
