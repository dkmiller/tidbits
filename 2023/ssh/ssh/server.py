import logging
import socket
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from subprocess import PIPE, run
from threading import Event
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
from ssh.process import kill, popen

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
class ParamikoServer(ServerBase, ServerInterface, SshServer):
    private_key: Path = None
    event: Event = field(default_factory=Event)

    def check_channel_request(self, kind, channelID):
        return OPEN_SUCCEEDED

    def get_allowed_auths(self, username):
        return "publickey"

    def check_auth_publickey(self, username, key):
        log.info("Check auth for %s", username)
        if username == self.host.user:
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

    # def check_channel_shell_request(self, channel):
    #     self.event.set()
    #     return True

    def check_channel_exec_request(self, channel, command):
        log.info("Running `%s`", command)

        result = run(command, shell=True, stdout=PIPE, stderr=PIPE)
        log.info("Result: %s", result)
        channel.send(result.stdout)
        channel.send_stderr(result.stderr)
        channel.send_exit_status(result.returncode)
        self.event.set()
        return True

    # def get_banner(self):
    #     return (f"Welcome, {self.host.user}!\n\r", "EN")

    # https://gist.github.com/cschwede/3e2c025408ab4af531651098331cce45
    # https://stackoverflow.com/a/68791717/
    def run(self):
        self.known_hosts.reset(self.host)
        sock = socket.socket()
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        log.info("Listening on %s", self.host)
        # https://stackoverflow.com/a/1365284/
        sock.bind(("127.0.0.1", self.host.port))
        sock.listen(100)
        log.info("Listening for connection")
        while True:
            client, addr = sock.accept()
            log.info("Listening for SSH connections")
            server = Transport(client)
            host_key = RSAKey(filename=str(self.private_key.absolute()))
            server.add_server_key(host_key)
            server.start_server(server=self)
            channel = server.accept(30)
            if channel is None:
                log.info("No auth request was made")
                exit(1)
            channel.event.wait(1)
            channel.close()

    @contextmanager
    def serve(self):
        from ssh.rsa import private_public_key_pair

        server_pair = private_public_key_pair()
        args = (
            "ssh-testing",
            "serve",
            self.public_key,
            str(server_pair.private.absolute()),
            "--host",
            self.host.host,
            "--port",
            str(self.host.port),
            "--user",
            self.host.user,
        )

        from ssh.port import ensure_free

        ensure_free(self.host.port)

        process = popen(args)
        import time

        time.sleep(0.5)
        try:
            yield
        finally:
            server_pair.private.unlink()
            server_pair.public.unlink()

            # (output, error) = process.communicate(timeout=0.1)

            kill(process)
            ensure_free(self.host.port)

            # log.info("Status: %s\nStdout: %s\nStderr: %s", result.status, output, error)


## Testing:
# pytest -k 'test_client_can_call_whoami_in_server and SshCliWrapper and ParamikoServer'
