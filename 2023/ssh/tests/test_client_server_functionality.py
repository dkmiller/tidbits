import logging
import shlex
from functools import wraps
from uuid import uuid4

import pytest
import requests

from ssh import FabricClient, NetcatClient, SshCliWrapper, dockerized_server_safe

log = logging.getLogger(__name__)


def standard(func):
    """
    Standard set of decorators for SSH client/server functionality testing. Injects all client/
    server implementations along with a standard timeout.

    TODO: find a way to abstract away the boilerplate client/server construction by modifying the
    wrapped function's signature:

    - https://stackoverflow.com/a/64447219/
    - https://docs.python.org/3/library/functools.html
    """

    # Cartesian product via parametrize: https://stackoverflow.com/q/22171681/
    @pytest.mark.parametrize("client", [FabricClient, SshCliWrapper])
    @pytest.mark.parametrize("server", [dockerized_server_safe])
    @pytest.mark.timeout(10)
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


@standard
def test_client_can_call_whoami_in_server(client, server, key_pair, ports, host):
    with server(host, key_pair.public, [ports.remote]):
        ssh = client(key_pair.private, host)
        whoami = ssh.exec("whoami")
        assert whoami.stdout.strip() == host.user


@standard
def test_client_can_touch_file_in_server(client, server, key_pair, ports, host):
    with server(host, key_pair.public, [ports.remote]):
        ssh = client(key_pair.private, host)
        file_name = str(uuid4())
        ssh.exec("touch", file_name)
        ls = ssh.exec("ls", file_name)
        assert ls.stdout.strip() == file_name


@standard
def test_client_can_run_uname_in_server(client, server, key_pair, ports, host):
    with server(host, key_pair.public, [ports.remote]):
        ssh = client(key_pair.private, host)
        uname = ssh.exec("uname", "-a")
        prefix = uname.stdout.split()[0]
        assert prefix in ["Darwin", "Linux"]


@standard
def test_client_can_write_to_file_in_server(client, server, key_pair, ports, host):
    with server(host, key_pair.public, [ports.remote]):
        ssh = client(key_pair.private, host)
        file_name = str(uuid4())
        file_contents = str(uuid4())

        ssh.exec("bash", "-c", shlex.quote(f"echo {file_contents} > {file_name}"))
        cat_contents = ssh.exec("cat", file_name)
        assert cat_contents.stdout.strip() == file_contents


@pytest.mark.parametrize(
    "executable", ["bash", "curl", "echo", "ls", "nc", "screen", "wget", "which"]
)
@standard
def test_client_can_run_which_in_server(
    client, server, key_pair, executable, ports, host
):
    with server(host, key_pair.public, [ports.remote]):
        ssh = client(key_pair.private, host)
        which = ssh.exec("which", executable)
        assert which.stdout.strip().split("/")[-1] == executable


@standard
def test_client_can_forward_port_from_server(client, server, key_pair, ports, host):
    netcat = NetcatClient()

    response_body = f"Hi from SSH server: {uuid4()}"
    netcat_command = netcat.ssh_exec(response_body, ports.remote)

    with server(host, key_pair.public, [ports.remote]):
        ssh = client(key_pair.private, host)
        unescaped_command = " ".join(netcat_command)
        log.info("Running %s", unescaped_command)
        # https://stackoverflow.com/a/50651839/
        ssh.exec(
            "screen",
            "-S",
            "netcat",
            "-L",
            "-Logfile",
            "screen.logs",
            "-m",
            "-d",
            unescaped_command,
        )

        assert ".netcat" in ssh.exec("screen -ls").stdout
        with ssh.forward(ports.local, remote_port=ports.remote):
            response = requests.get(
                f"http://{host.host}:{ports.local}/{uuid4()}", timeout=1
            )

        response.raise_for_status()
        # TODO: why is the "strip" required?
        assert response.text.strip() == response_body.strip()
        request = response.request

        netcat_logs = ssh.exec("cat", "screen.logs").stdout

        assert all(
            f"{name}: {value}" in netcat_logs for name, value in request.headers.items()
        )

        assert f"{request.method} {request.path_url}" in netcat_logs


@standard
def test_remote_screen_session_with_netcat_and_curl(
    client, server, key_pair, ports, host
):
    """
    Connect local -> remote server. Start a screen session with netcat exposed on a specified port
    remotely. Curl that screen session remotely.

    (All this needed to ensure remote netcat server is behaving properly, before bringing port
    forwarding into the mix.)
    """
    netcat = NetcatClient()

    response_body = f"Hi from SSH server: {uuid4()}"
    netcat_command = netcat.ssh_exec(response_body, ports.remote)

    with server(host, key_pair.public, [ports.remote]):
        ssh = client(key_pair.private, host)
        with ssh.forward(ports.local, remote_port=ports.remote):
            unescaped_command = " ".join(netcat_command)
            log.info("Running %s", unescaped_command)
            ssh.exec("screen", "-S", "netcat", "-m", "-d", unescaped_command)

            assert ".netcat" in ssh.exec("screen", "-ls").stdout

            curl = ssh.exec("curl", "-v", f"http://{host.host}:{ports.remote}/path")
            assert curl.status == 0
            assert response_body in curl.stdout
            assert "Excess found in a read" not in curl.stderr
