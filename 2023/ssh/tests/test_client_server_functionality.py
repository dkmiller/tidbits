import logging
import shlex
from getpass import getuser
from uuid import uuid4

import pytest
import requests

from ssh import NetcatClient, ScreenClient

log = logging.getLogger(__name__)


@pytest.mark.timeout(4)
def test_client_can_call_whoami_in_server(client):
    whoami = client.exec("whoami")
    assert whoami.ok_stdout() in [client.host.user, getuser()]


@pytest.mark.timeout(3)
def test_client_can_run_uname_in_server(client):
    uname = client.exec("uname", "-a")
    prefix = uname.ok_stdout().split()[0]
    assert prefix in ["Darwin", "Linux"]


@pytest.mark.timeout(4)
def test_client_can_touch_file_in_server(client):
    file_name = str(uuid4())
    client.exec("touch", file_name)
    ls = client.exec("ls", file_name)
    assert ls.ok_stdout() == file_name


@pytest.mark.parametrize(
    "executable", ["bash", "curl", "echo", "ls", "nc", "screen", "which"]
)
@pytest.mark.timeout(5)
def test_client_can_run_which_in_server(client, executable):
    which = client.exec("which", executable)
    assert which.ok_stdout().split("/")[-1] == executable


@pytest.mark.timeout(4)
def test_client_when_executable_does_not_exist(client):
    executable = str(uuid4())
    result = client.exec(executable)
    # https://www.baeldung.com/linux/status-codes#command-not-found
    assert result.status == 127
    assert "command not found" in result.stderr
    assert executable in result.stderr
    assert not result.stdout.strip()


@pytest.mark.timeout(4)
def test_client_can_write_to_file_in_server(client):
    file_name = str(uuid4())
    file_contents = str(uuid4())

    client.exec("bash", "-c", shlex.quote(f"echo {file_contents} > {file_name}"))
    cat_contents = client.exec("cat", file_name)
    assert cat_contents.ok_stdout() == file_contents


@pytest.mark.timeout(10)
def test_remote_screen_session_with_netcat_and_curl(client, ports):
    """
    Connect local -> remote server. Start a screen session with netcat exposed on a specified port
    remotely. Curl that screen session remotely.

    (All this needed to ensure remote netcat server is behaving properly, before bringing port
    forwarding into the mix.)
    """
    netcat = NetcatClient()
    screen = ScreenClient()

    response_body = f"Hi from SSH server: {uuid4()}"
    netcat_command = netcat.ssh_exec(response_body, ports.remote)
    args = screen.session(netcat_command, "netcat")
    client.exec(*args)

    assert ".netcat" in client.exec("screen", "-ls").ok_stdout()

    curl = client.exec("curl", "-v", f"http://{client.host.host}:{ports.remote}/path")
    assert curl.status == 0
    assert response_body in curl.ok_stdout()
    assert "Excess found in a read" not in curl.stderr


@pytest.mark.timeout(10)
def test_client_can_forward_port_from_server(client, ports):
    netcat = NetcatClient()
    screen = ScreenClient()

    response_body = f"Hi from SSH server: {uuid4()}"
    netcat_command = netcat.ssh_exec(response_body, ports.remote)

    args = screen.session(netcat_command, "netcat", "screen.logs")
    client.exec(*args)

    assert ".netcat" in client.exec("screen -ls").ok_stdout()
    with client.forward(ports.local, remote_port=ports.remote):
        response = requests.get(
            f"http://{client.host.host}:{ports.local}/{uuid4()}", timeout=1
        )

    response.raise_for_status()
    # TODO: why is the "strip" required?
    assert response.text.strip() == response_body.strip()
    request = response.request

    netcat_logs = client.exec("cat", "screen.logs").ok_stdout()

    assert all(
        f"{name}: {value}" in netcat_logs for name, value in request.headers.items()
    )

    assert f"{request.method} {request.path_url}" in netcat_logs
