import inspect
import logging
import shlex
from functools import wraps
from uuid import uuid4

import pytest
import requests

from ssh import FabricClient, NetcatClient, OpensshDockerWrapper, SshCliWrapper
from ssh.abstractions import SshClient, SshServer

log = logging.getLogger(__name__)


# TODO: this should be much simpler via indirect parameterization;
# - https://docs.pytest.org/en/stable/example/parametrize.html#indirect-parametrization
# or fixture params:
# - https://docs.pytest.org/en/7.1.x/how-to/fixtures.html#fixture-parametrize


# TODO: clean this up. It is no longer used.
def client_server_pairs(func):
    """
    Python does not support "parameterized fixtures", and in addition testing SSH client/server
    interaction is challenging.

    This specialized decorator uses a combination of parameterization, existing fixtures, and
    type hacking to inject `client:SshClient` and `server:SshServer`

    TODO: find a way to abstract away the boilerplate client/server construction by modifying the
    wrapped function's signature:

    - https://stackoverflow.com/a/64447219/
    - https://docs.python.org/3/library/functools.html
    """

    # TODO: why? You can't have "parameterized fixtures".

    original_sig: inspect.Signature = inspect.signature(func)

    scrubbed_parameters = [
        p
        for p in original_sig.parameters.values()
        if p.name not in ["client", "server"]
    ]

    injected_params = ["key_pair", "ports", "host", "client_type", "server_type"]

    for name in injected_params:
        if name not in original_sig.parameters:
            scrubbed_parameters += [
                inspect.Parameter(name=name, kind=inspect.Parameter.KEYWORD_ONLY)
            ]

    sig = original_sig.replace(
        parameters=scrubbed_parameters
        # + [
        #     inspect.Parameter(name="key_pair", kind=inspect.Parameter.KEYWORD_ONLY),
        #     inspect.Parameter(name="ports", kind=inspect.Parameter.KEYWORD_ONLY),
        #     inspect.Parameter(name="host", kind=inspect.Parameter.KEYWORD_ONLY),
        #     inspect.Parameter(name="client_type", kind=inspect.Parameter.KEYWORD_ONLY),
        #     inspect.Parameter(name="server_type", kind=inspect.Parameter.KEYWORD_ONLY),
        # ]
    )

    # raise Exception(inspect.signature(func))
    # Cartesian product via parametrize: https://stackoverflow.com/q/22171681/
    @pytest.mark.parametrize("client_type", [FabricClient, SshCliWrapper])
    @pytest.mark.parametrize("server_type", [OpensshDockerWrapper])
    @pytest.mark.timeout(10)
    @wraps(func)
    # pytest uses signature(*) to determine function signature:
    # - https://stackoverflow.com/q/51125966/2543689
    # - https://github.com/pytest-dev/pytest/blob/ef1b91ba8721e85430eead4c2849d8e072fe8350/src/_pytest/python.py#L893
    # - https://github.com/pytest-dev/pytest/blob/ef1b91ba8721e85430eead4c2849d8e072fe8350/src/_pytest/compat.py#L147
    def wrapper(*args, **kwargs):
        # key_pair, ports, host, client_type, server_type,
        server = kwargs["server_type"].construct(
            kwargs["host"],
            kwargs["key_pair"].public.read_text(),
            [kwargs["ports"].remote],
        )
        client = kwargs["client_type"](kwargs["key_pair"].private, kwargs["host"])

        # with server.construct(host, key_pair.public.read_text(), [ports.remote]).serve():
        #     ssh = client(key_pair.private, host)

        # raise Exception(f"{injected_params} {list(original_sig.parameters.keys())}")

        for name in injected_params:
            if name not in original_sig.parameters:
                del kwargs[name]

        with server.serve():
            return func(*args, client=client, **kwargs)

    wrapper.__signature__ = sig

    # Change function signature: https://stackoverflow.com/a/33112180/

    # https://codereview.stackexchange.com/a/272072

    return wrapper


@pytest.mark.timeout(3)
def test_client_can_call_whoami_in_server(client):
    whoami = client.exec("whoami")
    assert whoami.stdout.strip() == client.host.user


@pytest.mark.timeout(3)
def test_client_can_touch_file_in_server(client):
    file_name = str(uuid4())
    client.exec("touch", file_name)
    ls = client.exec("ls", file_name)
    assert ls.stdout.strip() == file_name


@pytest.mark.timeout(3)
def test_client_can_run_uname_in_server(client):
    uname = client.exec("uname", "-a")
    prefix = uname.stdout.split()[0]
    assert prefix in ["Darwin", "Linux"]


@pytest.mark.timeout(4)
def test_client_can_write_to_file_in_server(client):
    file_name = str(uuid4())
    file_contents = str(uuid4())

    client.exec("bash", "-c", shlex.quote(f"echo {file_contents} > {file_name}"))
    cat_contents = client.exec("cat", file_name)
    assert cat_contents.stdout.strip() == file_contents


@pytest.mark.parametrize(
    "executable", ["bash", "curl", "echo", "ls", "nc", "screen", "wget", "which"]
)
@pytest.mark.timeout(4)
def test_client_can_run_which_in_server(client, executable):
    which = client.exec("which", executable)
    assert which.stdout.strip().split("/")[-1] == executable


@pytest.mark.timeout(10)
def test_client_can_forward_port_from_server(client, ports):
    netcat = NetcatClient()

    response_body = f"Hi from SSH server: {uuid4()}"
    netcat_command = netcat.ssh_exec(response_body, ports.remote)

    unescaped_command = " ".join(netcat_command)
    log.info("Running %s", unescaped_command)
    # https://stackoverflow.com/a/50651839/
    client.exec(
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

    assert ".netcat" in client.exec("screen -ls").stdout
    with client.forward(ports.local, remote_port=ports.remote):
        response = requests.get(
            f"http://{client.host.host}:{ports.local}/{uuid4()}", timeout=1
        )

    response.raise_for_status()
    # TODO: why is the "strip" required?
    assert response.text.strip() == response_body.strip()
    request = response.request

    netcat_logs = client.exec("cat", "screen.logs").stdout

    assert all(
        f"{name}: {value}" in netcat_logs for name, value in request.headers.items()
    )

    assert f"{request.method} {request.path_url}" in netcat_logs


@pytest.mark.timeout(10)
def test_remote_screen_session_with_netcat_and_curl(client, ports):
    """
    Connect local -> remote server. Start a screen session with netcat exposed on a specified port
    remotely. Curl that screen session remotely.

    (All this needed to ensure remote netcat server is behaving properly, before bringing port
    forwarding into the mix.)
    """
    netcat = NetcatClient()

    response_body = f"Hi from SSH server: {uuid4()}"
    netcat_command = netcat.ssh_exec(response_body, ports.remote)

    unescaped_command = " ".join(netcat_command)
    log.info("Running %s", unescaped_command)
    client.exec("screen", "-S", "netcat", "-m", "-d", unescaped_command)

    assert ".netcat" in client.exec("screen", "-ls").stdout

    curl = client.exec("curl", "-v", f"http://{client.host.host}:{ports.remote}/path")
    assert curl.status == 0
    assert response_body in curl.stdout
    assert "Excess found in a read" not in curl.stderr
