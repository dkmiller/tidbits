import logging
import time
from subprocess import Popen
from uuid import uuid4

import pytest
import requests

from ssh import (
    KnownHostsClient,
    SshCliWrapper,
    SshHost,
    dockerized_server_safe,
    run_dockerized_server,
)
from ssh.netcat import NetcatClient
from ssh.port import assert_free

log = logging.getLogger(__name__)


def show_output(process: Popen):
    # https://stackoverflow.com/a/38956698/
    (output, error) = process.communicate()

    # This makes the wait possible
    status = process.wait()
    log.info("Process %s status: %s", process.pid, status)
    log.info("Output: %s", output)
    log.info("Error: %s", error)


def test_client_can_call_whoami_in_server(key_pair):
    host = SshHost("localhost", 2222, "dan")
    ssh_client = SshCliWrapper(key_pair["private"], host)
    server = run_dockerized_server(host, key_pair["public"].read_text())
    time.sleep(3)

    try:
        whoami = ssh_client.exec("whoami")
        assert whoami.decode().strip() == host.user
    finally:
        server.stop()


def test_client_can_touch_file_in_server(key_pair):
    host = SshHost("localhost", 2222, "dan")
    ssh_client = SshCliWrapper(key_pair["private"], host)
    server = run_dockerized_server(host, key_pair["public"].read_text())
    time.sleep(3)

    file_name = str(uuid4())

    try:
        ssh_client.exec("touch", file_name)
        ls = ssh_client.exec("ls", file_name)
        assert ls.decode().strip() == file_name
    finally:
        server.stop()


@pytest.mark.parametrize("executable", ["ls", "nc"])
def test_client_can_which_executable_in_server(key_pair, executable):
    host = SshHost("localhost", 2222, "dan")
    ssh_client = SshCliWrapper(key_pair["private"], host)
    server = run_dockerized_server(host, key_pair["public"].read_text())
    time.sleep(2)

    try:
        which_executable = ssh_client.exec("which", executable)
        assert which_executable.decode().strip().split("/")[-1] == executable
    finally:
        server.stop()


def test_client_can_forward_port_from_server(key_pair):
    # https://it-tools.tech/random-port-generator
    local_port = 63752
    remote_port = 24464
    assert_free(local_port)
    assert_free(remote_port)
    log.info("Forwarding remote port %s --> local port %s", remote_port, local_port)

    host = SshHost("localhost", 2222, "dan")
    KnownHostsClient().reset(host)

    private_key = key_pair["private"]
    public_key = key_pair["public"]

    assert private_key.is_file()
    assert public_key.is_file()

    request_path = f"/{uuid4()}"
    response_body = f"\nHi from SSH server: {uuid4()}\n"

    ssh_cli = SshCliWrapper(private_key, host)
    netcat = NetcatClient()
    netcat_command = netcat.ssh_exec(response_body, remote_port)

    with dockerized_server_safe(host, public_key.read_text(), [remote_port]):
        # TODO: find polling mechanism.
        time.sleep(3)
        port_forward_proc = ssh_cli.forward(local_port, remote_port)
        netcat_proc = ssh_cli.exec(*netcat_command)

        try:
            # TODO: find polling mechanism for finding out if netcat is ready.
            time.sleep(1)
            response = requests.get(f"http://{host.host}:{local_port}{request_path}")
            log.info(
                f"Response status {response.status_code}\nResponse text: {response.text}"
            )

        finally:
            # TODO: handle timeouts if these hang.
            show_output(port_forward_proc)
            show_output(netcat_proc)


#     random_str = f"Hi from SSH server: {uuid4()}"
#     log.info("Random string: `%s`", random_str)


#     ssh_client = SshCliWrapper(key_pair["private"], host)
#     server = run_dockerized_server(
#         host, key_pair["public"].read_text(), ports=[destination_port]
#     )
#     time.sleep(2)

#     try:
#         proc = ssh_client.forward(local_port, destination_port)
#         # https://unix.stackexchange.com/q/289364
#         # https://superuser.com/a/115556
#         # https://stackoverflow.com/a/19139134/

#         raw_response = """HTTP/1.1 200 OK
# Content-Type: text/plain
# Connection: close
# content-length: 57

# Hi from ssh server 50ca245d-a96e-47b1-a042-180dd751ee77
# """

#         #  | nc -l localhost 24464

#         command = (
#             f"""
# echo -e "{raw_response}" | nc -l localhost 24464
# """.strip()
#             .encode("unicode_escape")
#             .decode()
#         )

#         log.warning("command: %s", command)

#         # output = subprocess.check_output(
#         #         [
#         #             "ssh",
#         #             "-i",
#         #             "~/.ssh/id_rsa_16505ade1dbd42f38623fd2aef236a27",
#         #             "-p",
#         #             "2222",
#         #             "-o",
#         #             "StrictHostKeyChecking=accept-new",
#         #             "dan@localhost",
#         #             "/bin/bash",
#         #             "-c",
#         #             shlex.quote(command),
#         #         ]
#         #     )

#         # response_text = f"\nHi from ssh server {random_str}\n"
#         # response_whole = f"HTTP/1.1 200 OK\nContent-Type: text/plain\nConnection: close\ncontent-length: {len(response_text)}\n{response_text}"

#         # command = f'echo -e "{response_whole}" | nc -l localhost {destination_port}'

#         # TODO: why escaping? https://stackoverflow.com/a/15392758/
#         # command = command.encode("unicode_escape").decode()
#         # log.warning("Command: %s", command)
#         import shlex

#         # command_quoted = shlex.quote(command)
#         # log.warning("Quoted command: %s", command_quoted)

#         nc = ssh_client.exec_background("bash", "-c", shlex.quote(command))
#         import requests

#         try:
#             response = requests.get(f"http://localhost:{local_port}/foo", timeout=2)
#             log.warning("Status `%s`, text `%s`", response.status_code, response.text)

#         except Exception as e:
#             log.warning("Exception: %s", str(e))

#         # TODO: ssh_client.exec --> "process" variant
#         # ( https://github.com/dkmiller/tidbits/blob/main/2023/kubernetes/src/ptah/core/process.py )
#         # touch,
#         # run nc remotely ( https://unix.stackexchange.com/a/715981 )
#         # (sadly, no python or python3 + no screen)
#         # requests.get from local
#     finally:
#         server_logs = server.logs()
#         log.warning("Server logs: %s", server_logs)

#         server.stop()
#         proc.kill()
#         nc.kill()

#     raise Exception("boo!")
