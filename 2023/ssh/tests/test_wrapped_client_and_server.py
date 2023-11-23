import logging
import time
from uuid import uuid4

import pytest

from ssh import SshCliWrapper, SshHost, run_dockerized_server

log = logging.getLogger(__name__)


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
    # interesting = []
    # def record(*variable_names):
    #     for variable_name in variable_names:
    #         # TODO: this is a super gross way of debugging https://stackoverflow.com/a/9437799/
    #         interesting.append({variable_name: locals()[variable_name]})

    # https://it-tools.tech/random-port-generator
    local_port = 63752
    destination_port = 24464
    log.warning("local: %s, destination: %s", local_port, destination_port)

    # 50ca245d-a96e-47b1-a042-180dd751ee77
    random_str = str(uuid4())
    log.warning("Random string: %s", random_str)

    host = SshHost("localhost", 2222, "dan")
    ssh_client = SshCliWrapper(key_pair["private"], host)
    server = run_dockerized_server(
        host, key_pair["public"].read_text(), ports=[destination_port]
    )
    time.sleep(2)

    try:
        proc = ssh_client.forward(local_port, destination_port)
        # https://unix.stackexchange.com/q/289364
        # https://superuser.com/a/115556
        # https://stackoverflow.com/a/19139134/

        raw_response = """HTTP/1.1 200 OK
Content-Type: text/plain
Connection: close
content-length: 57

Hi from ssh server 50ca245d-a96e-47b1-a042-180dd751ee77
"""

        #  | nc -l localhost 24464

        command = (
            f"""
echo -e "{raw_response}" | nc -l localhost 24464
""".strip()
            .encode("unicode_escape")
            .decode()
        )

        log.warning("command: %s", command)

        # output = subprocess.check_output(
        #         [
        #             "ssh",
        #             "-i",
        #             "~/.ssh/id_rsa_16505ade1dbd42f38623fd2aef236a27",
        #             "-p",
        #             "2222",
        #             "-o",
        #             "StrictHostKeyChecking=accept-new",
        #             "dan@localhost",
        #             "/bin/bash",
        #             "-c",
        #             shlex.quote(command),
        #         ]
        #     )

        # response_text = f"\nHi from ssh server {random_str}\n"
        # response_whole = f"HTTP/1.1 200 OK\nContent-Type: text/plain\nConnection: close\ncontent-length: {len(response_text)}\n{response_text}"

        # command = f'echo -e "{response_whole}" | nc -l localhost {destination_port}'

        # TODO: why escaping? https://stackoverflow.com/a/15392758/
        # command = command.encode("unicode_escape").decode()
        # log.warning("Command: %s", command)
        import shlex

        # command_quoted = shlex.quote(command)
        # log.warning("Quoted command: %s", command_quoted)

        nc = ssh_client.exec_background("bash", "-c", shlex.quote(command))
        import requests

        try:
            response = requests.get(f"http://localhost:{local_port}/foo", timeout=2)
            log.warning("Status `%s`, text `%s`", response.status_code, response.text)

        except Exception as e:
            log.warning("Exception: %s", str(e))

        # TODO: ssh_client.exec --> "process" variant
        # ( https://github.com/dkmiller/tidbits/blob/main/2023/kubernetes/src/ptah/core/process.py )
        # touch,
        # run nc remotely ( https://unix.stackexchange.com/a/715981 )
        # (sadly, no python or python3 + no screen)
        # requests.get from local
    finally:
        server_logs = server.logs()
        log.warning("Server logs: %s", server_logs)

        server.stop()
        proc.kill()
        nc.kill()

    raise Exception("boo!")
