import time
from uuid import uuid4
import pytest

from ssh import SshCliWrapper, SshHost, run_dockerized_server


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
    host = SshHost("localhost", 2222, "dan")
    ssh_client = SshCliWrapper(key_pair["private"], host)
    server = run_dockerized_server(host, key_pair["public"].read_text())
    time.sleep(2)

    # ssh -L local_port:destination_server_ip:remote_port ssh_server_hostname

    # https://it-tools.tech/random-port-generator
    local_port = 24464
    destination_port = 63752

    try:
        proc = ssh_client.forward(local_port, destination_port)
        proc.kill()

        # TODO: ssh_client.exec --> "process" variant
        # ( https://github.com/dkmiller/tidbits/blob/main/2023/kubernetes/src/ptah/core/process.py )
        # touch,
        # run nc remotely ( https://unix.stackexchange.com/a/715981 )
        # (sadly, no python or python3 + no screen)
        # requests.get from local
    finally:
        server.stop()
