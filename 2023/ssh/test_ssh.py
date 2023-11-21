import os
import time

import pytest
from fabric import Config, Connection
from paramiko.config import SSHConfig

from ssh import SshHost, private_public_key_pair, run_dockerized_server


def test_key_pair(key_pair):
    assert "private" in key_pair
    assert "public" in key_pair

    assert os.path.isfile(key_pair["private"])
    assert os.path.isfile(key_pair["public"])

    assert ".ssh" in str(key_pair["private"])
    assert ".ssh" in str(key_pair["public"])


def test_ssh_with_dockerized_server():
    ssh_host = SshHost("localhost", 2222, "dan")
    private, public = private_public_key_pair()
    container = run_dockerized_server(ssh_host, public.read_text())
    time.sleep(3)

    # https://phoenixnap.com/kb/ssh-config
    ssh_conf = SSHConfig.from_text(
        f"""
    Host localhost
        HostName localhost
        User {ssh_host.user}
        IdentityFile {str(private)}
    """
    )

    config = Config(ssh_config=ssh_conf)
    conn = Connection(
        ssh_host.host, user=ssh_host.user, port=ssh_host.port, config=config
    )

    # https://github.com/fabric/fabric/issues/2071
    result = conn.run("uname -s", hide=True)

    assert result.exited == 0
    assert result.stdout.strip().lower() == "linux"

    result = conn.run("whoami", hide=True)

    assert result.exited == 0
    assert result.stdout.strip().lower() == ssh_host.user

    container.stop()


@pytest.mark.skip(reason="This should be used to validate separately started container")
def test_ssh():
    host = "localhost"
    idfile = "/Users/dan/.ssh/id_rsa_eb6ddc9dc80d4518a36535870040b1cd"
    user = "dan"
    port = 2222

    # https://phoenixnap.com/kb/ssh-config
    ssh_conf = SSHConfig.from_text(
        f"""
    Host localhost
        HostName localhost
        User {user}
        IdentityFile {idfile}
    """
    )

    config = Config(ssh_config=ssh_conf)
    conn = Connection(host, user=user, port=port, config=config)

    # https://github.com/fabric/fabric/issues/2071
    result = conn.run("uname -s", hide=True)

    assert result.exited == 0
    assert result.stdout.strip().lower() == "linux"

    result = conn.run("whoami", hide=True)

    assert result.exited == 0
    assert result.stdout.strip().lower() == user
