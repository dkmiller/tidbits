import os

from fabric import Config, Connection
from paramiko.config import SSHConfig


def test_key_pair(key_pair):
    assert "private" in key_pair
    assert "public" in key_pair

    assert os.path.isfile(key_pair["private"])
    assert os.path.isfile(key_pair["public"])

    assert ".ssh" in str(key_pair["private"])
    assert ".ssh" in str(key_pair["public"])


def test_ssh():

    user = "dan"
    idfile = "/Users/dan/.ssh/id_rsa_2df17ced400c4b399a6294b5ec742870"

    # https://phoenixnap.com/kb/ssh-config
    ssh_conf = SSHConfig.from_text(
        f"""
    Host localhost
        HostName localhost
        User {user}
        IdentityFile {idfile}
    """
    )

    # cfg = Config(ssh_config_path=ssh["private"])
    cfg = Config(ssh_config=ssh_conf)
    # cfg = Config(overrides={"authentication": {"identities": ["/Users/dan/.ssh/id_rsa_2df17ced400c4b399a6294b5ec742870"]}})

    # ssh -i ~/.ssh/id_rsa_168278178fde4c068391ca0269fab057 dan@localhost -p 2222

    # https://github.com/fabric/fabric/issues/2071
    result = Connection("localhost", user=user, port=2222, config=cfg).run(
        "uname -s", hide=True
    )

    assert result.exited == 0
    assert result.stdout.strip().lower() == "linux"

    # 'stdout': 'Linux\n

    # https://askubuntu.com/a/660556
    # https://stackoverflow.com/a/5255550/
    # https://kb.iu.edu/d/aews
    # ssh -i /Users/dan/.ssh/id_rsa_168278178fde4c068391ca0269fab057 localhost -p 52946 -o ConnectTimeout=1


# authentication: Authentication-related options.

#         identities
