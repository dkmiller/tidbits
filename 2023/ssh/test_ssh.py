import os


def test_key_pair(key_pair):
    assert "private" in key_pair
    assert "public" in key_pair

    assert os.path.isfile(key_pair["private"])
    assert os.path.isfile(key_pair["public"])

    assert ".ssh" in str(key_pair["private"])
    assert ".ssh" in str(key_pair["public"])


def test_ssh(ssh):
    from fabric import Config, Connection
    from invoke.config import Config as IC
    cfg = Config(overrides={"authentication": {"identities": [ssh["private"]]}})

    # https://askubuntu.com/a/660556
    # https://stackoverflow.com/a/5255550/
    # https://kb.iu.edu/d/aews
    # ssh -i /Users/dan/.ssh/id_rsa_168278178fde4c068391ca0269fab057 localhost -p 52946 -o ConnectTimeout=1

# authentication: Authentication-related options.

#         identities