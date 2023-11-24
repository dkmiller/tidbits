from ssh.port import assert_free


def test_key_pair(key_pair):
    assert key_pair.private.is_file()
    assert ".ssh" in str(key_pair.private)
    # TODO: use https://security.stackexchange.com/a/256118 and https://stackoverflow.com/a/1861970/
    # for a human-readable checking of file permissions.
    assert key_pair.private.stat().st_mode == 0o100600

    assert key_pair.public.is_file()
    assert ".ssh" in str(key_pair.public)
    assert key_pair.public.suffix == ".pub"
    assert key_pair.public.stat().st_mode == 0o100600


def test_ports(ports):
    assert isinstance(ports.local, int)
    assert isinstance(ports.remote, int)

    assert_free(ports.local)
    assert_free(ports.remote)
