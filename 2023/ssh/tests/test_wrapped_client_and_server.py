import time

from ssh import SshCliWrapper, SshHost, run_dockerized_server


def test_client_can_touch_file_in_server(key_pair):
    host = SshHost("localhost", 2222, "dan")
    ssh_client = SshCliWrapper(key_pair["private"], host)
    server = run_dockerized_server(host, key_pair["public"].read_text())
    time.sleep(3)

    try:
        whoami = ssh_client.exec("whoami")
        assert whoami.decode().strip() == host.user
    finally:
        server.stop()
