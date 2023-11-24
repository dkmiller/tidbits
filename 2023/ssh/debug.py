import logging
import time
from pathlib import Path
from uuid import uuid4

import requests

from ssh import SshCliWrapper, SshHost, run_dockerized_server
from ssh.known_hosts import KnownHostsClient
from ssh.netcat import NetcatClient
from ssh.port import assert_free


def show_output(proc):
    # https://stackoverflow.com/a/38956698/
    (output, err) = proc.communicate()

    # This makes the wait possible
    p_status = proc.wait()
    print(f"----- output -----\n{output}\n----------")


logging.basicConfig(level="INFO")


host = SshHost("localhost", 2222, "dan")

KnownHostsClient().reset(host)


local_port = 63752
remote_port = 24464

private_key = Path.home().resolve() / ".ssh/id_rsa_16505ade1dbd42f38623fd2aef236a27"
public_key = Path.home().resolve() / ".ssh/id_rsa_16505ade1dbd42f38623fd2aef236a27.pub"
request_path = f"/{uuid4()}"


assert_free(local_port)
assert_free(remote_port)


ssh_server = run_dockerized_server(host, public_key.read_text(), [remote_port])

time.sleep(3)


response_body = f"\nHi from SSH server: {uuid4()}\n"

ssh_cli = SshCliWrapper(private_key, host)


try:
    port_forward_proc = ssh_cli.forward(local_port, remote_port)

    # ==================================================================================================

    netcat_proc = ssh_cli.exec(*NetcatClient().ssh_exec(response_body, remote_port))

    time.sleep(1)
    response = requests.get(f"http://{host.host}:{local_port}{request_path}")
    print(f"Response status {response.status_code}\nResponse text: {response.text}")

finally:
    ssh_server.stop()
    show_output(netcat_proc)
    show_output(port_forward_proc)
