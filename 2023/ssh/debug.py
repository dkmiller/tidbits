import logging
import time
from pathlib import Path
from uuid import uuid4

import requests

from ssh import SshCliWrapper, SshHost
from ssh.known_hosts import KnownHostsClient
from ssh.netcat import NetcatClient

logging.basicConfig(level="INFO")

host = SshHost("localhost", 2222, "dan")

KnownHostsClient().reset(host)


local_port = 63752
remote_port = 24464
private_key = Path.home().resolve() / ".ssh/id_rsa_16505ade1dbd42f38623fd2aef236a27"
request_path = f"/{uuid4()}"

response_body = f"\nHi from SSH server: {uuid4()}\n"

ssh_cli = SshCliWrapper(private_key, host)


port_forward_proc = ssh_cli.forward(local_port, remote_port)

netcat_proc = ssh_cli.exec(*NetcatClient().ssh_exec(response_body, remote_port))

try:
    time.sleep(1)
    response = requests.get(f"http://{host.host}:{local_port}{request_path}")
    print(f"Response status {response.status_code}\nResponse text: {response.text}")
except Exception as e:
    print(e)


def show_output(proc):
    # https://stackoverflow.com/a/38956698/
    (output, err) = proc.communicate()

    # This makes the wait possible
    p_status = proc.wait()
    print(f"----- output -----\n{output}\n----------")


show_output(netcat_proc)
show_output(port_forward_proc)
