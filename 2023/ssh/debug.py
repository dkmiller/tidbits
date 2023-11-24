import shlex
import subprocess
import time
from pathlib import Path

import requests

from ssh import SshCliWrapper, SshHost
from ssh.known_hosts import KnownHostsClient

host = SshHost("localhost", 2222, "dan")

KnownHostsClient().reset(host)


local_port = 63752
remote_port = 24464
private_key = Path.home().resolve() / ".ssh/id_rsa_16505ade1dbd42f38623fd2aef236a27"


ssh_cli = SshCliWrapper(private_key, host)


port_forward_proc = ssh_cli.forward(local_port, remote_port)

# port_forward_output = subprocess.check_output(
#     [
#         "ssh",
#         "-i",
#         "~/.ssh/id_rsa_16505ade1dbd42f38623fd2aef236a27",
#         "-p",
#         str(host.port),
#         "-o",
#         "StrictHostKeyChecking=accept-new",
#         "-fN",
#         "-L",
#         f"{local_port}:{host.host}:{remote_port}",
#         f"{host.user}@{host.host}",
#     ]
# )

# print(f"----- port forward output -----\n{port_forward_output.decode()}\n----------")


# /bin/bash -c 'echo\ -e\ ""HTTP/1.1 200 OK\\nContent-Type: text/plain\nConnection: close\\ncontent-length: 57\\n\\nHi from ssh server 50ca245d-a96e-47b1-a042-180dd751ee77\\n"" \| nc -l localhost 24464'

raw_response = """HTTP/1.1 200 OK
Content-Type: text/plain
Connection: close
content-length: 57

Hi from ssh server 50ca245d-a96e-47b1-a042-180dd751ee77
"""

command = (
    f"""
echo -e "{raw_response}" | nc -l localhost {remote_port}
""".strip()
    .encode("unicode_escape")
    .decode()
)

print(f"----- command -----\n{command}\n----------")


netcat_proc = subprocess.Popen(
    [
        "ssh",
        "-i",
        str(private_key),
        "-p",
        str(host.port),
        "-o",
        "StrictHostKeyChecking=accept-new",
        f"{host.user}@{host.host}",
        "/bin/bash",
        "-c",
        shlex.quote(command),
    ]
)


try:
    time.sleep(1)
    print(requests.get(f"http://{host.host}:{local_port}/foo").text)
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

# # port_forward_proc


# print(f"----- output -----\n{output}\n----------")

# # port_forward_proc.kill()

# # https://stackoverflow.com/a/38956698/
# (output, err) = netcat_proc.communicate()

# # This makes the wait possible
# p_status = netcat_proc.wait()


# # port_forward_proc


# print(f"----- output -----\n{output}\n----------")
