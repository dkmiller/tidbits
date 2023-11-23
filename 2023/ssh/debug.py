import subprocess
import shlex


port_forward_output = subprocess.check_output(
        [
            "ssh",
            "-i",
            "~/.ssh/id_rsa_16505ade1dbd42f38623fd2aef236a27",
            "-p",
            "2222",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-fN",
            "-L",
            "63752:localhost:24464",
            "dan@localhost",
        ]
    )

print(f"----- output -----\n{port_forward_output.decode()}\n----------")



# /bin/bash -c 'echo\ -e\ ""HTTP/1.1 200 OK\\nContent-Type: text/plain\nConnection: close\\ncontent-length: 57\\n\\nHi from ssh server 50ca245d-a96e-47b1-a042-180dd751ee77\\n"" \| nc -l localhost 24464'

raw_response = """HTTP/1.1 200 OK
Content-Type: text/plain
Connection: close
content-length: 57

Hi from ssh server 50ca245d-a96e-47b1-a042-180dd751ee77
"""

#  | nc -l localhost 24464

command = f"""
echo -e "{raw_response}" | nc -l localhost 24464
""".strip().encode("unicode_escape").decode()

print(f"----- command -----\n{command}\n----------")


output = subprocess.check_output(
        [
            "ssh",
            "-i",
            "~/.ssh/id_rsa_16505ade1dbd42f38623fd2aef236a27",
            "-p",
            "2222",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "dan@localhost",
            "/bin/bash",
            "-c",
            shlex.quote(command),
        ]
    )

print(f"----- output -----\n{output.decode()}\n----------")


# curl -v http://localhost:24464/foo
