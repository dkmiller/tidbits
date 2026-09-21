import logging
from pathlib import Path

import paramiko
import typer

from ssh import OpensshDockerWrapper, ParamikoServer, SshHost
from ssh.rsa import private_public_key_pair

# https://stackoverflow.com/a/76375308/
app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def rsa():
    pair = private_public_key_pair()
    print(
        f"""
Private key: {pair.private.absolute()}

Public key: {pair.public.absolute()}

(Separate)

python -m http.server 12345 --directory $PWD

ssh-testing serve "$(cat {pair.public.absolute()})" {pair.private.absolute()}

ssh-testing serve-docker "$(cat {pair.public.absolute()})"

ssh -F /dev/null -o StrictHostKeyChecking=accept-new -i {pair.private} -p 2222 -N -L 12346:localhost:12345 dan@localhost

http://localhost:12346/
"""
    )


@app.command()
def serve(
    public_key: str,
    private_key: str,
    user: str = "dan",
    host: str = "localhost",
    port: int = 2222,
):
    server = ParamikoServer(
        SshHost(host, port, user), public_key, [], private_key=Path(private_key)
    )

    logging.basicConfig(level="INFO")

    paramiko.util.log_to_file("demo_server.log")

    server.run()


@app.command()
def serve_docker(
    public_key: str,
    user: str = "dan",
    host: str = "localhost",
    port: int = 2222,
    forward: int = 12346,
):
    server = OpensshDockerWrapper(SshHost(host, port, user), public_key, [forward])

    logging.basicConfig(level="INFO")

    with server.serve():
        import time

        time.sleep(120)
