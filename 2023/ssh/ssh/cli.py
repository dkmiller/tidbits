import logging
from pathlib import Path

import paramiko
import typer

from ssh import ParamikoServer, SshHost
from ssh.rsa import private_public_key_pair

# https://stackoverflow.com/a/76375308/
app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def rsa():
    pair = private_public_key_pair()
    print(
        # Ignore ~/.ssh/config:
        # https://www.cyberciti.biz/faq/tell-ssh-to-exclude-ignore-config-file/
        f"""
Private key: {pair.private.absolute()}

Public key: {pair.public.absolute()}

ssh-testing serve "$(cat {pair.public.absolute()})" {pair.private.absolute()}

ssh -F /dev/null -o StrictHostKeyChecking=accept-new -i {pair.private} -p 2222 dan@localhost
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
