import typer

from ssh.rsa import private_public_key_pair

# https://stackoverflow.com/a/76375308/
app = typer.Typer(pretty_exceptions_enable=False)


def gen_rsa():
    private, _ = private_public_key_pair()
    hex = private.stem.split("_")[-1]
    print(hex)


@app.command()
def server(sha: str):
    import logging
    from pathlib import Path

    import paramiko

    from ssh.server import run_server

    logging.basicConfig(level="INFO")

    paramiko.util.log_to_file("demo_server.log")

    run_server("dan", 5555, Path.home().resolve() / ".ssh" / f"id_rsa_{sha}")


# TODO: why do I need to CTRL+C to exit after running this?
# ssh -i ~/.ssh/id_rsa_16505ade1dbd42f38623fd2aef236a27 -p 5555 dan@localhost my-command2
