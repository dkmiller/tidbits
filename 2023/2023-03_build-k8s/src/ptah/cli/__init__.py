import typer
from injector import Injector
from rich import print

from ptah.core import DockerClient, ImageClient


def _injector(src: str, output: str):
    return Injector([ImageClient(src)], auto_bind=True)


app = typer.Typer()


@app.command()
def build(src: str = ".", output: str = ".build"):
    """
    Build any relevant Docker images and the corresponding Kubernetes
    manifests.
    """
    print(f"Building [bold]{src}[/bold] ↦ [bold]{output}[/bold]")

    injector = _injector(src, output)
    docker = injector.get(DockerClient)
    docker.build()


@app.command()
def deploy(name: str, formal: bool = False):
    """
    Publishes relevant Docker images to the appropriate feed; then ship any
    Kubernetes changes.
    """
    if formal:
        print(f"Goodbye Ms. {name}. Have a good day.")
    else:
        print(f"Bye {name}!")
