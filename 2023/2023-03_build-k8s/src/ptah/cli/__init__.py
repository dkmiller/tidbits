import webbrowser

import pyperclip
import typer
from injector import Injector
from rich import print

from ptah.core import (
    CacheClient,
    DockerClient,
    ImageClient,
    KubernetesClient,
    ShellClient,
)


def _injector(src: str, output: str):
    return Injector([CacheClient(), ImageClient(src)], auto_bind=True)


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
    k8s = injector.get(KubernetesClient)
    docker.build()
    k8s.build(src, output)


@app.command()
def ship(src: str = ".", output: str = ".build"):
    """
    Publishes relevant Docker images to the appropriate feed; then ship any
    Kubernetes changes.
    """
    build(src, output)

    injector = _injector(src, output)
    docker = injector.get(DockerClient)
    k8s = injector.get(KubernetesClient)

    docker.push()
    k8s.apply(output)


@app.command()
def dash(namespace: str = "kubernetes-dashboard", user: str = "admin-user"):
    """
    Obtain the appropriate auth token, then open the Kubernetes dashboard with that token copied to
    the clipboard.
    """
    injector = _injector(None, None)  # type: ignore
    shell = injector.get(ShellClient)
    token = shell("kubectl", "-n", namespace, "create", "token", user)
    pyperclip.copy(token)
    url = f"http://localhost:8001/api/v1/namespaces/{namespace}/services/https:{namespace}:/proxy/"
    webbrowser.open(url)
