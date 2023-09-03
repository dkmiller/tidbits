import typer
from injector import Injector
from rich import print

import ptah.core as pc


def _injector(src: str, output: str):
    return Injector([pc.CacheClient(), pc.ImageClient(src)], auto_bind=True)


app = typer.Typer()


@app.command()
def build(src: str = ".", output: str = ".build"):
    """
    Build any relevant Docker images and the corresponding Kubernetes
    manifests.
    """
    print(f"Building [bold]{src}[/bold] ↦ [bold]{output}[/bold]")

    injector = _injector(src, output)
    docker = injector.get(pc.DockerClient)
    helm = injector.get(pc.Helm)
    k8s = injector.get(pc.KubernetesClient)
    docker.build()
    k8s.build(src, output)
    helm.build(src)


@app.command()
def ship(src: str = ".", output: str = ".build"):
    """
    Publishes relevant Docker images to the appropriate feed; then ship any
    Kubernetes changes.
    """
    build(src, output)

    injector = _injector(src, output)
    docker = injector.get(pc.DockerClient)
    helm = injector.get(pc.Helm)
    k8s = injector.get(pc.KubernetesClient)

    helm.apply(src)
    docker.push()
    k8s.apply(output)


@app.command()
def dash(namespace: str = "kubernetes-dashboard", user: str = "admin-user", grafana: bool = False):
    """
    Obtain the appropriate auth token, then open the Kubernetes dashboard with that token copied to
    the clipboard.
    """
    injector = _injector(None, None)  # type: ignore

    # Make sure ports are being forwarded.
    fwd = injector.get(pc.Forward)
    fwd.ensure()

    dashboard = injector.get(pc.Dashboard)
    if grafana:
        dashboard.grafana("default")
    else:
        dashboard.spawn(namespace, user)


@app.command()
def ssh(pod: str):
    """
    Obtains the pod with given label and opens an SSH session with it.
    """
    injector = _injector(None, None)  # type: ignore

    _ssh = injector.get(pc.Ssh)
    _ssh.start(pod)


@app.command()
def forward(kill: bool = False):
    """
    Non-blocking command searches for relevant processes, kills them, then
    Use `Popen` from non-blocking command to search for relevant processes,
    then spawn `os.system(kubectl...)` inside a retry block + infinite loop.

    `ptah forward --kill` shuts down all the port-forwards.
    """
    injector = _injector(None, None)  # type: ignore
    fwd = injector.get(pc.Forward)

    if kill:
        fwd.terminate()
    else:
        fwd.ensure()
