from fastapi import Depends
from kubernetes_asyncio import config
from kubernetes_asyncio.client import ApiClient, CoreV1Api

from server.models import Workspace
from server._docker import docker_images


config.load_incluster_config()


def args(workspace: Workspace):
    match workspace.image_alias:
        case "jupyterlab":
            return ["jupyter", "lab", "--allow-root", "--ip=0.0.0.0", f"--port={workspace.port}", "--IdentityProvider.token=''"]
        case "vscode":
            return ["code-server", "--bind-addr", f"0.0.0.0:{workspace.port}", "--auth", "none"]
        case default:
            raise RuntimeError(f"Image alias {default} not supported.")


def pod_spec(
        workspace: Workspace,
        ) -> tuple[dict, str]:
    image_mapping = docker_images()

    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": workspace.id,
            "labels": {"workspace.name": workspace.name},
        },
        "spec": {
            "containers": [
                {
                    "image": image_mapping[workspace.image_alias],
                    "name": "sleep",
                    "args": args(workspace),
                    "ports": [{"containerPort": workspace.port}],
                }
            ]
        },
    }, "default"


async def api_client():
    # https://github.com/tomplus/kubernetes_asyncio?tab=readme-ov-file#example
    async with ApiClient() as api:
        yield api


async def v1_api(api: ApiClient = Depends(api_client)):
    # https://fastapi.tiangolo.com/tutorial/dependencies/dependencies-with-yield/#a-database-dependency-with-yield
    yield CoreV1Api(api)
