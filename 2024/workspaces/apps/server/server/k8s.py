from fastapi import Depends
from kubernetes_asyncio import config
from kubernetes_asyncio.client import ApiClient, AppsV1Api, CoreV1Api

from server.models import Workspace
from server._docker import docker_images


config.load_incluster_config()


def args(workspace: Workspace):
    match workspace.image_alias:
        case "jupyterlab":
            return [
                "jupyter",
                "lab",
                "--allow-root",
                "--ip=0.0.0.0",
                f"--port={workspace.port}",
                "--IdentityProvider.token=''",
            ]
        case "vscode":
            return [
                "code-server",
                "--bind-addr",
                f"0.0.0.0:{workspace.port}",
                "--auth",
                "none",
            ]
        case default:
            raise RuntimeError(f"Image alias {default} not supported.")


def service_spec(workspace: Workspace) -> dict:
    return {
        "kind": "Service",
        "apiVersion": "v1",
        "metadata": {
            "name": f"{workspace.id}-service",
        },
        "spec": {
            "selector": {"app": workspace.id},
            "ports": [{"port": workspace.port}],
        },
    }


def pod_spec(workspace: Workspace) -> tuple[dict, str]:
    """
    Inspired by:
    https://github.com/dkmiller/tidbits/blob/facb960704671729abfc361284d7a017bc2054a9/2023/kubernetes/examples/e2e/pods/api.yaml
    """
    image_mapping = docker_images()

    pod = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": workspace.id,
            "labels": {"app": workspace.id, "workspace.name": workspace.name},
        },
        "spec": {
            "containers": [
                {
                    "image": image_mapping[workspace.image_alias],
                    "name": "workspace",
                    "args": args(workspace),
                    "ports": [{"containerPort": workspace.port}],
                }
            ]
        },
    }

    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": f"{workspace.id}-deployment"},
        "labels": {"app": workspace.id},
        "spec": {
            "replicas": 1,
            "selector": {"matchLabels": {"app": workspace.id}},
            "template": pod,
        },
    }

    svc = service_spec(workspace)

    return {"kind": "List", "items": [deployment, svc]}, "default"


async def api_client():
    # https://github.com/tomplus/kubernetes_asyncio?tab=readme-ov-file#example
    async with ApiClient() as api:
        yield api


async def v1_api(api: ApiClient = Depends(api_client)):
    # https://fastapi.tiangolo.com/tutorial/dependencies/dependencies-with-yield/#a-database-dependency-with-yield
    yield CoreV1Api(api)


async def v1_apps(api: ApiClient = Depends(api_client)):
    # https://fastapi.tiangolo.com/tutorial/dependencies/dependencies-with-yield/#a-database-dependency-with-yield
    yield AppsV1Api(api)
