from fastapi import Depends
from kubernetes_asyncio import config
from kubernetes_asyncio.client import ApiClient, CoreV1Api

from server.models import Workspace


config.load_incluster_config()


def pod_spec(workspace: Workspace) -> tuple[dict, str]:
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
                    "image": "busybox",
                    "name": "sleep",
                    "args": ["/bin/sh", "-c", "while true; do date; sleep 5; done"],
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
