import asyncio
from dataclasses import dataclass

from injector import inject
from kubernetes_asyncio import config, utils
from kubernetes_asyncio.client import ApiClient, AppsV1Api, CoreV1Api

from server.manifest import Manifest
from server.models import Workspace

config.load_incluster_config()


@inject
@dataclass
class K8s:
    api: ApiClient
    manifest: Manifest

    async def get_pod(self, workspace: Workspace):
        ret = await CoreV1Api(self.api).list_pod_for_all_namespaces()
        pods = [p for p in ret.items if p.metadata.labels.get("app") == workspace.id]
        if pods:
            return pods[0]
        return None

    async def create(self, workspace: Workspace):
        objects = await utils.create_from_dict(
            self.api,
            self.manifest.spec(workspace),
            namespace=self.manifest.namespace(workspace),
        )
        print(f"Created {len(objects)} objects")

    async def delete(self, workspace: Workspace):
        await asyncio.gather(
            # https://stackoverflow.com/a/74642309
            AppsV1Api(self.api).delete_namespaced_deployment(
                self.manifest.deployment_name(workspace),
                namespace=self.manifest.namespace(workspace),
            ),  # type: ignore
            CoreV1Api(self.api).delete_namespaced_service(
                self.manifest.service_name(workspace),
                namespace=self.manifest.namespace(workspace),
                propagation_policy="Foreground",
            ),  # type: ignore
        )
