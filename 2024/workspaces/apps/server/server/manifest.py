import os
from dataclasses import dataclass

from injector import inject

from server.models import Workspace
from server.variants import AbstractVariant


@inject
@dataclass
class Manifest:
    def deployment_name(self, workspace: Workspace) -> str:
        return f"{workspace.id}-deployment"

    def deployment_spec(self, workspace: Workspace) -> dict:
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": self.deployment_name(workspace)},
            "labels": {"app": workspace.id},
            "spec": {
                "replicas": 1,
                "selector": {"matchLabels": {"app": workspace.id}},
                "template": self.pod_spec(workspace),
            },
        }

    def docker_images(self) -> dict:
        raw = os.environ["WORKSPACE_DOCKER_IMAGES"]
        uris = [uri.strip() for uri in raw.split(" ") if uri]
        return {uri.split(":")[0]: uri for uri in uris}

    def namespace(self, workspace: Workspace) -> str:
        # TODO: make this configurable.
        return "default"

    def pod_spec(self, workspace: Workspace) -> dict:
        """
        Inspired by:
        https://github.com/dkmiller/tidbits/blob/facb960704671729abfc361284d7a017bc2054a9/2023/kubernetes/examples/e2e/pods/api.yaml
        """

        variant = AbstractVariant.variant(workspace.image_alias)

        return {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": workspace.id,
                "labels": {"app": workspace.id, "workspace.name": workspace.name},
            },
            "spec": {
                "containers": [
                    {
                        "image": self.docker_images()[workspace.image_alias],
                        "name": "workspace",
                        "args": variant.args(workspace.port),
                        "ports": [{"containerPort": workspace.port}],
                        "readinessProbe": {
                            "httpGet": {
                                "path": variant.readiness(),
                                "port": workspace.port,
                            }
                        },
                    }
                ]
            },
        }

    def service_name(self, workspace: Workspace) -> str:
        return f"{workspace.id}-service"

    def service_spec(self, workspace: Workspace) -> dict:
        return {
            "kind": "Service",
            "apiVersion": "v1",
            "metadata": {
                "name": self.service_name(workspace),
            },
            "spec": {
                "selector": {"app": workspace.id},
                "ports": [{"port": workspace.port}],
            },
        }

    def spec(self, workspace: Workspace) -> dict:
        deployment = self.deployment_spec(workspace)
        service = self.service_spec(workspace)
        return {"kind": "List", "items": [deployment, service]}
