from dataclasses import dataclass

from injector import inject
from fastapi import HTTPException, status

from server.models import Workspace
from server.variants import Variants


@inject
@dataclass
class Manifest:
    variants: Variants

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

    def namespace(self, workspace: Workspace) -> str:
        # TODO: make this configurable.
        return "default"

    def pod_spec(self, workspace: Workspace) -> dict:
        """
        Inspired by:
        https://github.com/dkmiller/tidbits/blob/facb960704671729abfc361284d7a017bc2054a9/2023/kubernetes/examples/e2e/pods/api.yaml
        """

        if not (variant := self.variants.resolve(workspace)):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Unknown {workspace.variant=}")

        return {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": workspace.id,
                "labels": {"app": workspace.id},
            },
            "spec": {
                "containers": [
                    {
                        "image": variant.docker_image,
                        "name": "workspace",
                        "args": variant.container_args,
                        "ports": [{"containerPort": workspace.port}],
                        "readinessProbe": {
                            "httpGet": {
                                "path": variant.readiness,
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
