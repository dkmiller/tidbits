from dataclasses import dataclass
from pathlib import Path

from injector import inject
from rich import print

from ptah.core.shell import ShellClient


@inject
@dataclass
class Helm:
    """
    Manage syncing and applying Helm charts via `helmfile.yaml`. Follows:
    - https://dev.to/javidjms/helmfile-deploy-multiple-charts-in-your-cluster-k8s-422j
    - https://helmfile.readthedocs.io/
    """

    shell: ShellClient

    def helmfile_exists(self, target) -> bool:
        helmfile = Path(target) / "helmfile.yaml"
        return helmfile.is_file()

    def build(self, target: str) -> None:
        if self.helmfile_exists(target):
            print("Syncing Helmfile")
            self.shell("helmfile", "sync")

    def apply(self, target: str) -> None:
        if self.helmfile_exists(target):
            print("Applying Helmfile")
            self.shell("helmfile", "apply")
