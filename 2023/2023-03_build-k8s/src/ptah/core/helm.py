from dataclasses import dataclass
from pathlib import Path

from injector import inject
from rich import print

from ptah.core.shell import ShellClient


@inject
@dataclass
class Helm:
    shell: ShellClient

    def helmfile_exists(self, target) -> bool:
        helmfile = Path(target) / "helmfile.yaml"
        return helmfile.is_file()

    def build(self, target: str) -> None:
        if self.helmfile_exists(target):
            print("Syncing Helmfile")
            # https://dev.to/javidjms/helmfile-deploy-multiple-charts-in-your-cluster-k8s-422j
            self.shell("helmfile", "sync")

    def apply(self, target: str) -> None:
        if self.helmfile_exists(target):
            print("Applying Helmfile")
            # https://dev.to/javidjms/helmfile-deploy-multiple-charts-in-your-cluster-k8s-422j
            self.shell("helmfile", "apply")
