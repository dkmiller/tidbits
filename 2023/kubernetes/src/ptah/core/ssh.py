import os
from dataclasses import dataclass

from injector import inject
from rich import print

from ptah.core.shell import ShellClient


@inject
@dataclass(frozen=True)
class Ssh:
    shell: ShellClient

    def start(self, pod: str):
        """
        Follows:

        - https://stackoverflow.com/a/55897287
        - https://stackoverflow.com/a/52691455

        TODO: similarity-based pod identification.
        - https://kubernetes.io/docs/concepts/overview/working-with-objects/labels/#list-and-watch-filtering
        """
        pods = self.shell(
            "kubectl", "get", "pods", "--no-headers", "-o", "custom-columns=:metadata.name"
        )
        lines = pods.splitlines()
        pod_name = [l for l in lines if l.startswith(f"{pod}-")][0]
        command = f"kubectl exec -it {pod_name} -- /bin/bash"
        print(f"Running command\n\n\t{command}\n")
        os.system(command)
