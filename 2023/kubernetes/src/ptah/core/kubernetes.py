import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List

from injector import inject
from rich import print

from ptah.core.image import ImageDefinition
from ptah.core.shell import ShellClient


@inject
@dataclass
class KubernetesClient:
    """
    TODO: proper way of passing source and target information around.
    """

    image_definitions: List[ImageDefinition]
    shell: ShellClient

    def build(self, source: str, target: str) -> None:
        try:
            shutil.rmtree(target)
        except FileNotFoundError:
            pass

        yamls = list(Path(source).rglob("*.yaml"))

        for yaml in yamls:
            content = yaml.read_text()

            # TODO: proper way of detecting Kubernetes specs.
            if "spec:" not in content and "metadata:" not in content:
                continue

            for image in self.image_definitions:
                content = content.replace(f"{image.name}:${{ptah}}", image.uri)

            relative = str(yaml.relative_to(source))
            target_path = Path(target) / relative
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(content)

    def apply(self, target: str) -> None:
        # https://stackoverflow.com/a/59493623
        output = self.shell("kubectl", "apply", "-R", "-f", target)
        watch = []
        skip = 0
        # TODO: stream events:
        # https://stackoverflow.com/a/51931477/2543689
        # kubectl get events --field-selector involvedObject.name=ui-deployment-7458788c98-szpxt
        for line in output.splitlines():
            resource, status = line.split(maxsplit=1)
            if resource.startswith("deployment.") and status != "unchanged":
                # https://linuxhint.com/kubectl-list-deployments/
                # kubectl rollout status deployment.apps/ui-deployment
                watch.append(["kubectl", "rollout", "status", resource])
            else:
                skip += 1

        msg = f"Watching {len(watch)} Kubernetes resources"
        if skip:
            msg += f" ({skip} unchanged)"
        print(msg)

        for w in watch:
            self.shell.run(w)
