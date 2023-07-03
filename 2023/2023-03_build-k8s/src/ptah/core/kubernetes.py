import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List

from injector import inject

from ptah.core.image import ImageDefinition


@inject
@dataclass
class KubernetesClient:
    image_definitions: List[ImageDefinition]

    def build(self, source: str, target: str) -> None:
        try:
            shutil.rmtree(target)
        except FileNotFoundError:
            pass

        yamls = list(Path(source).rglob("*.yaml"))

        for yaml in yamls:
            content = yaml.read_text()
            # TODO: proper way of detecting Kubernetes specs.
            if "spec:" not in content:
                continue

            for image in self.image_definitions:
                content = content.replace(f"{image.name}:${{ptah}}", image.uri)

            relative = str(yaml.relative_to(source))
            target_path = Path(target) / relative
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(content)

    def apply(self, target: str) -> None:
        pass


# # https://stackoverflow.com/a/59493623
# kubectl apply -R -f k8s

# TODO: how to wait till deployment completes?
