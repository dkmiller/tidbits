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
        shutil.rmtree(target)
        yamls = list(Path(source).rglob("*.yaml"))

        for yaml in yamls:
            content = yaml.read_text()

            for image in self.image_definitions:
                content = content.replace(f"{image.name}:${{ptah}}", image.uri)

            relative = str(yaml.relative_to(source))
            target_path = Path(target) / relative
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(content)
