from dataclasses import dataclass, field
from typing import Iterable

from docker import DockerClient as ExtDockerClient
from docker import from_env
from docker.errors import ImageNotFound
from injector import inject
from rich import print
from rich.console import Console

from ptah.core.image import ImageDefinition

console = Console()


@inject
@dataclass(frozen=True)
class DockerClient:
    image_definitions: Iterable[ImageDefinition]
    _client: ExtDockerClient = field(default_factory=from_env)

    def build(self) -> None:
        defs = list(self.image_definitions)
        already_built = 0
        need_to_build = []
        for image in defs:
            try:
                self._client.images.get(image.uri)
                already_built += 1
            except ImageNotFound:
                need_to_build.append(image)

            # TODO: handle when Docker daemon is off.

        msg = f"Building {len(need_to_build)} Docker images"
        if already_built:
            msg += f" ({already_built} already built)"
        print(msg)

        for image in need_to_build:
            with console.status(f"Building {image.uri}"):
                path = str(image.location.parent.absolute())
                self._client.images.build(dockerfile=image.location.name, path=path, tag=image.uri)
