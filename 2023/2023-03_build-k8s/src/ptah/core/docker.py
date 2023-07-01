import subprocess
from dataclasses import dataclass, field
from typing import List

from cachelib import FileSystemCache
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
    cache: FileSystemCache
    image_definitions: List[ImageDefinition]
    _client: ExtDockerClient = field(default_factory=from_env)

    def build(self) -> None:
        already_built = 0
        need_to_build = []
        for image in self.image_definitions:
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

    def _push(self, images: ImageDefinition) -> None:
        # kind load docker-image api:0.0.1 ui:0.0.4
        # TODO: handle pushing to a remote registry.
        # https://codeberg.org/hjacobs/pytest-kind/src/branch/main/pytest_kind/cluster.py

        pass

    def push(self) -> None:
        push = []
        skip = 0
        for image in self.image_definitions:
            if self.cache.has(image.uri):
                skip += 1
            else:
                push.append(image)

        uris = [i.uri for i in push]

        msg = f"Pushing {len(uris)} images"
        if skip:
            msg += f" ({skip} already pushed)"
        print(msg)

        if push:
            with console.status(f"Pushing {uris}"):
                # TODO: redirect output in a nicer way.
                # https://codeberg.org/hjacobs/pytest-kind/src/branch/main/pytest_kind/cluster.py
                subprocess.run(["kind", "load", "docker-image"] + uris, check=True)
                for uri in uris:
                    self.cache.add(uri, "any")
